"""
Baseline loss functions for comparison with CascadingCircleLoss.

All losses share the same interface:
    forward(sentence_features, labels, hard_relevency, soft_relevency) -> Tensor

Dataset layout:
    sentence_features[0]            = anchor (query)
    sentence_features[1]            = positive (ground-truth passage)
    sentence_features[2 : 2+H]      = hard negatives  (H passages)
    sentence_features[2+H : 2+H+S]  = soft negatives  (S graded passages)
    hard_relevency                  = (B, H) relevance scores for hard negatives
    soft_relevency                  = (B, S) relevance scores for soft negatives
"""
from __future__ import annotations

from typing import Dict, Iterable

import torch
import torch.nn.functional as F
from torch import nn


# ════════════════════════════════════════════════════════════════════════
# 1. InfoNCE Loss — positive vs hard negatives only
# ════════════════════════════════════════════════════════════════════════

class InfoNCELoss(nn.Module):
    """
    Standard InfoNCE (cross-entropy) loss.
    Treats the positive as the target and all hard negatives as distractors.
    Soft negatives / relevancy scores are ignored.

    Ref: van den Oord et al. 2018 — "Representation Learning with Contrastive
         Predictive Coding". scale=20 corresponds to τ=0.05, the standard
         sentence-transformers / text retrieval temperature.
    """

    def __init__(self, model, scale: float = 20.0):
        super().__init__()
        self.model = model
        self.scale = scale
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def forward(
        self,
        sentence_features: Iterable[Dict[str, torch.Tensor]],
        labels: torch.Tensor,
        hard_relevency: torch.Tensor,
        soft_relevency: torch.Tensor,
    ) -> torch.Tensor:
        embeddings = [self.model(f)["sentence_embedding"] for f in sentence_features]
        B = embeddings[0].size(0)
        device = embeddings[0].device

        anchors = F.normalize(embeddings[0], p=2, dim=-1)
        positives = F.normalize(embeddings[1], p=2, dim=-1)

        n_hard = hard_relevency.size(1)
        hard_negatives = torch.stack(embeddings[2:2 + n_hard], dim=1)
        hard_negatives = F.normalize(hard_negatives, p=2, dim=-1)

        pos_logit = torch.sum(anchors * positives, dim=-1, keepdim=True) * self.scale
        neg_logits = torch.einsum('bd,bnd->bn', anchors, hard_negatives) * self.scale
        logits = torch.cat([pos_logit, neg_logits], dim=1)

        target = torch.zeros(B, dtype=torch.long, device=device)
        return self.cross_entropy_loss(logits, target)


# ════════════════════════════════════════════════════════════════════════
# 2. Circle Loss (flat, original) — no cascade chain
# ════════════════════════════════════════════════════════════════════════

class CircleLoss(nn.Module):
    """
    Original Circle Loss (Sun et al., CVPR 2020).
    Treats positive as one group, all negatives (hard + soft) as another group.
    No sequential ordering — flat separation between positive and negatives.

    Ref: Sun et al. 2020 — γ=64 (scale factor), m=0.25 (relaxation margin).
    """

    def __init__(self, model, scale: float = 64.0, gamma: float = 64.0, margin: float = 0.25):
        super().__init__()
        self.model = model
        self.scale = scale
        self.gamma = gamma
        self.margin = margin
        self.soft_plus = nn.Softplus()

    def forward(
        self,
        sentence_features: Iterable[Dict[str, torch.Tensor]],
        labels: torch.Tensor,
        hard_relevency: torch.Tensor,
        soft_relevency: torch.Tensor,
    ) -> torch.Tensor:
        embeddings = [self.model(f)["sentence_embedding"] for f in sentence_features]
        B = embeddings[0].size(0)
        device = embeddings[0].device

        anchors = F.normalize(embeddings[0], p=2, dim=-1)
        positives = F.normalize(embeddings[1], p=2, dim=-1)

        n_hard = hard_relevency.size(1)
        n_soft = soft_relevency.size(1)

        hard_neg = torch.stack(embeddings[2:2 + n_hard], dim=1)
        hard_neg = F.normalize(hard_neg, p=2, dim=-1)
        soft_neg = torch.stack(embeddings[2 + n_hard:2 + n_hard + n_soft], dim=1)
        soft_neg = F.normalize(soft_neg, p=2, dim=-1)

        all_neg = torch.cat([hard_neg, soft_neg], dim=1)  # (B, H+S, D)

        # Cosine similarities
        sp = torch.sum(anchors * positives, dim=-1)  # (B,)
        sn = torch.einsum('bd,bnd->bn', anchors, all_neg)  # (B, H+S)

        # Optimal margins: O_p = 1 + margin, O_n = -margin
        O_p = 1.0 + self.margin
        O_n = -self.margin

        # Self-paced weights (detached)
        with torch.no_grad():
            alpha_p = F.relu(O_p - sp.detach())  # (B,)
            alpha_n = F.relu(sn.detach() - O_n)  # (B, H+S)

        # Delta margins
        delta_p = 1.0 - self.margin
        delta_n = self.margin

        # Circle loss logits
        logit_p = -self.gamma * alpha_p * (sp - delta_p)  # (B,)
        logit_n = self.gamma * alpha_n * (sn - delta_n)   # (B, H+S)

        loss = self.soft_plus(
            torch.logsumexp(logit_n, dim=1) + torch.logsumexp(logit_p.unsqueeze(1), dim=1)
        )
        return loss.mean()


# ════════════════════════════════════════════════════════════════════════
# 3. Triplet Loss — anchor, positive, hardest negative
# ════════════════════════════════════════════════════════════════════════

class TripletLoss(nn.Module):
    """
    Online triplet loss with hardest-negative mining.
    For each sample, picks the hardest hard negative (highest similarity to anchor).

    Ref: Schroff et al. 2015 (FaceNet) — margin=0.2.
    """

    def __init__(self, model, margin: float = 0.2):
        super().__init__()
        self.model = model
        self.margin = margin
        self.triplet_loss = nn.TripletMarginWithDistanceLoss(
            distance_function=lambda x, y: 1.0 - F.cosine_similarity(x, y),
            margin=margin,
        )

    def forward(
        self,
        sentence_features: Iterable[Dict[str, torch.Tensor]],
        labels: torch.Tensor,
        hard_relevency: torch.Tensor,
        soft_relevency: torch.Tensor,
    ) -> torch.Tensor:
        embeddings = [self.model(f)["sentence_embedding"] for f in sentence_features]

        anchors = F.normalize(embeddings[0], p=2, dim=-1)      # (B, D)
        positives = F.normalize(embeddings[1], p=2, dim=-1)    # (B, D)

        n_hard = hard_relevency.size(1)
        hard_neg = torch.stack(embeddings[2:2 + n_hard], dim=1)  # (B, H, D)
        hard_neg = F.normalize(hard_neg, p=2, dim=-1)

        # Mine hardest negative per sample
        sims = torch.einsum('bd,bnd->bn', anchors, hard_neg)    # (B, H)
        hardest_idx = sims.argmax(dim=1)                         # (B,)
        hardest_neg = hard_neg[torch.arange(anchors.size(0)), hardest_idx]  # (B, D)

        return self.triplet_loss(anchors, positives, hardest_neg)


# ════════════════════════════════════════════════════════════════════════
# 4. Multiple Negatives Ranking Loss — positive vs all negatives
# ════════════════════════════════════════════════════════════════════════

class MultipleNegativesRankingLoss(nn.Module):
    """
    Standard MNRL: InfoNCE over positive + all negatives (hard + soft).
    No relevance grading — all negatives are treated equally.

    Ref: sentence-transformers default scale=20 (τ=0.05).
         Originally inspired by Henderson et al. 2017.
    """

    def __init__(self, model, scale: float = 20.0):
        super().__init__()
        self.model = model
        self.scale = scale
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def forward(
        self,
        sentence_features: Iterable[Dict[str, torch.Tensor]],
        labels: torch.Tensor,
        hard_relevency: torch.Tensor,
        soft_relevency: torch.Tensor,
    ) -> torch.Tensor:
        embeddings = [self.model(f)["sentence_embedding"] for f in sentence_features]
        B = embeddings[0].size(0)
        device = embeddings[0].device

        anchors = F.normalize(embeddings[0], p=2, dim=-1)
        positives = F.normalize(embeddings[1], p=2, dim=-1)

        n_hard = hard_relevency.size(1)
        n_soft = soft_relevency.size(1)

        hard_neg = torch.stack(embeddings[2:2 + n_hard], dim=1)
        hard_neg = F.normalize(hard_neg, p=2, dim=-1)
        soft_neg = torch.stack(embeddings[2 + n_hard:2 + n_hard + n_soft], dim=1)
        soft_neg = F.normalize(soft_neg, p=2, dim=-1)

        all_neg = torch.cat([hard_neg, soft_neg], dim=1)  # (B, H+S, D)

        pos_logit = torch.sum(anchors * positives, dim=-1, keepdim=True) * self.scale
        neg_logits = torch.einsum('bd,bnd->bn', anchors, all_neg) * self.scale
        logits = torch.cat([pos_logit, neg_logits], dim=1)  # (B, 1+H+S)

        target = torch.zeros(B, dtype=torch.long, device=device)
        return self.cross_entropy_loss(logits, target)


# ════════════════════════════════════════════════════════════════════════
# 5. MarginMSE Loss — regression on pairwise similarity differences
# ════════════════════════════════════════════════════════════════════════

class MarginMSELoss(nn.Module):
    """
    MarginMSE: for every pair (d_i, d_j) in the batch, enforce that
        sim(q, d_i) - sim(q, d_j) ≈ score_i - score_j

    Uses both hard and soft negatives with their relevance scores.
    The positive has an implicit relevance of 1.0.

    Ref: Hofstätter et al. 2020 — "Improving Efficient Neural Ranking Models
         with Cross-Architecture Knowledge Distillation". Pure regression loss,
         no temperature/scale parameter in the original formulation.
    """

    def __init__(self, model, scale: float = 1.0):
        super().__init__()
        self.model = model
        self.scale = scale

    def forward(
        self,
        sentence_features: Iterable[Dict[str, torch.Tensor]],
        labels: torch.Tensor,
        hard_relevency: torch.Tensor,
        soft_relevency: torch.Tensor,
    ) -> torch.Tensor:
        embeddings = [self.model(f)["sentence_embedding"] for f in sentence_features]
        B = embeddings[0].size(0)
        device = embeddings[0].device

        anchors = F.normalize(embeddings[0], p=2, dim=-1)
        positives = F.normalize(embeddings[1], p=2, dim=-1)

        n_hard = hard_relevency.size(1)
        n_soft = soft_relevency.size(1)

        hard_neg = torch.stack(embeddings[2:2 + n_hard], dim=1)
        hard_neg = F.normalize(hard_neg, p=2, dim=-1)
        soft_neg = torch.stack(embeddings[2 + n_hard:2 + n_hard + n_soft], dim=1)
        soft_neg = F.normalize(soft_neg, p=2, dim=-1)

        # All documents: positive + hard + soft
        all_docs = torch.cat([positives.unsqueeze(1), hard_neg, soft_neg], dim=1)  # (B, 1+H+S, D)
        all_scores = torch.cat([
            torch.ones(B, 1, device=device),
            hard_relevency.float(),
            soft_relevency.float(),
        ], dim=1)  # (B, 1+H+S)

        # Similarities to all docs
        all_sims = torch.einsum('bd,bnd->bn', anchors, all_docs) * self.scale  # (B, 1+H+S)

        # Pairwise margin MSE: for all (i, j) pairs where score_i > score_j
        N = all_sims.size(1)
        sim_diff = all_sims.unsqueeze(2) - all_sims.unsqueeze(1)      # (B, N, N)
        score_diff = all_scores.unsqueeze(2) - all_scores.unsqueeze(1)  # (B, N, N)

        # Only consider pairs where score_i > score_j (upper triangle of score ordering)
        mask = (score_diff > 0).float()

        loss = mask * (sim_diff - score_diff) ** 2
        n_pairs = mask.sum().clamp(min=1.0)
        return loss.sum() / n_pairs


# ════════════════════════════════════════════════════════════════════════
# 6. RINCE — Ranking Info Noise Contrastive Estimation
#    (Hoffmann et al., AAAI 2022)
# ════════════════════════════════════════════════════════════════════════

class RINCELoss(nn.Module):
    """
    Ranking InfoNCE with rank-dependent temperatures.

    For each relevance boundary, items above the boundary are treated as
    positives and items below as negatives.  Higher-relevance boundaries
    use a lower temperature (tighter clustering), lower boundaries use a
    higher temperature (looser grouping).

    Ref: Hoffmann et al., AAAI 2022 — min_tau=0.1, max_tau=0.6 (CIFAR-100).
    """

    def __init__(
        self,
        model,
        min_tau: float = 0.1,
        max_tau: float = 0.6,
    ):
        super().__init__()
        self.model = model
        self.min_tau = min_tau
        self.max_tau = max_tau

    def _dynamic_tau(self, relevance: float) -> float:
        """Map relevance in [0,1] → temperature in [min_tau, max_tau].
        Higher relevance → lower temperature (tighter)."""
        return self.min_tau + (1.0 - relevance) * (self.max_tau - self.min_tau)

    def forward(
        self,
        sentence_features: Iterable[Dict[str, torch.Tensor]],
        labels: torch.Tensor,
        hard_relevency: torch.Tensor,
        soft_relevency: torch.Tensor,
    ) -> torch.Tensor:
        embeddings = [self.model(f)["sentence_embedding"] for f in sentence_features]
        B = embeddings[0].size(0)
        device = embeddings[0].device

        anchors = F.normalize(embeddings[0], p=2, dim=-1)
        positives = F.normalize(embeddings[1], p=2, dim=-1)

        n_hard = hard_relevency.size(1)
        n_soft = soft_relevency.size(1)
        hard_neg = F.normalize(torch.stack(embeddings[2:2 + n_hard], dim=1), p=2, dim=-1)
        soft_neg = F.normalize(torch.stack(embeddings[2 + n_hard:2 + n_hard + n_soft], dim=1), p=2, dim=-1)

        all_docs = torch.cat([positives.unsqueeze(1), soft_neg, hard_neg], dim=1)
        all_scores = torch.cat([
            torch.ones(B, 1, device=device),
            soft_relevency.float(),
            hard_relevency.float(),
        ], dim=1)

        all_sims = torch.einsum('bd,bnd->bn', anchors, all_docs)

        # Collect unique relevance thresholds across the batch (descending)
        unique_thresholds = all_scores.unique(sorted=True).flip(0)
        unique_thresholds = unique_thresholds[unique_thresholds > 0]

        total_loss = torch.zeros(1, device=device)
        n_terms = 0

        for threshold in unique_thresholds:
            pos_mask = all_scores >= threshold          # (B, N)
            neg_mask = all_scores < threshold           # (B, N)
            has_both = pos_mask.any(dim=1) & neg_mask.any(dim=1)

            if not has_both.any():
                continue

            tau = self._dynamic_tau(threshold.item())

            pos_sims = all_sims / tau
            neg_sims = all_sims / tau

            # Mask out: set non-pos to -inf, non-neg to -inf
            pos_sims = pos_sims.masked_fill(~pos_mask, float('-inf'))
            neg_sims = neg_sims.masked_fill(~neg_mask, float('-inf'))

            # InfoNCE per rank: -log( sum_exp_pos / (sum_exp_pos + sum_exp_neg) )
            log_sum_pos = torch.logsumexp(pos_sims, dim=1)         # (B,)
            log_sum_all = torch.logsumexp(
                torch.cat([pos_sims, neg_sims], dim=1), dim=1
            )                                                       # (B,)

            rank_loss = -(log_sum_pos - log_sum_all)
            rank_loss = rank_loss[has_both]

            total_loss = total_loss + rank_loss.mean()
            n_terms += 1

        if n_terms > 0:
            total_loss = total_loss / n_terms

        return total_loss


# ════════════════════════════════════════════════════════════════════════
# 7. CLOC — Contrastive Learning for Ordinal Classification
#    Multi-Margin N-pair Loss (Pitawela et al., CVPR 2025)
# ════════════════════════════════════════════════════════════════════════

class CLOCLoss(nn.Module):
    """
    Multi-Margin N-pair loss for ordinal / graded relevance.

    For every pair (d_i, d_j) where relevance_i > relevance_j, enforces:
        sim(q, d_i) > sim(q, d_j) + margin(ordinal_distance)

    Margins are mapped through learnable cumulative distances (softplus),
    so the margin between rank 0 and rank 3 is larger than between rank 0
    and rank 1.

    Ref: Pitawela et al., CVPR 2025 — learnable margins initialized
         randomly in [0.5, 1.0] with softplus parameterization.
    """

    def __init__(
        self,
        model,
        n_margin_bins: int = 10,
        learnable_margins: bool = True,
        init_margin: float = 0.5,
    ):
        super().__init__()
        self.model = model
        self.n_bins = n_margin_bins

        raw = self._inv_softplus(
            torch.rand(n_margin_bins) * 0.5 + init_margin
        )
        if learnable_margins:
            self.raw_margins = nn.Parameter(raw)
        else:
            self.register_buffer("raw_margins", raw)

    @staticmethod
    def _inv_softplus(t: torch.Tensor) -> torch.Tensor:
        return torch.where(t > 20, t, torch.log(torch.exp(t) - 1.0))

    @property
    def margins(self) -> torch.Tensor:
        """Cumulative margins: margin for ordinal distance d = sum of bin margins 0..d-1."""
        per_bin = F.softplus(self.raw_margins)
        return torch.cumsum(per_bin, dim=0)

    def _get_margin(self, score_diff: torch.Tensor) -> torch.Tensor:
        """Interpolate cumulative margin from continuous relevance difference in [0,1]."""
        margins = self.margins
        idx = score_diff * (self.n_bins - 1)
        idx_lo = idx.long().clamp(0, self.n_bins - 1)
        idx_hi = (idx_lo + 1).clamp(0, self.n_bins - 1)
        frac = idx - idx_lo.float()
        return margins[idx_lo] * (1.0 - frac) + margins[idx_hi] * frac

    def forward(
        self,
        sentence_features: Iterable[Dict[str, torch.Tensor]],
        labels: torch.Tensor,
        hard_relevency: torch.Tensor,
        soft_relevency: torch.Tensor,
    ) -> torch.Tensor:
        embeddings = [self.model(f)["sentence_embedding"] for f in sentence_features]
        B = embeddings[0].size(0)
        device = embeddings[0].device

        anchors = F.normalize(embeddings[0], p=2, dim=-1)
        positives = F.normalize(embeddings[1], p=2, dim=-1)

        n_hard = hard_relevency.size(1)
        n_soft = soft_relevency.size(1)
        hard_neg = F.normalize(torch.stack(embeddings[2:2 + n_hard], dim=1), p=2, dim=-1)
        soft_neg = F.normalize(torch.stack(embeddings[2 + n_hard:2 + n_hard + n_soft], dim=1), p=2, dim=-1)

        all_docs = torch.cat([positives.unsqueeze(1), soft_neg, hard_neg], dim=1)
        all_scores = torch.cat([
            torch.ones(B, 1, device=device),
            soft_relevency.float(),
            hard_relevency.float(),
        ], dim=1)

        all_sims = torch.einsum('bd,bnd->bn', anchors, all_docs)  # (B, N)
        N = all_sims.size(1)

        # Pairwise differences: (B, N, N)
        sim_i = all_sims.unsqueeze(2).expand(-1, -1, N)
        sim_j = all_sims.unsqueeze(1).expand(-1, N, -1)
        score_i = all_scores.unsqueeze(2).expand(-1, -1, N)
        score_j = all_scores.unsqueeze(1).expand(-1, N, -1)

        # Only pairs where score_i > score_j
        valid = score_i > score_j
        score_diff = (score_i - score_j).clamp(min=0.0)

        # Ordinal margins that scale with relevance distance
        pair_margins = self._get_margin(score_diff)

        # N-pair hinge: relu( -sim_i + sim_j + margin )
        hinge = F.relu(-sim_i + sim_j + pair_margins)
        hinge = hinge * valid.float()

        n_pairs = valid.float().sum().clamp(min=1.0)
        return hinge.sum() / n_pairs
