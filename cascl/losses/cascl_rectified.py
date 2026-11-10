from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
from torch import Tensor, nn
from typing import Iterable, Dict
import torch.nn.functional as F

from sentence_transformers import util
from sentence_transformers.SentenceTransformer import SentenceTransformer


class CascadingCircleLoss(nn.Module):
    """
    Cascading Circle Loss (CasCL) for graded contrastive learning.

    Dataset layout:
        sentence_features[0]            = anchor (query)
        sentence_features[1]            = positive (ground-truth passage)
        sentence_features[2 : 2+H]      = hard negatives  (H passages)
        sentence_features[2+H : 2+H+S]  = soft negatives  (S graded passages)
        hard_relevency                  = (B, H) relevance scores for hard negatives
        soft_relevency                  = (B, S) relevance scores for soft negatives

    Hard and soft negatives are used directly — no masking required.

    Loss = λ₁·L_binary(positives; hard_negs)
         + λ₂·L_cascade(true_pos → soft_neg₁ → ... → hardest_hard_neg)

    Args:
        model:    SentenceTransformer model
        scale:    temperature scale for InfoNCE (= 1/τ)
        gamma:    cascade circle sharpness (default 64)
        delta_0:  base margin (used when delta_warmup=False, default 0.3)
        delta_warmup: enable linear warmup of delta_0 (default False)
        delta_0_start: initial margin at step 0 when warmup is on (default 0.1)
        delta_0_end:   final margin after warmup (default 0.3)
        delta_warmup_steps: optimizer steps to ramp delta_0 (default 500)
        gradient_accumulation_steps: forward passes per optimizer step (default 1)
        lambda1:  weight for InfoNCE term (default 1.0)
        lambda2:  weight for cascade term (default 0.5)
        use_relevance_gain: weight cascade links by rⱼ^p (default True)
        relevance_weight_power: exponent p for the gain function w=r^p.
            p<1 → concave (slow decay, default 0.5 = √r)
            p=1 → linear
            p>1 → convex (fast decay, similar to original 2^r−1)
        alpha_min: minimum floor for self-paced weights (default 0.1).
            0.0 = original self-pacing (shuts off → best nDCG, weaker MAP/MRR)
            1.0 = no self-pacing (α=1 always → best MAP/MRR, weaker nDCG)
            0.05–0.15 = sweet spot (self-pacing with residual push → best ALL metrics)
    """

    def __init__(
        self,
        model,
        scale: float = 20.0,
        gamma: float = 64.0,
        delta_0: float = 0.3,
        delta_warmup: bool = False,
        delta_0_start: float = 0.1,
        delta_0_end: float = 0.3,
        delta_warmup_steps: int = 500,
        gradient_accumulation_steps: int = 1,
        lambda1: float = 1.0,
        lambda2: float = 0.5,
        use_relevance_gain: bool = True,
        relevance_weight_power: float = 0.5,
        alpha_min: float = 0.1,
        # Ablation flags
        disable_infonce: bool = False,
        disable_cascade: bool = False,
        disable_self_paced: bool = False,
        disable_margins: bool = False,
        disable_graded_order: bool = False,
    ):
        super().__init__()
        self.model = model
        self.scale = scale
        self.gamma = gamma
        self._delta_0_fixed = delta_0
        self.delta_warmup = delta_warmup
        self.delta_0_start = delta_0_start
        self.delta_0_end = delta_0_end
        self._delta_warmup_forward_steps = delta_warmup_steps * gradient_accumulation_steps
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.use_relevance_gain = use_relevance_gain
        self.relevance_weight_power = relevance_weight_power
        self.alpha_min = alpha_min
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        if self.delta_warmup:
            self.register_buffer("_step", torch.tensor(0, dtype=torch.long))
        # Ablation controls
        self.disable_infonce = disable_infonce
        self.disable_cascade = disable_cascade
        self.disable_self_paced = disable_self_paced
        self.disable_margins = disable_margins
        self.disable_graded_order = disable_graded_order

    # ----- helpers --------------------------------------------------------
    @staticmethod
    def _normalize(x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x, p=2, dim=-1)

    @property
    def delta_0(self) -> float:
        if not self.delta_warmup:
            return self._delta_0_fixed
        t = min(self._step.item() / max(self._delta_warmup_forward_steps, 1), 1.0)
        return self.delta_0_start + t * (self.delta_0_end - self.delta_0_start)

    # ----- forward --------------------------------------------------------
    def forward(
        self,
        sentence_features: Iterable[Dict[str, torch.Tensor]],
        labels: torch.Tensor,
        hard_relevency: torch.Tensor,
        soft_relevency: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            sentence_features:
                [0]               = anchor (query)
                [1]               = positive (ground-truth passage)
                [2 : 2+H]        = hard negatives  (H passages)
                [2+H : 2+H+S]    = soft negatives  (S passages, graded)
            labels:          (B,) — not used, kept for interface compat
            hard_relevency:  (B, H) — relevance scores for hard negatives (typically 0)
            soft_relevency:  (B, S) — relevance scores for soft negatives (graded, in [0,1])
        """
        # ── Step counter for delta_0 warmup ──────────────────────────
        if self.delta_warmup and self.training:
            self._step += 1

        # ── Encode ───────────────────────────────────────────────────
        embeddings = [self.model(f)["sentence_embedding"] for f in sentence_features]
        B = embeddings[0].size(0)
        device = embeddings[0].device

        anchors   = self._normalize(embeddings[0])            # (B, D)
        positives = self._normalize(embeddings[1])            # (B, D)

        n_hard = hard_relevency.size(1)
        n_soft = soft_relevency.size(1)

        hard_negatives = torch.stack(embeddings[2:2 + n_hard], dim=1)            # (B, H, D)
        hard_negatives = self._normalize(hard_negatives)

        soft_negatives = torch.stack(embeddings[2 + n_hard:2 + n_hard + n_soft], dim=1)  # (B, S, D)
        soft_negatives = self._normalize(soft_negatives)

        # ── Similarities ─────────────────────────────────────────────
        pos_sim   = torch.sum(anchors * positives, dim=-1, keepdim=True)          # (B, 1)
        hard_sims = torch.einsum('bd,bnd->bn', anchors, hard_negatives)           # (B, H)
        soft_sims = torch.einsum('bd,bnd->bn', anchors, soft_negatives)           # (B, S)

        # ════════════════════════════════════════════════════════════
        # Term 1: InfoNCE — true positive vs hard negatives
        # ════════════════════════════════════════════════════════════
        if not self.disable_infonce:
            pos_logit      = pos_sim * self.scale                                     # (B, 1)
            hard_logits    = hard_sims * self.scale                                   # (B, H)
            logits         = torch.cat([pos_logit, hard_logits], dim=1)               # (B, 1+H)
            infonce_labels = torch.zeros(B, dtype=torch.long, device=device)
            loss_infonce   = self.cross_entropy_loss(logits, infonce_labels)
        else:
            loss_infonce = torch.zeros(1, device=device)

        # ════════════════════════════════════════════════════════════
        # Term 2: Cascade (order-preserving chain)
        #
        # Chain per sample:
        #   true_positive (r=1.0) → soft_neg₁ → ... → soft_negₛ → hardest_hard_neg (r=0)
        #
        # Each link enforces:  sim(higher) > sim(lower) + margin
        # ════════════════════════════════════════════════════════════
        loss_cascade = torch.zeros(1, device=device)
        n_cascade_samples = 0

        if not self.disable_cascade:
            for i in range(B):
                if n_soft == 0:
                    continue

                graded_sims   = soft_sims[i]                    # (S,)
                graded_scores = soft_relevency[i].float()       # (S,)

                # Ablation: disable graded ordering → random chain order
                if not self.disable_graded_order:
                    order = graded_scores.argsort(descending=True)
                    graded_sims   = graded_sims[order]
                    graded_scores = graded_scores[order]

                s_hard_neg_i = hard_sims[i].max()               # hardest hard negative

                # Chain: [true_pos, soft_1, ..., soft_S, hardest_hard_neg]
                chain_sims = torch.cat([pos_sim[i], graded_sims, s_hard_neg_i.unsqueeze(0)])
                chain_scores = torch.cat([
                    torch.ones(1, device=device),
                    (graded_scores - 0.05).clamp(min=0.0),
                    torch.zeros(1, device=device),
                ])

                # ── Self-paced weights with floor ─────────────────
                # α = max([target − actual]₊, α_min)
                # α_min > 0 keeps a residual push so items below negatives
                # still receive gradient (fixes MAP/MRR), while self-pacing
                # still concentrates on misplaced items (preserves nDCG).
                if not self.disable_self_paced:
                    with torch.no_grad():
                        alpha_upper = torch.clamp(
                            F.relu(chain_scores[:-1] - chain_sims[:-1].detach()),
                            min=self.alpha_min,
                        )
                        alpha_lower = torch.clamp(
                            F.relu(chain_sims[1:].detach() - chain_scores[1:]),
                            min=self.alpha_min,
                        )
                else:
                    alpha_upper = torch.ones(chain_sims.size(0) - 1, device=device)
                    alpha_lower = torch.ones(chain_sims.size(0) - 1, device=device)

                # ── Proportional margins ──────────────────────────
                if not self.disable_margins:
                    delta = self.delta_0 * (chain_scores[:-1] - chain_scores[1:])
                else:
                    delta = torch.zeros(chain_sims.size(0) - 1, device=device)

                # ── Link logits ───────────────────────────────────
                logit = self.gamma * (
                    alpha_lower * chain_sims[1:]
                    - alpha_upper * chain_sims[:-1]
                    + delta
                )

                # ── Relevance gain weights ────────────────────────
                if self.use_relevance_gain:
                    w = chain_scores[:-1].clamp(min=0.0) ** self.relevance_weight_power
                    w = w.clamp(min=1e-3)
                else:
                    w = torch.ones_like(logit)

                loss_cascade = loss_cascade + (w * F.softplus(logit)).sum()
                n_cascade_samples += 1

            if n_cascade_samples > 0:
                loss_cascade = loss_cascade / n_cascade_samples

        # ── Combined loss ────────────────────────────────────────
        loss = self.lambda1 * loss_infonce + self.lambda2 * loss_cascade

        return loss
