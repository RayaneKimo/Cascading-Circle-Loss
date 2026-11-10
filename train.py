#!/usr/bin/env python3
"""
CasCL training entry point.

Loads the data described in the README, builds the selected loss, and trains
with a SentenceTransformerTrainer, evaluating on the test set every eval_steps.

To re-run the experiments, prefer scripts/run_experiments.sh, which wraps this
script (training + evaluation of every checkpoint) with named presets.

Examples:
    # One run of the main CasCL loss
    python train.py --config configs/qwen3_0.6b_cascl.yml --loss cascl

    # Two-regime variant, relevance gain off
    python train.py --loss cascl_two_regime --margin_safe 0.05 --use_relevance_gain off

    # Any config value can be overridden from the command line
    python train.py --loss cascl --data_dir data/example \
        --set train_config.per_device_train_batch_size=8 model_config.attn_implementation=sdpa
"""
import argparse
import logging
import os
from datetime import datetime
from typing import Dict

import torch
from omegaconf import DictConfig, OmegaConf
from sentence_transformers import SentenceTransformer
from sentence_transformers import SentenceTransformerTrainingArguments
from transformers import set_seed

from cascl.io_utils import load_file, save_file
from cascl.data_utils import RetrieverDataset
from cascl.data_collator import SentenceTransformerDataCollator
from cascl.evaluation import get_evaluator
from cascl.trainer import ExtendedSentenceTransformerTrainer
from cascl.losses import (
    CascadingCircleLoss,
    CascadingCircleLossSupCon,
    CascadingCircleLossRectified,
    CascadingCircleLossTwoRegime,
    CascadingCircleLossNegRelative,
    InfoNCELoss,
    CircleLoss,
    TripletLoss,
    MultipleNegativesRankingLoss as MNRLoss,
    MarginMSELoss,
    RINCELoss,
    CLOCLoss,
)

from transformers import modeling_utils
if not hasattr(modeling_utils, "ALL_PARALLEL_STYLES") or modeling_utils.ALL_PARALLEL_STYLES is None:
    modeling_utils.ALL_PARALLEL_STYLES = ["tp", "none", "colwise", "rowwise"]

logger = logging.getLogger(__name__)


LOSS_REGISTRY = {
    "cascl": "CascadingCircleLoss (full)",
    "cascl_supcon": "CascadingCircleLoss with SupCon (extra positives pulled together)",
    "cascl_rectified": "CascadingCircleLoss rectified (alpha_min floor on self-paced weights)",
    "cascl_two_regime": "CascadingCircleLoss two-regime self-pacing (emergency + refinement)",
    "cascl_neg_relative": "CascadingCircleLoss neg-relative self-pacing (target = r_j + s_neg)",
    "infonce": "InfoNCELoss",
    "circle": "CircleLoss",
    "triplet": "TripletLoss",
    "mnrl": "MultipleNegativesRankingLoss",
    "margin_mse": "MarginMSELoss",
    "rince": "RINCELoss (rank-dependent temperature InfoNCE)",
    "cloc": "CLOCLoss (multi-margin N-pair, learnable ordinal margins)",
    # Ablation variants
    "cascl_no_infonce": "CasCL without InfoNCE term (cascade only)",
    "cascl_no_cascade": "CasCL without cascade term (InfoNCE only on hard negs)",
    "cascl_no_selfpaced": "CasCL without self-paced adaptive weights",
    "cascl_no_margins": "CasCL without proportional margins (delta=0)",
    "cascl_no_grading": "CasCL without graded ordering (random chain)",
    "cascl_no_relevance_weight": "CasCL without relevance gain weighting (uniform w)",
}

def build_loss(loss_name: str, model, strategy, params: Dict = None):
    """Instantiate the selected loss function using config values.
    params can override config values (e.g. alpha_min per run)."""
    params = params or {}
    tc = strategy.train_config
    scale = getattr(tc, "scale", 15)
    gamma = getattr(tc, "gamma", 64)
    delta_0 = getattr(tc, "delta_0", 0.3)
    delta_warmup = getattr(tc, "delta_warmup", False)
    delta_0_start = getattr(tc, "delta_0_start", 0.1)
    delta_0_end = getattr(tc, "delta_0_end", 0.3)
    delta_warmup_steps = getattr(tc, "delta_warmup_steps", 500)
    grad_accum = getattr(tc, "gradient_accumulation_steps", 1)
    lambda1 = getattr(tc, "lambda1", 1.0)
    lambda2 = getattr(tc, "lambda2", 0.5)
    use_relevance_gain = params.get("use_relevance_gain", getattr(tc, "use_relevance_gain", True))

    cascl_kwargs = dict(
        scale=scale, gamma=gamma, delta_0=delta_0,
        delta_warmup=delta_warmup, delta_0_start=delta_0_start,
        delta_0_end=delta_0_end, delta_warmup_steps=delta_warmup_steps,
        gradient_accumulation_steps=grad_accum,
        lambda1=lambda1, lambda2=lambda2,
        use_relevance_gain=use_relevance_gain,
    )

    if loss_name == "cascl":
        return CascadingCircleLoss(model, **cascl_kwargs)
    elif loss_name == "cascl_supcon":
        return CascadingCircleLossSupCon(model, **cascl_kwargs)
    elif loss_name == "cascl_rectified":
        alpha_min = params.get("alpha_min", getattr(tc, "alpha_min", 0.1))
        return CascadingCircleLossRectified(model, **cascl_kwargs, alpha_min=alpha_min)
    elif loss_name == "cascl_two_regime":
        margin_safe = params.get("margin_safe", getattr(tc, "margin_safe", 0.05))
        return CascadingCircleLossTwoRegime(model, **cascl_kwargs, margin_safe=margin_safe)
    elif loss_name == "cascl_neg_relative":
        alpha_mode = params.get("alpha_mode", getattr(tc, "alpha_mode", "relative"))
        return CascadingCircleLossNegRelative(model, **cascl_kwargs, alpha_mode=alpha_mode)
    elif loss_name == "infonce":
        # van den Oord et al. 2018 / sentence-transformers standard: scale=20 (τ=0.05)
        return InfoNCELoss(model, scale=20.0)
    elif loss_name == "circle":
        # Sun et al., CVPR 2020: γ=64, m=0.25
        return CircleLoss(model, scale=64.0, gamma=64.0, margin=0.25)
    elif loss_name == "triplet":
        # Schroff et al. (FaceNet), 2015: margin=0.2
        return TripletLoss(model, margin=0.2)
    elif loss_name == "mnrl":
        # sentence-transformers standard: scale=20 (τ=0.05)
        return MNRLoss(model, scale=20.0)
    elif loss_name == "margin_mse":
        # Hofstätter et al. 2020: pure regression, no scaling
        return MarginMSELoss(model, scale=1.0)
    elif loss_name == "rince":
        # Hoffmann et al., AAAI 2022: min_tau=0.1, max_tau=0.6
        return RINCELoss(model, min_tau=0.1, max_tau=0.6)
    elif loss_name == "cloc":
        # Pitawela et al., CVPR 2025: learnable margins init in [0.5, 1.0]
        return CLOCLoss(model, n_margin_bins=10, learnable_margins=True, init_margin=0.5)
    # ── Ablation variants ──
    elif loss_name == "cascl_no_infonce":
        return CascadingCircleLoss(model, **cascl_kwargs, disable_infonce=True)
    elif loss_name == "cascl_no_cascade":
        return CascadingCircleLoss(model, **cascl_kwargs, disable_cascade=True)
    elif loss_name == "cascl_no_selfpaced":
        return CascadingCircleLoss(model, **cascl_kwargs, disable_self_paced=True)
    elif loss_name == "cascl_no_margins":
        return CascadingCircleLoss(model, **cascl_kwargs, disable_margins=True)
    elif loss_name == "cascl_no_grading":
        return CascadingCircleLoss(model, **cascl_kwargs, disable_graded_order=True)
    elif loss_name == "cascl_no_relevance_weight":
        return CascadingCircleLoss(model, **{**cascl_kwargs, "use_relevance_gain": False})
    else:
        raise ValueError(
            f"Unknown loss '{loss_name}'. "
            f"Available: {list(LOSS_REGISTRY.keys())}"
        )


def format_run_name(strategy, params: Dict, run_prefix: str) -> str:
    """Run directory name, identical to the naming used in the experiments."""
    tc = strategy.train_config
    loss_name = params.get("loss", "cascl")
    bs = tc.per_device_train_batch_size
    ep = tc.num_train_epochs
    lr = tc.learning_rate
    ga = tc.gradient_accumulation_steps
    date = datetime.now().strftime("%d%m%Y")
    base = f"{run_prefix}-bs{bs}-ga{ga}-lr{lr}-ep-{ep}-{date}"
    if loss_name in ("cascl", "cascl_supcon", "cascl_rectified", "cascl_two_regime", "cascl_neg_relative"):
        prefix = ""
    elif loss_name.startswith("cascl_no_"):
        prefix = "ablation-"
    else:
        prefix = "baseline-"

    if loss_name.startswith("cascl"):
        dw = getattr(tc, "delta_warmup", False)
        if dw:
            delta_str = (
                f"-delta_warmup_{getattr(tc, 'delta_0_start', 0.1)}"
                f"_to_{getattr(tc, 'delta_0_end', 0.3)}"
                f"_in_{getattr(tc, 'delta_warmup_steps', 500)}steps"
            )
        else:
            delta_str = f"-{getattr(tc, 'delta_0', 0.3)}delta0"
        name = (
            f"{prefix}{base}"
            f"-{getattr(tc, 'lambda1', 1.0)}lambda_1"
            f"-{getattr(tc, 'lambda2', 0.5)}lambda_2"
            f"-scale_{getattr(tc, 'scale', 15)}"
            f"{delta_str}"
            f"-{getattr(tc, 'gamma', 64)}gamma"
            f"-{loss_name}"
        )
        if loss_name == "cascl_rectified":
            am = params.get("alpha_min", getattr(tc, "alpha_min", 0.1))
            name += f"-alpha_min_{am}"
        elif loss_name == "cascl_two_regime":
            ms = params.get("margin_safe", getattr(tc, "margin_safe", 0.05))
            name += f"-margin_safe_{ms}"
        elif loss_name == "cascl_neg_relative":
            am = params.get("alpha_mode", getattr(tc, "alpha_mode", "relative"))
            name += f"-alpha_{am}"
        if "use_relevance_gain" in params:
            urg = params["use_relevance_gain"]
            name += f"-rel_gain_{'on' if urg else 'off'}"
        return name

    baseline_suffix = {
        "infonce": "loss_infonce-scale_20",
        "circle": "loss_circle-gamma_64-m_0.25",
        "triplet": "loss_triplet-margin_0.2",
        "mnrl": "loss_mnrl-scale_20",
        "margin_mse": "loss_margin_mse",
        "rince": "loss_rince-tau_0.1_0.6",
        "cloc": "loss_cloc-bins_10-init_0.5-learnable",
    }
    return f"{prefix}{base}-{baseline_suffix.get(loss_name, f'loss_{loss_name}')}"


def train(strategy: DictConfig, run_name: str, params: Dict, output_dir: str):
    print(strategy)

    # load dataset
    dataset = RetrieverDataset(config=strategy.dataset_config)

    # set seeds
    set_seed(strategy.train_config.seed)

    model = SentenceTransformer(
        strategy.model_config.base_model_reference,
        trust_remote_code=True,
        tokenizer_kwargs={"model_max_length": strategy.model_config.model_max_length, "padding_side": "left"},
        model_kwargs={
            "attn_implementation": getattr(strategy.model_config, "attn_implementation", "flash_attention_2"),
            "device_map": "auto",
            "torch_dtype": torch.bfloat16,
        },
    )
    run_directory = os.path.join(output_dir, run_name)
    os.makedirs(run_directory, exist_ok=True)
    save_file(
        content={**OmegaConf.to_container(strategy, resolve=True), "run_params": params},
        file_path=os.path.join(run_directory, "config.json"),
        indent=2,
    )

    # define loss function
    loss_name = params.get("loss", "cascl")
    loss_fct = build_loss(loss_name, model, strategy, params)

    tc = strategy.train_config
    training_arguments = SentenceTransformerTrainingArguments(
        output_dir=run_directory,
        num_train_epochs=tc.num_train_epochs,
        per_device_train_batch_size=tc.per_device_train_batch_size,
        per_device_eval_batch_size=tc.per_device_eval_batch_size,
        gradient_accumulation_steps=tc.gradient_accumulation_steps,
        learning_rate=tc.learning_rate,
        warmup_ratio=tc.warmup_ratio,
        fp16=tc.fp16,
        bf16=tc.bf16,
        weight_decay=tc.weight_decay,
        gradient_checkpointing=True,
        eval_strategy=tc.eval_strategy,
        eval_steps=tc.eval_steps,
        save_strategy=tc.save_strategy,
        save_steps=tc.save_steps,
        save_total_limit=tc.save_total_limit,
        metric_for_best_model=tc.metric_for_best_model,
        logging_steps=tc.logging_steps,
        run_name=run_name,
        report_to="tensorboard",
    )

    # document-level IR evaluator + graded NDCG, run every eval_steps
    evaluator = get_evaluator(dataset=dataset)

    training_dataset = dataset.training_dataset.training_dataset
    trainer = ExtendedSentenceTransformerTrainer(
        model=model,
        args=training_arguments,
        train_dataset=training_dataset,
        loss=loss_fct,
        evaluator=evaluator,
        data_collator=SentenceTransformerDataCollator(tokenize_fn=model.tokenize),
    )
    trainer.train()


def parse_args():
    p = argparse.ArgumentParser(description="Train an embedding model with CasCL (or a baseline loss)")
    p.add_argument("--config", default="configs/qwen3_0.6b_cascl.yml", help="YAML strategy config")
    p.add_argument("--output_dir", default="models", help="Runs are saved in <output_dir>/<run_name>/")
    p.add_argument("--run_prefix", default="qwen06B-neg-10", help="Prefix of the run directory name")

    p.add_argument("--loss", choices=list(LOSS_REGISTRY), default="cascl", help="Loss to train with")
    p.add_argument("--data_dir", default=None,
                   help="Directory with train.json / test.json / documents.json (overrides the config paths)")
    p.add_argument("--set", nargs="+", default=[], metavar="KEY=VALUE",
                   help="Override config values, e.g. train_config.num_train_epochs=1")
    p.add_argument("--run_dir_file", default=None,
                   help="Append the run directory to this file (used by scripts/run_experiments.sh)")
    p.add_argument("--print_run_name", action="store_true",
                   help="Print the run directory name and exit (no training)")

    # per-run overrides (single-run mode only)
    p.add_argument("--alpha_min", type=float, default=None, help="cascl_rectified: floor on self-paced weights")
    p.add_argument("--margin_safe", type=float, default=None, help="cascl_two_regime: margin above hardest negative")
    p.add_argument("--alpha_mode", choices=["relative", "absolute"], default=None, help="cascl_neg_relative mode")
    p.add_argument("--use_relevance_gain", choices=["on", "off"], default=None,
                   help="Override train_config.use_relevance_gain for this run")
    return p.parse_args()


def main():
    logging.basicConfig(level=logging.INFO)
    args = parse_args()

    strategy = OmegaConf.create(load_file(file_path=args.config))
    if args.data_dir:
        strategy.dataset_config.train_dataset_path = os.path.join(args.data_dir, "train.json")
        strategy.dataset_config.test_dataset_path = os.path.join(args.data_dir, "test.json")
        strategy.dataset_config.document_dataset_path = os.path.join(args.data_dir, "documents.json")
    if args.set:
        strategy = OmegaConf.merge(strategy, OmegaConf.from_dotlist(args.set))

    params = {"loss": args.loss}
    if args.alpha_min is not None:
        params["alpha_min"] = args.alpha_min
    if args.margin_safe is not None:
        params["margin_safe"] = args.margin_safe
    if args.alpha_mode is not None:
        params["alpha_mode"] = args.alpha_mode
    if args.use_relevance_gain is not None:
        params["use_relevance_gain"] = args.use_relevance_gain == "on"

    run_name = format_run_name(strategy, params, args.run_prefix)
    if args.print_run_name:
        print(run_name)
        return
    logger.info(f"▶ Run: {run_name}  ({LOSS_REGISTRY[params['loss']]})")
    if args.run_dir_file:
        with open(args.run_dir_file, "a") as f:
            f.write(os.path.join(args.output_dir, run_name) + "\n")
    train(strategy=strategy, run_name=run_name, params=params, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
