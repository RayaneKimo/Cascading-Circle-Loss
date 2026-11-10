from __future__ import annotations

from typing import Any, Callable, TYPE_CHECKING

import torch
from torch import nn

from sentence_transformers import SentenceTransformerTrainer
from sentence_transformers.evaluation import SentenceEvaluator
from sentence_transformers.training_args import SentenceTransformerTrainingArguments

if TYPE_CHECKING:
    from sentence_transformers.SentenceTransformer import SentenceTransformer

from transformers.data.data_collator import DataCollator
from transformers import EvalPrediction, PreTrainedTokenizerBase, TrainerCallback

from datasets import Dataset
from datasets import DatasetDict
from datasets import IterableDataset


class ExtendedSentenceTransformerTrainer(SentenceTransformerTrainer):
    """SentenceTransformerTrainer that forwards the hard/soft relevance tensors to the loss."""

    def __init__(
        self,
        model: SentenceTransformer | None = None,
        args: SentenceTransformerTrainingArguments = None,
        train_dataset: Dataset | DatasetDict | IterableDataset | dict[str, Dataset] | None = None,
        eval_dataset: Dataset | DatasetDict | IterableDataset | dict[str, Dataset] | None = None,
        loss: nn.Module
        | dict[str, nn.Module]
        | Callable[[SentenceTransformer], torch.nn.Module]
        | dict[str, Callable[[SentenceTransformer], torch.nn.Module]]
        | None = None,
        evaluator: SentenceEvaluator | list[SentenceEvaluator] | None = None,
        data_collator: DataCollator | None = None,
        tokenizer: PreTrainedTokenizerBase | Callable | None = None,
        model_init: Callable[[], SentenceTransformer] | None = None,
        compute_metrics: Callable[[EvalPrediction], dict] | None = None,
        callbacks: list[TrainerCallback] | None = None,
        optimizers: tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        preprocess_logits_for_metrics: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
    ) -> None:
        super().__init__(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            loss=loss,
            evaluator=evaluator,
            data_collator=data_collator,
            tokenizer=tokenizer,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )

    def compute_loss(
        self,
        model: SentenceTransformer,
        inputs: dict[str, torch.Tensor | Any],
        return_outputs: bool = False,
        num_items_in_batch=None,
        **kwargs
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, Any]]:
        """
        Computes the loss, passing ``hard_relevency`` / ``soft_relevency`` (B, N) tensors
        from the batch to the loss as extra arguments:
            loss_fn(features, labels, hard_relevency, soft_relevency)
        """
        dataset_name = inputs.pop("dataset_name", None)
        hard_relevency = inputs["hard_relevency"]
        soft_relevency = inputs["soft_relevency"]
        del inputs["hard_relevency"]
        del inputs["soft_relevency"]
        features, labels = self.collect_features(inputs)
        loss_fn = self.loss

        if isinstance(loss_fn, dict) and dataset_name:
            loss_fn = loss_fn[dataset_name]

        # Insert the wrapped (e.g. distributed or compiled) model into the loss function,
        # if the loss stores the model. Only called once per process
        if (
            model == self.model_wrapped
            and hasattr(loss_fn, "model")  # Only if the loss stores the model
            and loss_fn.model != model  # Only if the wrapped model is not already stored
        ):
            loss_fn = self.override_model_in_loss(loss_fn, model)
        loss = loss_fn(features, labels, hard_relevency, soft_relevency)
        if return_outputs:
            # Sentence Transformer losses do not return outputs; prediction_loss_only=True
            # in SentenceTransformerTrainingArguments means the output is not used.
            return loss, {}
        return loss
