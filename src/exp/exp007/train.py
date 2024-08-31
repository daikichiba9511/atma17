import functools
import pathlib
from typing import Any

import numpy as np
import polars as pl
import pydantic
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from datasets import Dataset
from sklearn import model_selection
from transformers import AutoTokenizer, DataCollatorWithPadding, Trainer, TrainingArguments

from src import constants, log, metrics, utils

# 実験で閉じてるのだけ相対import
from .models import Atma17CustomModel

logger = log.get_root_logger()

EXP_NO = __file__.split("/")[-2]
DESCRIPTION = """
simple baseline
+ promptの作り方を変える Title: <title> [SEP] Review_Text: <review> text>
+ fill_null("none")で埋める
+ aux task
"""


class Config(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True, frozen=True, protected_namespaces=())
    name: str = pydantic.Field(default=EXP_NO)
    description: str = pydantic.Field(default=DESCRIPTION)

    # -- General
    is_debug: bool = pydantic.Field(default=False)
    root_dir: pathlib.Path = pydantic.Field(default=constants.ROOT)
    """Root directory. alias to constants.ROOT"""
    input_dir: pathlib.Path = pydantic.Field(default=constants.INPUT_DIR)
    """input directory. alias to constants.INPUT_DIR"""
    output_dir: pathlib.Path = pydantic.Field(default=constants.OUTPUT_DIR / EXP_NO)
    """output directory. constants.OUTPUT_DIR/EXP_NO"""
    data_dir: pathlib.Path = pydantic.Field(default=constants.DATA_DIR)
    """data directory. alias to constants.DATA_DIR"""
    seed: int = pydantic.Field(default=42)

    train_csv_fp: pathlib.Path = pydantic.Field(default=constants.DATA_DIR / "train.csv")
    train_meta_csv_fp: pathlib.Path = pydantic.Field(default=constants.DATA_DIR / "clothing_master.csv")
    test_csv_fp: pathlib.Path = pydantic.Field(default=constants.DATA_DIR / "test.csv")

    target_cols: list[str] = pydantic.Field(default_factory=lambda: ["Recommended IND"])
    model_path: str = pydantic.Field(default="microsoft/deberta-v3-large")
    # model_path: str = pydantic.Field(default="microsoft/deberta-v3-base")

    max_length: int = pydantic.Field(default=256 * 2)

    lr: float = pydantic.Field(default=2e-5)
    num_train_epochs: int = pydantic.Field(default=5)
    per_device_train_batch_size: int = pydantic.Field(default=8 // 1)
    per_device_valid_batch_size: int = pydantic.Field(default=8 // 1)
    gradient_accumulation_steps: int = pydantic.Field(default=4 * 1)

    steps: int = pydantic.Field(default=25)


def preprocessing(df: pl.DataFrame) -> pl.DataFrame:
    df = df.with_columns(
        pl.col("Title").fill_null("").alias("Title"),
        pl.col("Review Text").fill_null("").alias("Review Text"),
    ).with_columns(
        ("Titlte: " + pl.col("Title") + " [SEP] " + "Review Text: " + pl.col("Review Text")).alias("prompt"),
    )
    return df


def tokenize(
    data: dict,
    tokenizer: transformers.PreTrainedTokenizer | transformers.PreTrainedTokenizerFast,
    max_length: int = 256,
    padding: str | bool = False,
    truncation: bool = True,
) -> transformers.BatchEncoding:
    return tokenizer(data["prompt"], truncation=truncation, max_length=max_length, padding=padding)


def to_onehot(data: dict) -> dict:
    data["aux_labels"] = nn.functional.one_hot(torch.tensor(data["Rating"]), 6).float()
    return data


def compute_metrics(eval_pred: transformers.EvalPrediction) -> dict:
    preds, all_labels = eval_pred
    if isinstance(all_labels, tuple):
        labels = all_labels[0]
    else:
        labels = all_labels
    if isinstance(preds, tuple):
        preds = preds[0]

    preds = torch.softmax(torch.from_numpy(preds), dim=1).numpy()
    return {"auc": metrics.score(y_true=labels, y_pred=preds[:, 1])}


class LoggingCallback(transformers.TrainerCallback):
    def on_evaluate(self, args: Any, state: Any, contro: Any, **kwargs: Any) -> None:
        logger.info(f"Eval on Trainer: {state.log_history[-1]}")


class FocalLoss(nn.Module):
    def __init__(self, reduction: str = "none", alpha: int | float = 1, gamma: int | float = 2) -> None:
        super().__init__()
        self.reduction = reduction
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        pt = torch.exp(-bce_loss)
        loss = self.alpha * (1.0 - pt) ** self.gamma * bce_loss
        if self.reduction == "none":
            loss = loss
        elif self.reduction == "sum":
            loss = loss.sum()
        elif self.reduction == "mean":
            loss = loss.mean()
        return loss


class SmoothFocalLoss(nn.Module):
    def __init__(
        self, reduction: str = "none", alpha: int | float = 1, gamma: int | float = 2, smoothing: float = 0.0
    ) -> None:
        super().__init__()
        self.reduction = reduction
        self.focal_loss = FocalLoss(reduction="none", alpha=alpha, gamma=gamma)
        self.smoothing = smoothing

    @staticmethod
    def _smooth(targets: torch.Tensor, smoothing: float = 0.0) -> torch.Tensor:
        assert 0 <= smoothing < 1
        with torch.no_grad():
            targets = targets * (1.0 - smoothing) + 0.5 * smoothing
        return targets

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = self._smooth(targets, self.smoothing)
        loss = self.focal_loss(inputs, targets)
        if self.reduction == "none":
            loss = loss
        elif self.reduction == "sum":
            loss = loss.sum()
        elif self.reduction == "mean":
            loss = loss.mean()
        return loss


class CustomTrainer(Trainer):
    """
    References:
    1. https://dev.classmethod.jp/articles/huggingface-usage-custom-loss-func/
    """

    def compute_loss(
        self, model: Atma17CustomModel, inputs: dict, return_outputs: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        aux_labels = inputs.pop("aux_labels")
        outputs = model(**inputs)

        aux_loss = F.cross_entropy(input=outputs.aux_logits, target=aux_labels, label_smoothing=0.0)
        outputs["aux_loss"] = aux_loss

        loss = F.cross_entropy(
            input=outputs.logits,
            target=nn.functional.one_hot(inputs["labels"], 2).float(),
            weight=outputs.attentions,
            # label_smoothing=0.1,
        )
        weight_aux_loss = 0.5
        loss = loss + weight_aux_loss * aux_loss
        return (loss, outputs) if return_outputs else loss


def _test_dataset() -> None:
    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-large")
    train_df = pl.read_csv(constants.DATA_DIR / "train.csv")
    train_df = preprocessing(train_df)
    train_df = train_df.with_columns(pl.col("Recommended IND").cast(pl.Int32).alias("labels"))
    ds_train = (
        Dataset.from_pandas(
            train_df.to_pandas(use_pyarrow_extension_array=True).iloc[:100].loc[:, ["prompt", "labels", "Rating"]]
        )
        .map(functools.partial(tokenize, tokenizer=tokenizer, max_length=256, padding=True))
        .map(to_onehot)
        # .remove_columns(["prompt", "__index_level_0__"])
    )
    ds_train.set_format(type="torch", columns=["input_ids", "attention_mask", "labels", "aux_labels"])
    batch = next(iter(ds_train))
    print(batch)
    print(batch.keys())  # type: ignore


def main() -> None:
    cfg = Config(is_debug=False)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    log.attach_file_handler(logger, str(cfg.output_dir / "train.log"))
    utils.pinfo(cfg.model_dump())

    utils.seed_everything(cfg.seed)

    train_df = pl.read_csv(cfg.train_csv_fp)
    meta_df = pl.read_csv(cfg.train_meta_csv_fp)
    test_df = pl.read_csv(cfg.test_csv_fp)

    if cfg.is_debug:
        train_df = train_df.head(100)

    train_df = preprocessing(train_df)
    test_df = preprocessing(test_df)

    train_df = train_df.with_columns(pl.col(cfg.target_cols[0]).cast(pl.Int32).alias("labels"))
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_path)

    preds = np.zeros((len(train_df), 2))
    folds = np.zeros(len(train_df))
    kfold = model_selection.StratifiedKFold(n_splits=5, shuffle=True, random_state=cfg.seed)
    for fold, (train_idx, valid_idx) in enumerate(
        kfold.split(train_df.to_pandas(use_pyarrow_extension_array=True), train_df[cfg.target_cols[0]])
    ):
        utils.seed_everything(cfg.seed)
        logger.info(f"Start Training Fold: {fold}")
        ds_train = (
            Dataset.from_pandas(
                train_df.to_pandas(use_pyarrow_extension_array=True)
                .iloc[train_idx]
                .loc[:, ["prompt", "labels", "Rating"]],
            )
            .map(functools.partial(tokenize, tokenizer=tokenizer, max_length=cfg.max_length, padding="max_length"))
            .map(to_onehot)
            .remove_columns(["prompt", "__index_level_0__", "Rating"])
        )
        # ds_train.set_format(type="torch", columns=["input_ids", "attention_mask", "labels", "aux_labels"])
        ds_valid = (
            Dataset.from_pandas(
                train_df.to_pandas(use_pyarrow_extension_array=True)
                .iloc[valid_idx]
                .loc[:, ["prompt", "labels", "Rating"]]
            )
            .map(functools.partial(tokenize, tokenizer=tokenizer, max_length=cfg.max_length, padding="max_length"))
            .map(to_onehot)
            .remove_columns(["prompt", "__index_level_0__", "Rating"])
        )
        # ds_valid.set_format(type="torch", columns=["input_ids", "attention_mask", "labels", "aux_labels"])
        train_args = TrainingArguments(
            output_dir=str(cfg.output_dir / f"fold{fold}"),
            fp16=True,
            learning_rate=cfg.lr,
            per_device_train_batch_size=cfg.per_device_train_batch_size,
            per_device_eval_batch_size=cfg.per_device_valid_batch_size,
            gradient_accumulation_steps=cfg.gradient_accumulation_steps,
            report_to="none",
            eval_strategy="steps",
            do_eval=True,
            eval_steps=cfg.steps,
            save_total_limit=3,
            save_strategy="steps",
            save_steps=cfg.steps,
            logging_steps=cfg.steps,
            lr_scheduler_type="cosine",
            metric_for_best_model="auc",  # AUCを評価に使用する
            greater_is_better=True,
            warmup_ratio=0.1,
            weight_decay=0.01,
            save_safetensors=True,
            seed=cfg.seed,
            data_seed=cfg.seed,
            optim="adamw_torch",
            load_best_model_at_end=True,
            label_names=["labels", "aux_labels"],
        )
        model = Atma17CustomModel(model_path=cfg.model_path)
        # dummy forward for lazy initialization
        i = torch.randint(0, 1000, (8, 256))
        _ = model(i, torch.randint(0, 2, (8,)))

        trainer = CustomTrainer(
            model=model,
            args=train_args,
            train_dataset=ds_train,
            eval_dataset=ds_valid,
            data_collator=DataCollatorWithPadding(tokenizer),
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
            callbacks=[
                LoggingCallback(),
            ],
        )
        trainer.train()

        trainer.save_model(str(cfg.output_dir / f"deverta-large-seed{cfg.seed}-fold{fold}"))
        tokenizer.save_pretrained(str(cfg.output_dir / f"deverta-large-seed{cfg.seed}-fold{fold}"))
        torch.save(
            trainer.model.state_dict(),
            str(cfg.output_dir / f"deverta-large-seed{cfg.seed}-fold{fold}" / f"best_model_fold{fold}.pth"),
        )
        preds_ = trainer.predict(ds_valid).predictions[0]  # type: ignore
        preds_ = torch.tensor(preds_).softmax(dim=1).numpy()  # type: ignore
        preds[valid_idx] = preds_
        folds[valid_idx] = fold

    train_df = train_df.with_columns(pl.Series("preds", preds[:, 1]), pl.Series("folds", folds))
    train_df.write_parquet(str(cfg.output_dir / "oof.parquet"))
    auc_score = metrics.score(y_pred=train_df["preds"].to_numpy(), y_true=train_df["labels"].to_numpy())
    logger.info(f"OOF AUC: {auc_score}")
    logger.info("End Training")


if __name__ == "__main__":
    # _test_dataset()
    main()
