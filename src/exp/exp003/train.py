import functools
import pathlib
from typing import Any

import numpy as np
import polars as pl
import pydantic
import torch
import torch.nn as nn
import transformers
from datasets import Dataset
from sklearn import model_selection
from transformers import AutoTokenizer, DataCollatorWithPadding, Trainer, TrainingArguments, logging

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
+ feature engineering
    + stats of positive feedback count by clothing id
    + stats of age by clothing id
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

    max_length: int = pydantic.Field(default=256)

    lr: float = pydantic.Field(default=2e-4)
    num_train_epochs: int = pydantic.Field(default=5)
    per_device_train_batch_size: int = pydantic.Field(default=4)
    per_device_valid_batch_size: int = pydantic.Field(default=4)
    gradient_accumulation_steps: int = pydantic.Field(default=8)

    steps: int = pydantic.Field(default=25)


def preprocessing(df: pl.DataFrame) -> pl.DataFrame:
    df = df.with_columns(
        pl.col("Title").fill_null("").alias("Title"),
        pl.col("Review Text").fill_null("").alias("Review Text"),
    ).with_columns(
        ("Titlte: " + pl.col("Title") + " [SEP] " + "Review Text: " + pl.col("Review Text")).alias("prompt"),
    )
    df = df.join(
        df.group_by("Clothing ID").agg(
            mean_positive_feedback_count=pl.mean("Positive Feedback Count").fill_null(-1),
            std_positive_feedback_count=pl.std("Positive Feedback Count").fill_null(-1),
            min_positvie_feedback_count=pl.min("Positive Feedback Count").fill_null(-1),
            max_positive_feedback_count=pl.max("Positive Feedback Count").fill_null(-1),
        ),
        on="Clothing ID",
    ).join(
        df.group_by("Clothing ID").agg(
            mean_age=pl.mean("Age").fill_null(0),
            std_age=pl.std("Age").fill_null(0),
            min_age=pl.min("Age").fill_null(0),
            max_age=pl.max("Age").fill_null(0),
        ),
        on="Clothing ID",
    )
    df = df.with_columns(
        # --- feats for Positive Feedback Count
        diff_mean_positive_feedback_count=pl.col("Positive Feedback Count") - pl.col("mean_positive_feedback_count"),
        diff_std_positive_feedback_count=pl.col("Positive Feedback Count") - pl.col("std_positive_feedback_count"),
        diff_min_positvie_feedback_count=pl.col("Positive Feedback Count") - pl.col("min_positvie_feedback_count"),
        diff_max_positive_feedback_count=pl.col("Positive Feedback Count") - pl.col("max_positive_feedback_count"),
        # --- feats for Age
        diff_mean_age=pl.col("Age") - pl.col("mean_age"),
        diff_std_age=pl.col("Age") - pl.col("std_age"),
        diff_min_age=pl.col("Age") - pl.col("min_age"),
        diff_max_age=pl.col("Age") - pl.col("max_age"),
    )
    return df


USE_FEATURES = [
    # --- feats for Positive Feedback Count
    "mean_positive_feedback_count",
    "std_positive_feedback_count",
    "min_positvie_feedback_count",
    "max_positive_feedback_count",
    "diff_mean_positive_feedback_count",
    "diff_std_positive_feedback_count",
    "diff_min_positvie_feedback_count",
    "diff_max_positive_feedback_count",
    # --- feats for Age
    "mean_age",
    "std_age",
    "min_age",
    "max_age",
    "diff_mean_age",
    "diff_std_age",
    "diff_min_age",
    "diff_max_age",
]


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
    assert preds.shape[1] == 2
    return {"auc": metrics.score(y_true=labels, y_pred=preds[:, 1])}


class LoggingCallback(transformers.TrainerCallback):
    def on_evaluate(self, args: Any, state: Any, contro: Any, **kwargs: Any) -> None:
        logger.info(f"Eval on Trainer: {state.log_history[-1]}")


class CustomTrainer(Trainer):
    """
    References:
    1. https://dev.classmethod.jp/articles/huggingface-usage-custom-loss-func/
    """

    def compute_loss(
        self, model: Atma17CustomModel, inputs: dict, return_outputs: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        aux_labels = inputs.pop("aux_labels")

        more_features = []
        for col in USE_FEATURES:
            feat = inputs.pop(col)
            more_features.append(feat.reshape(-1, 1))
        more_features = torch.cat(more_features, dim=1)
        inputs["more_features"] = more_features
        outputs = model(**inputs)

        loss = nn.SmoothL1Loss()(outputs.logits, inputs["labels"])

        aux_loss = nn.functional.cross_entropy(input=outputs.aux_logits, target=aux_labels)
        outputs["aux_loss"] = aux_loss

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
            Dataset.from_pandas(train_df.to_pandas(use_pyarrow_extension_array=True).iloc[train_idx])
            .map(
                functools.partial(
                    tokenize, tokenizer=tokenizer, max_length=cfg.max_length, padding="max_length", truncation=True
                )
            )
            .map(to_onehot)
            .remove_columns(["prompt", "__index_level_0__"])
        )
        ds_valid = (
            Dataset.from_pandas(train_df.to_pandas(use_pyarrow_extension_array=True).iloc[valid_idx])
            .map(
                functools.partial(
                    tokenize, tokenizer=tokenizer, max_length=cfg.max_length, padding="max_length", truncation=True
                )
            )
            .map(to_onehot)
            .remove_columns(["prompt", "__index_level_0__"])
        )
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
            label_names=["labels", "aux_labels"] + USE_FEATURES,
        )
        model = Atma17CustomModel(cfg.model_path)
        # dummy forward for lazy initialization
        _ = model(
            input_ids=torch.randint(0, 1000, (8, cfg.max_length)),
            labels=torch.randint(0, 2, (8,)),
            attention_mask=torch.randint(0, 1, (8, cfg.max_length)),
            more_features=torch.randn(8, 16),
        )

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
