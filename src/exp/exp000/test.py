import functools
import pathlib

import numpy as np
import polars as pl
import pydantic
import torch
import torch.nn as nn
import torch.utils.data as torch_data
import transformers
from datasets import Dataset
from sklearn import model_selection
from tqdm.auto import tqdm
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
)

from src import constants, log, utils

from .models import Atma17CustomModel

logger = log.get_root_logger()

EXP_NO = __file__.split("/")[-2]
DESCRIPTION = """
simple baseline
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

    lr: float = pydantic.Field(default=2e-5)
    num_train_epochs: int = pydantic.Field(default=5)
    per_device_train_batch_size: int = pydantic.Field(default=8)
    per_device_valid_batch_size: int = pydantic.Field(default=8)
    gradient_accumulation_steps: int = pydantic.Field(default=4)

    steps: int = pydantic.Field(default=25)


WEIGHT_PATHS = [
    f"output/{EXP_NO}/deverta-large-seed42-fold0",
    f"output/{EXP_NO}/deverta-large-seed42-fold1",
    f"output/{EXP_NO}/deverta-large-seed42-fold2",
    f"output/{EXP_NO}/deverta-large-seed42-fold3",
    f"output/{EXP_NO}/deverta-large-seed42-fold4",
]


def preprocessing(df: pl.DataFrame) -> pl.DataFrame:
    df = df.with_columns(
        pl.col("Title").fill_null("").alias("Title"),
        pl.col("Review Text").fill_null("").alias("Review Text"),
    ).with_columns(
        (pl.col("Title") + " [TITLE] " + pl.col("Review Text")).alias("prompt"),
    )
    return df


def tokenize(
    df: dict,
    tokenizer: transformers.PreTrainedTokenizer | transformers.PreTrainedTokenizerFast,
    max_length: int = 256,
    padding: str | bool = False,
) -> transformers.BatchEncoding:
    return tokenizer(df["prompt"], truncation=True, max_length=max_length, padding=padding)


def main() -> None:
    cfg = Config(is_debug=False)
    utils.pinfo(cfg.model_dump())

    test_df = pl.read_csv(cfg.test_csv_fp)
    test_df = preprocessing(test_df)
    if cfg.is_debug:
        test_df = test_df.head(100)

    tokenizers = []
    models = []
    for fold, weight_path in enumerate(WEIGHT_PATHS):
        tokenizer = AutoTokenizer.from_pretrained(weight_path)
        model = Atma17CustomModel(model_path=cfg.model_path)
        msg = model.load_state_dict(torch.load(weight_path + f"/best_model_fold{fold}.pth"))
        print(msg)
        tokenizers.append(tokenizer)
        models.append(model)

    ds_test = Dataset.from_pandas(test_df[["prompt"]].to_pandas())
    predictions = []
    for tokenizer, model in zip(tokenizers, models):
        ds_test = ds_test.map(
            functools.partial(tokenize, tokenizer=tokenizer, max_length=cfg.max_length, padding=False)
        )
        ds_test.set_format(type="torch", columns=["input_ids"])
        test_loader = torch_data.DataLoader(
            ds_test,  # type: ignore
            batch_size=cfg.per_device_valid_batch_size,
            collate_fn=DataCollatorWithPadding(tokenizer),
        )
        model = model.eval().to("cuda")
        preds = []
        for batch in tqdm(test_loader, total=len(test_loader)):
            batch = {k: v.to("cuda") for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)
            preds.append(outputs.logits)
        preds = torch.cat(preds, dim=0)
        predictions.append(preds)
    predictions = torch.stack(predictions, dim=1).mean(dim=1)
    predictions = torch.softmax(predictions.cpu(), dim=1).numpy()
    sub = pl.DataFrame({"target": predictions[:, 1]})
    print(sub)
    sub.write_csv(cfg.output_dir / "submission.csv")


if __name__ == "__main__":
    main()
