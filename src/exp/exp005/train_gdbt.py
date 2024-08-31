import pathlib

import polars as pl
import pydantic

from src import constants, log, utils

logger = log.get_root_logger()

EXP_NO = __file__.split("/")[-2]
DESCRIPTION = """
simple baseline
"""


class Config(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True, frozen=True)
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
    train_meta_csv_fp: pathlib.Path = pydantic.Field(default=constants.DATA_DIR / "clothing_meta.csv")
    test_csv_fp: pathlib.Path = pydantic.Field(default=constants.DATA_DIR / "test.csv")


def main() -> None:
    cfg = Config()
    utils.pinfo(cfg.model_dump())


if __name__ == "__main__":
    main()
