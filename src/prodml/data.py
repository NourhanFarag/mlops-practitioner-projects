from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from prodml.config import settings


def load_data(path: Path | None = None) -> pd.DataFrame:
    data_path = path or settings.data_path

    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found at {data_path}")

    return pd.read_parquet(data_path)


def split_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df, val_df = train_test_split(
        df,
        test_size=settings.validation_size,
        random_state=settings.random_state,
    )

    return train_df, val_df
