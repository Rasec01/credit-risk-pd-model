"""Load the data and split it into a development sample and a test sample."""
import re

import pandas as pd
from sklearn.model_selection import train_test_split

TARGET = "default"


def clean_name(name: str) -> str:
    """'ROA(C) before interest...' -> 'roa_c_before_interest...'"""
    return re.sub(r"[^0-9a-z]+", "_", name.strip().lower()).strip("_")


def load_data(path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [clean_name(c) for c in df.columns]
    if TARGET not in df.columns:
        raise ValueError(f"The file needs a '{TARGET}' column (1 = defaulted, 0 = not).")
    df = df.dropna(subset=[TARGET])
    df[TARGET] = df[TARGET].astype(int)
    # Constant columns carry no information
    return df.loc[:, df.nunique() > 1]


def split_data(df: pd.DataFrame, test_size: float = 0.3, seed: int = 42):
    """
    Stratified split: the default rate is the same in both samples. Everything
    that learns from the data (bins, weights, model) uses the development sample only.
    """
    dev, test = train_test_split(df, test_size=test_size, stratify=df[TARGET], random_state=seed)
    X_cols = [c for c in df.columns if c != TARGET]
    return dev[X_cols], test[X_cols], dev[TARGET].to_numpy(), test[TARGET].to_numpy()
