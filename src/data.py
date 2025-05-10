"""SMS Spam Collection — download, load and split.

The dataset is the UCI SMS Spam Collection (Almeida & Hidalgo, 2011): 5,574
labelled SMS messages, ~13% spam. Stored locally as a tab-separated file with
two columns: label (ham/spam) and text.
"""
from __future__ import annotations

import csv
import io
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

DATA_URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
DATA_FILE = DATA_DIR / "SMSSpamCollection"


def download_if_missing(url: str = DATA_URL, target_dir: Path = DATA_DIR) -> Path:
    """Fetch and extract the SMS Spam zip if the dataset isn't already on disk."""
    target_dir.mkdir(parents=True, exist_ok=True)
    if DATA_FILE.exists():
        return DATA_FILE
    with urllib.request.urlopen(url) as resp:
        archive = zipfile.ZipFile(io.BytesIO(resp.read()))
        archive.extractall(target_dir)
    return DATA_FILE


def load_dataset(path: Path = DATA_FILE) -> pd.DataFrame:
    """Return the dataset as a DataFrame with columns ['label', 'text', 'is_spam']."""
    if not path.exists():
        download_if_missing()
    df = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=["label", "text"],
        quoting=csv.QUOTE_NONE,
    )
    df["is_spam"] = (df["label"] == "spam").astype(int)
    return df


def split_train_test(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified split on the spam label so both folds keep the ~13% prior."""
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df["is_spam"],
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


if __name__ == "__main__":
    df = load_dataset()
    train_df, test_df = split_train_test(df)
    print(f"Total:  {len(df):>5}  ({df['is_spam'].mean():.1%} spam)")
    print(f"Train:  {len(train_df):>5}  ({train_df['is_spam'].mean():.1%} spam)")
    print(f"Test:   {len(test_df):>5}  ({test_df['is_spam'].mean():.1%} spam)")
