from __future__ import annotations

from pathlib import Path
from urllib.request import urlretrieve

import pandas as pd

DATA_URL = (
    "https://raw.githubusercontent.com/LaurentVeyssier/"
    "Starbucks_case_study_Udacity_Data_Science/master/training.csv"
)
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PATH = ROOT / "data" / "training.csv"
EXPECTED_COLUMNS = {"ID", "Promotion", "purchase", *{f"V{i}" for i in range(1, 8)}}


def validate(df: pd.DataFrame) -> pd.DataFrame:
    missing = EXPECTED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if not set(df["Promotion"].unique()).issubset({"Yes", "No"}):
        raise ValueError("Promotion must contain only Yes/No")
    if not set(df["purchase"].unique()).issubset({0, 1}):
        raise ValueError("purchase must be binary")
    return df


def load(path: str | Path = DEFAULT_PATH) -> pd.DataFrame:
    return validate(pd.read_csv(path))


def download(path: str | Path = DEFAULT_PATH) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        urlretrieve(DATA_URL, path)
    validate(pd.read_csv(path, nrows=1000))
    return path


def main() -> None:
    path = download()
    df = load(path)
    print(f"Saved {len(df):,} rows to {path}")


if __name__ == "__main__":
    main()
