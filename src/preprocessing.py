import re
from pathlib import Path

import pandas as pd


def clean_text(text):
    """Basic text cleaning for ticket descriptions."""
    if text is None:
        return ""

    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_and_clean_data(path="../data/tickets.csv"):
    data_path = Path(path)
    df = pd.read_csv(data_path)

    required = ["ticket_description", "category", "priority", "status"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    print("Dataset shape:", df.shape)
    print("\nFirst 5 records:")
    print(df.head())

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nDuplicate records:", df.duplicated().sum())
    df = df.drop_duplicates().copy()

    for col in required:
        df[col] = df[col].fillna("Unknown").astype(str).str.strip()

    df["category"] = df["category"].replace({"": "Unknown"})
    df["priority"] = df["priority"].replace({"": "Unknown"})
    df["status"] = df["status"].replace({"": "Unknown"})
    df["clean_description"] = df["ticket_description"].apply(clean_text)

    print("\nDataset information:")
    df.info()

    return df


if __name__ == "__main__":
    load_and_clean_data()
