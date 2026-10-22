from pathlib import Path
import csv
import hashlib
import json
import re
import unicodedata

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
JIGSAW_DIR = ROOT / "data" / "raw" / "jigsaw"
SMS_PATH = ROOT / "data" / "raw" / "sms" / "SMSSpamCollection"
OUT = ROOT / "data" / "processed"

LABELS = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate",
]

COLS = [
    "id",
    "text",
    "source_dataset",
    "source_row_id",
    "spam_label",
    "spam_label_available",
    *LABELS,
    "toxicity_labels_available",
]

RANDOM_STATE = 42


def normalize_text(value):
    if pd.isna(value):
        return ""
    value = unicodedata.normalize("NFKC", str(value))
    value = value.replace("\u200b", "").replace("\ufeff", "")
    return re.sub(r"\s+", " ", value).strip()


def require_columns(df, columns, filename):
    missing = set(columns) - set(df.columns)
    if missing:
        raise ValueError(f"{filename} is missing columns: {sorted(missing)}")


def read_jigsaw_files():
    paths = {
        "train": JIGSAW_DIR / "train.csv",
        "test": JIGSAW_DIR / "test.csv",
        "labels": JIGSAW_DIR / "test_labels.csv",
        "sample_submission": JIGSAW_DIR / "sample_submission.csv",
    }

    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing Jigsaw {name} file: {path}")

    train_raw = pd.read_csv(paths["train"])
    test_raw = pd.read_csv(paths["test"])
    labels_raw = pd.read_csv(paths["labels"])
    sample_submission = pd.read_csv(paths["sample_submission"])

    require_columns(train_raw, ["id", "comment_text", *LABELS], "train.csv")
    require_columns(test_raw, ["id", "comment_text"], "test.csv")
    require_columns(labels_raw, ["id", *LABELS], "test_labels.csv")
    require_columns(
        sample_submission, ["id", *LABELS], "sample_submission.csv"
    )

    train = train_raw[["id", "comment_text", *LABELS]].rename(
        columns={"comment_text": "text"}
    )
    train["text"] = train["text"].map(normalize_text)
    train = train[train["text"].ne("")].copy()
    train = train.drop_duplicates(subset=["id"], keep="first")

    for label in LABELS:
        train[label] = pd.to_numeric(train[label], errors="coerce")

    train = train.dropna(subset=LABELS).copy()

    for label in LABELS:
        if not train[label].isin([0, 1]).all():
            raise ValueError(f"train.csv contains non-binary values in {label}")
        train[label] = train[label].astype("int8")

    train["source_dataset"] = "jigsaw_toxic_comment"
    train["source_row_id"] = train["id"].astype(str)
    train["spam_label"] = pd.NA
    train["spam_label_available"] = False
    train["toxicity_labels_available"] = True
    train = train[COLS].reset_index(drop=True)

    test = test_raw[["id", "comment_text"]].rename(
        columns={"comment_text": "text"}
    )
    test["text"] = test["text"].map(normalize_text)
    test = test[test["text"].ne("")].copy()
    test = test.drop_duplicates(subset=["id"], keep="first")
    test["source_dataset"] = "jigsaw_toxic_comment_test"
    test["source_row_id"] = test["id"].astype(str)
    test["spam_label"] = pd.NA
    test["spam_label_available"] = False

    for label in LABELS:
        test[label] = pd.NA

    test["toxicity_labels_available"] = False
    test = test[COLS].reset_index(drop=True)

    labels = labels_raw[["id", *LABELS]].copy()
    for label in LABELS:
        labels[label] = pd.to_numeric(labels[label], errors="coerce")

    labeled_test = test[["id", "text", "source_dataset", "source_row_id"]].merge(
        labels, on="id", how="inner", validate="one_to_one"
    )

    valid_mask = labeled_test[LABELS].isin([0, 1]).all(axis=1)
    labeled_test = labeled_test.loc[valid_mask].copy()

    for label in LABELS:
        labeled_test[label] = labeled_test[label].astype("int8")

    labeled_test["spam_label"] = pd.NA
    labeled_test["spam_label_available"] = False
    labeled_test["toxicity_labels_available"] = True
    labeled_test["source_dataset"] = "jigsaw_official_test_labeled"
    labeled_test = labeled_test[COLS].reset_index(drop=True)

    sample_submission.to_csv(
        OUT / "jigsaw_sample_submission.csv", index=False
    )

    return train, test, labeled_test, sample_submission, len(labels_raw)


def read_sms():
    if not SMS_PATH.is_file():
        raise FileNotFoundError(f"Missing SMS dataset: {SMS_PATH}")

    sms = pd.read_csv(
        SMS_PATH,
        sep="\t",
        header=None,
        names=["spam_label", "text"],
        encoding="utf-8",
        quoting=csv.QUOTE_NONE,
        on_bad_lines="error",
    )

    sms["spam_label"] = sms["spam_label"].astype("string").str.strip().str.lower()

    if not set(sms["spam_label"].dropna().unique()).issubset({"ham", "spam"}):
        raise ValueError("SMS dataset contains unexpected labels")

    sms["text"] = sms["text"].map(normalize_text)
    sms = sms[sms["text"].ne("")].drop_duplicates(
        subset=["spam_label", "text"], keep="first"
    ).copy()
    sms = sms.reset_index(drop=True)
    sms["id"] = pd.NA
    sms["source_dataset"] = "sms_spam_collection"
    sms["source_row_id"] = [
        hashlib.sha256(f"sms\0{label}\0{text}".encode()).hexdigest()[:20]
        for label, text in zip(sms["spam_label"], sms["text"])
    ]
    sms["spam_label_available"] = True
    sms["toxicity_labels_available"] = False
    sms["spam_label"] = sms["spam_label"].astype("string")

    for label in LABELS:
        sms[label] = pd.NA

    return sms[COLS].reset_index(drop=True)


def save_splits(df, target, prefix):
    if df.empty:
        raise ValueError(f"No rows available for {prefix} split.")
    if df[target].nunique() < 2:
        raise ValueError(f"{target} must contain at least two classes.")

    train, temporary = train_test_split(
        df,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=df[target],
    )

    validation, test = train_test_split(
        temporary,
        test_size=0.5,
        random_state=RANDOM_STATE,
        stratify=temporary[target],
    )

    for name, split in [
        ("train", train),
        ("validation", validation),
        ("test", test),
    ]:
        split.to_csv(OUT / f"{prefix}_{name}.csv", index=False)

    return train, validation, test


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    train, official_test, official_test_labeled, sample_submission, raw_test_label_rows = (
        read_jigsaw_files()
    )
    sms = read_sms()

    train.to_csv(OUT / "jigsaw_cleaned.csv", index=False)
    official_test.to_csv(OUT / "jigsaw_official_test.csv", index=False)
    official_test_labeled.to_csv(
        OUT / "jigsaw_official_test_labeled.csv", index=False
    )
    sms.to_csv(OUT / "sms_cleaned.csv", index=False)

    master = pd.concat([train, official_test, sms], ignore_index=True)
    master.to_csv(OUT / "unified_master.csv", index=False)

    toxicity_train, toxicity_validation, toxicity_test = save_splits(
        train, "toxic", "toxicity"
    )
    spam_train, spam_validation, spam_test = save_splits(
        sms, "spam_label", "spam"
    )

    manifest = {
        "jigsaw_train_clean_rows": int(len(train)),
        "jigsaw_official_test_rows": int(len(official_test)),
        "jigsaw_test_labels_raw_rows": int(raw_test_label_rows),
        "jigsaw_official_test_labeled_rows": int(len(official_test_labeled)),
        "jigsaw_sample_submission_rows": int(len(sample_submission)),
        "sms_clean_rows": int(len(sms)),
        "unified_master_rows": int(len(master)),
        "toxicity_train_rows": int(len(toxicity_train)),
        "toxicity_validation_rows": int(len(toxicity_validation)),
        "toxicity_test_rows": int(len(toxicity_test)),
        "spam_train_rows": int(len(spam_train)),
        "spam_validation_rows": int(len(spam_validation)),
        "spam_test_rows": int(len(spam_test)),
        "source_counts": master["source_dataset"].value_counts().to_dict(),
        "sms_label_counts": sms["spam_label"].value_counts().to_dict(),
        "jigsaw_positive_counts": {
            label: int(train[label].sum()) for label in LABELS
        },
        "official_test_label_policy": (
            "Rows containing -1 or missing values in any of the six labels "
            "are excluded from the labeled evaluation output."
        ),
        "sample_submission_policy": (
            "Saved for reference only; never used as training labels."
        ),
        "cleaning": (
            "Unicode NFKC; remove zero-width spaces and BOM; collapse "
            "whitespace; trim; preserve punctuation, case, negation, and emojis."
        ),
        "label_policy": (
            "Unknown labels remain missing; spam labels are not toxicity labels."
        ),
        "split_policy": "80/10/10 with random_state=42 and stratification.",
    }

    (OUT / "data_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, default=str))
    print(f"\nProcessed files saved to: {OUT}")


if __name__ == "__main__":
    main()
