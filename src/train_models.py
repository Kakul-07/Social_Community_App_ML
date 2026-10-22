from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import (
    classification_report,
    f1_score,
    fbeta_score,
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
OUT = ROOT / "artifacts"
OUT.mkdir(parents=True, exist_ok=True)

LABELS = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate",
]


def read_split(filename, required_columns):
    path = DATA / filename

    if not path.is_file():
        raise FileNotFoundError(
            f"Required file not found: {path}. Run python src/prepare_data.py first."
        )

    df = pd.read_csv(path)
    missing = set(required_columns) - set(df.columns)

    if missing:
        raise ValueError(
            f"{filename} is missing columns: {sorted(missing)}"
        )

    return df


def threshold_by_f1(y_true, probabilities, candidates):
    scores = [
        f1_score(
            y_true,
            probabilities >= threshold,
            zero_division=0,
        )
        for threshold in candidates
    ]

    return float(candidates[int(np.argmax(scores))])


def threshold_by_fbeta(
    y_true,
    probabilities,
    candidates,
    beta=2.0,
):
    scores = [
        fbeta_score(
            y_true,
            probabilities >= threshold,
            beta=beta,
            zero_division=0,
        )
        for threshold in candidates
    ]

    return float(candidates[int(np.argmax(scores))])


tox_train = read_split(
    "toxicity_train.csv",
    ["text", *LABELS],
)
tox_validation = read_split(
    "toxicity_validation.csv",
    ["text", *LABELS],
)
tox_test = read_split(
    "toxicity_test.csv",
    ["text", *LABELS],
)

for filename, df in [
    ("toxicity_train.csv", tox_train),
    ("toxicity_validation.csv", tox_validation),
    ("toxicity_test.csv", tox_test),
]:
    for label in LABELS:
        df[label] = pd.to_numeric(
            df[label],
            errors="coerce",
        )

    if df[LABELS].isna().any().any():
        raise ValueError(
            f"{filename} contains missing toxicity labels."
        )

    if not df[LABELS].isin([0, 1]).all().all():
        raise ValueError(
            f"{filename} contains non-binary toxicity labels."
        )

X_train_text = tox_train["text"].fillna("").astype(str)
X_validation_text = tox_validation["text"].fillna("").astype(str)
X_test_text = tox_test["text"].fillna("").astype(str)

Y_train = tox_train[LABELS].astype(int).to_numpy()
Y_validation = tox_validation[LABELS].astype(int).to_numpy()
Y_test = tox_test[LABELS].astype(int).to_numpy()

tox_features = FeatureUnion([
    (
        "word",
        TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=2,
            max_features=100000,
            sublinear_tf=True,
        ),
    ),
    (
        "char",
        TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=3,
            max_features=120000,
            sublinear_tf=True,
            dtype=np.float32,
        ),
    ),
])

X_train = tox_features.fit_transform(X_train_text)
X_validation = tox_features.transform(X_validation_text)
X_test = tox_features.transform(X_test_text)

tox_model = OneVsRestClassifier(
    LogisticRegression(max_iter=1000),
    n_jobs=1,
)

tox_model.fit(X_train, Y_train)

validation_probabilities = tox_model.predict_proba(
    X_validation
)

thresholds = {}

for index, label in enumerate(LABELS):
    thresholds[label] = threshold_by_f1(
        Y_validation[:, index],
        validation_probabilities[:, index],
        np.arange(0.10, 0.91, 0.02),
    )

test_probabilities = tox_model.predict_proba(X_test)

test_predictions = np.column_stack([
    test_probabilities[:, index] >= thresholds[label]
    for index, label in enumerate(LABELS)
]).astype(int)

tox_report = classification_report(
    Y_test,
    test_predictions,
    target_names=LABELS,
    output_dict=True,
    zero_division=0,
)

tox_result = {
    "macro_f1": float(
        f1_score(
            Y_test,
            test_predictions,
            average="macro",
            zero_division=0,
        )
    ),
    "micro_f1": float(
        f1_score(
            Y_test,
            test_predictions,
            average="micro",
            zero_division=0,
        )
    ),
    "classification_report": tox_report,
    "confusion_matrices": {
        label: confusion_matrix(
            Y_test[:, index],
            test_predictions[:, index],
        ).tolist()
        for index, label in enumerate(LABELS)
    },
    "thresholds": thresholds,
    "test_rows": int(len(tox_test)),
    "threshold_tuning": "Per-label F1 maximization on validation split only.",
    "test_source": "Stratified holdout from labeled train.csv, not official test.csv.",
}

joblib.dump(
    {
        "features": tox_features,
        "model": tox_model,
        "labels": LABELS,
        "thresholds": thresholds,
    },
    OUT / "toxicity_word_char_tfidf.joblib",
    compress=3,
)

(OUT / "toxicity_thresholds.json").write_text(
    json.dumps(thresholds, indent=2),
    encoding="utf-8",
)

(OUT / "toxicity_test_report.json").write_text(
    json.dumps(tox_result, indent=2),
    encoding="utf-8",
)


spam_train = read_split(
    "spam_train.csv",
    ["text", "spam_label"],
)
spam_validation = read_split(
    "spam_validation.csv",
    ["text", "spam_label"],
)
spam_test = read_split(
    "spam_test.csv",
    ["text", "spam_label"],
)


def spam_targets(df, filename):
    labels = (
        df["spam_label"]
        .astype("string")
        .str.lower()
        .str.strip()
    )

    if not labels.isin(["ham", "spam"]).all():
        raise ValueError(
            f"{filename} contains missing or invalid spam labels."
        )

    return (labels == "spam").astype(int).to_numpy()


y_spam_train = spam_targets(
    spam_train,
    "spam_train.csv",
)
y_spam_validation = spam_targets(
    spam_validation,
    "spam_validation.csv",
)
y_spam_test = spam_targets(
    spam_test,
    "spam_test.csv",
)

spam_features = FeatureUnion([
    (
        "word",
        TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=1,
            max_features=30000,
            sublinear_tf=True,
        ),
    ),
    (
        "char",
        TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=1,
            max_features=30000,
            sublinear_tf=True,
            dtype=np.float32,
        ),
    ),
])

X_spam_train = spam_features.fit_transform(
    spam_train["text"].fillna("").astype(str)
)

X_spam_validation = spam_features.transform(
    spam_validation["text"].fillna("").astype(str)
)

X_spam_test = spam_features.transform(
    spam_test["text"].fillna("").astype(str)
)

candidate_models = {
    "logistic_regression": LogisticRegression(
        C=4.0,
        class_weight="balanced",
        max_iter=1000,
        solver="liblinear",
    ),
    "complement_naive_bayes": ComplementNB(
        alpha=0.1,
    ),
}

threshold_candidates = np.arange(0.10, 0.91, 0.01)

model_comparison = []
best_model = None
best_model_name = None
best_threshold = 0.5
best_validation_f2 = -1.0

for model_name, candidate in candidate_models.items():
    print(f"\nTraining spam model: {model_name}")

    candidate.fit(
        X_spam_train,
        y_spam_train,
    )

    validation_probabilities = candidate.predict_proba(
        X_spam_validation
    )[:, 1]

    candidate_threshold = threshold_by_fbeta(
        y_spam_validation,
        validation_probabilities,
        threshold_candidates,
        beta=2.0,
    )

    validation_predictions = (
        validation_probabilities >= candidate_threshold
    ).astype(int)

    precision = precision_score(
        y_spam_validation,
        validation_predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_spam_validation,
        validation_predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y_spam_validation,
        validation_predictions,
        zero_division=0,
    )

    f2 = fbeta_score(
        y_spam_validation,
        validation_predictions,
        beta=2.0,
        zero_division=0,
    )

    row = {
        "model": model_name,
        "threshold": float(candidate_threshold),
        "validation_precision": float(precision),
        "validation_recall": float(recall),
        "validation_f1": float(f1),
        "validation_f2": float(f2),
    }

    model_comparison.append(row)

    print(
        f"Threshold={candidate_threshold:.2f} | "
        f"Precision={precision:.4f} | "
        f"Recall={recall:.4f} | "
        f"F1={f1:.4f} | "
        f"F2={f2:.4f}"
    )

    if f2 > best_validation_f2:
        best_validation_f2 = float(f2)
        best_model = candidate
        best_model_name = model_name
        best_threshold = float(candidate_threshold)

spam_test_probabilities = best_model.predict_proba(
    X_spam_test
)[:, 1]

spam_predictions = (
    spam_test_probabilities >= best_threshold
).astype(int)

spam_report = classification_report(
    y_spam_test,
    spam_predictions,
    labels=[0, 1],
    target_names=["ham", "spam"],
    output_dict=True,
    zero_division=0,
)

spam_result = {
    "selected_model": best_model_name,
    "accuracy": float(
        accuracy_score(
            y_spam_test,
            spam_predictions,
        )
    ),
    "spam_precision": float(
        precision_score(
            y_spam_test,
            spam_predictions,
            zero_division=0,
        )
    ),
    "spam_recall": float(
        recall_score(
            y_spam_test,
            spam_predictions,
            zero_division=0,
        )
    ),
    "spam_f1": float(
        f1_score(
            y_spam_test,
            spam_predictions,
            zero_division=0,
        )
    ),
    "classification_report": spam_report,
    "confusion_matrix": confusion_matrix(
        y_spam_test,
        spam_predictions,
        labels=[0, 1],
    ).tolist(),
    "threshold": best_threshold,
    "validation_f2": best_validation_f2,
    "model_comparison": model_comparison,
    "test_rows": int(len(spam_test)),
    "threshold_tuning": (
        "F2 maximization on validation split only; "
        "test split was not used to choose the threshold."
    ),
}

joblib.dump(
    {
        "features": spam_features,
        "model": best_model,
        "threshold": best_threshold,
        "labels": ["ham", "spam"],
        "model_name": best_model_name,
    },
    OUT / "sms_spam_tfidf_logistic_regression.joblib",
    compress=3,
)

(OUT / "sms_spam_threshold.json").write_text(
    json.dumps(
        {
            "spam_threshold": best_threshold,
            "selected_model": best_model_name,
            "tuning_metric": "validation F2",
        },
        indent=2,
    ),
    encoding="utf-8",
)

(OUT / "sms_spam_test_report.json").write_text(
    json.dumps(spam_result, indent=2),
    encoding="utf-8",
)

(OUT / "sms_spam_model_comparison.json").write_text(
    json.dumps(model_comparison, indent=2),
    encoding="utf-8",
)

print("\nTraining finished. Reports and thresholds are saved in artifacts.")

print(
    "Toxicity macro F1:",
    round(tox_result["macro_f1"], 4),
    "| micro F1:",
    round(tox_result["micro_f1"], 4),
)

print(
    "Selected spam model:",
    best_model_name,
    "| threshold:",
    round(best_threshold, 4),
)

print(
    "SMS spam accuracy:",
    round(spam_result["accuracy"], 4),
    "| spam precision:",
    round(spam_result["spam_precision"], 4),
    "| spam recall:",
    round(spam_result["spam_recall"], 4),
    "| spam F1:",
    round(spam_result["spam_f1"], 4),
)
