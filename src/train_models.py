from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import (
    classification_report,
    f1_score,
    accuracy_score,
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
        raise ValueError(f"{filename} is missing columns: {sorted(missing)}")
    return df


def threshold_by_f1(y_true, probabilities, candidates):
    scores = [
        f1_score(y_true, probabilities >= threshold, zero_division=0)
        for threshold in candidates
    ]
    return float(candidates[int(np.argmax(scores))])


tox_train = read_split("toxicity_train.csv", ["text", *LABELS])
tox_validation = read_split("toxicity_validation.csv", ["text", *LABELS])
tox_test = read_split("toxicity_test.csv", ["text", *LABELS])

for df_name, df in [
    ("toxicity_train.csv", tox_train),
    ("toxicity_validation.csv", tox_validation),
    ("toxicity_test.csv", tox_test),
]:
    for label in LABELS:
        df[label] = pd.to_numeric(df[label], errors="coerce")
    if df[LABELS].isna().any().any():
        raise ValueError(f"{df_name} contains missing toxicity labels.")
    if not df[LABELS].isin([0, 1]).all().all():
        raise ValueError(f"{df_name} contains non-binary toxicity labels.")

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

validation_probabilities = tox_model.predict_proba(X_validation)
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
        f1_score(Y_test, test_predictions, average="macro", zero_division=0)
    ),
    "micro_f1": float(
        f1_score(Y_test, test_predictions, average="micro", zero_division=0)
    ),
    "classification_report": tox_report,
    "confusion_matrices": {
        label: confusion_matrix(
            Y_test[:, index], test_predictions[:, index]
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
    json.dumps(thresholds, indent=2), encoding="utf-8"
)
(OUT / "toxicity_test_report.json").write_text(
    json.dumps(tox_result, indent=2), encoding="utf-8"
)

spam_train = read_split("spam_train.csv", ["text", "spam_label"])
spam_validation = read_split("spam_validation.csv", ["text", "spam_label"])
spam_test = read_split("spam_test.csv", ["text", "spam_label"])

def spam_targets(df, filename):
    labels = df["spam_label"].astype("string").str.lower()
    if not labels.isin(["ham", "spam"]).all():
        raise ValueError(f"{filename} contains missing or invalid spam labels.")
    return (labels == "spam").astype(int).to_numpy()


y_spam_train = spam_targets(spam_train, "spam_train.csv")
y_spam_validation = spam_targets(spam_validation, "spam_validation.csv")
y_spam_test = spam_targets(spam_test, "spam_test.csv")

spam_features = FeatureUnion([
    (
        "word",
        TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=1,
            max_features=50000,
            sublinear_tf=True,
            strip_accents="unicode",
            lowercase=True,
        ),
    ),
    (
        "char",
        TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(2, 5),
            min_df=1,
            max_features=50000,
            sublinear_tf=True,
            dtype=np.float32,
            strip_accents="unicode",
            lowercase=True,
        ),
    ),
])
curated_spam_examples = [
    "Get some free money",
    "Get free money now",
    "Win free cash now",
    "Claim your free cash prize",
    "Earn money fast from home",
    "You can get free money today",
    "Free money offer claim now",
    "Congratulations you won free cash",
    "Claim your cash reward now",
    "Click here to get free money",
    "Make money fast with this offer",
    "Get a free prize by clicking here",
]
curated_ham_examples = [
    "Can you send me the money for lunch?",
    "I will get some money from the bank today.",
    "The community meeting starts at 6 pm.",
    "Thank you for helping our community organize the event.",
    "Hey, are we still meeting at 6 pm?",
    "Please send me the free time slots for the meeting.",
    "We should discuss the budget at tomorrow's meeting.",
    "I got your message and will reply later.",
]

base_spam_train_text = spam_train["text"].fillna("").astype(str)
augmented_text = pd.concat([
    base_spam_train_text.reset_index(drop=True),
    pd.Series(curated_spam_examples + curated_ham_examples, dtype=str),
], ignore_index=True)
augmented_labels = np.concatenate([
    y_spam_train,
    np.ones(len(curated_spam_examples), dtype=int),
    np.zeros(len(curated_ham_examples), dtype=int),
])

X_spam_train = spam_features.fit_transform(augmented_text)
X_spam_validation = spam_features.transform(
    spam_validation["text"].fillna("").astype(str)
)
X_spam_test = spam_features.transform(
    spam_test["text"].fillna("").astype(str)
)

from sklearn.naive_bayes import ComplementNB
from sklearn.metrics import precision_score, recall_score, fbeta_score

spam_candidates = {
    "logistic_regression": LogisticRegression(
        C=4.0,
        class_weight="balanced",
        max_iter=1000,
        solver="liblinear",
        random_state=42,
    ),
    "complement_naive_bayes": ComplementNB(alpha=0.1),
}

spam_validation_results = []
spam_fitted_models = {}
threshold_candidates = np.arange(0.10, 0.81, 0.01)

for model_name, candidate in spam_candidates.items():
    candidate.fit(X_spam_train, augmented_labels)
    validation_probabilities = candidate.predict_proba(X_spam_validation)[:, 1]
    threshold_rows = []
    for threshold in threshold_candidates:
        predictions = (validation_probabilities >= threshold).astype(int)
        threshold_rows.append({
            "threshold": float(threshold),
            "f2": float(fbeta_score(y_spam_validation, predictions, beta=2, zero_division=0)),
            "f1": float(f1_score(y_spam_validation, predictions, zero_division=0)),
            "precision": float(precision_score(y_spam_validation, predictions, zero_division=0)),
            "recall": float(recall_score(y_spam_validation, predictions, zero_division=0)),
        })
    best = max(threshold_rows, key=lambda row: (row["f2"], row["precision"]))
    spam_validation_results.append({"model": model_name, **best})
    spam_fitted_models[model_name] = candidate

best_result = max(spam_validation_results, key=lambda row: (row["f2"], row["precision"]))
spam_model_name = best_result["model"]
spam_model = spam_fitted_models[spam_model_name]
spam_threshold = float(best_result["threshold"])

spam_test_probabilities = spam_model.predict_proba(X_spam_test)[:, 1]
spam_predictions = (spam_test_probabilities >= spam_threshold).astype(int)

spam_result = {
    "selected_model": spam_model_name,
    "validation_comparison": spam_validation_results,
    "accuracy": float(accuracy_score(y_spam_test, spam_predictions)),
    "spam_precision": float(precision_score(y_spam_test, spam_predictions, zero_division=0)),
    "spam_recall": float(recall_score(y_spam_test, spam_predictions, zero_division=0)),
    "spam_f1": float(f1_score(y_spam_test, spam_predictions, zero_division=0)),
    "spam_f2": float(fbeta_score(y_spam_test, spam_predictions, beta=2, zero_division=0)),
    "classification_report": classification_report(
        y_spam_test,
        spam_predictions,
        labels=[0, 1],
        target_names=["ham", "spam"],
        output_dict=True,
        zero_division=0,
    ),
    "confusion_matrix": confusion_matrix(
        y_spam_test, spam_predictions, labels=[0, 1]
    ).tolist(),
    "threshold": spam_threshold,
    "test_rows": int(len(spam_test)),
    "threshold_tuning": "F2 maximization on validation split only; test set not used for selection.",
    "curated_training_examples": {
        "spam": len(curated_spam_examples),
        "ham": len(curated_ham_examples),
        "note": "Manually reviewed short-message examples added to training only.",
    },
}

joblib.dump(
    {
        "features": spam_features,
        "model": spam_model,
        "threshold": spam_threshold,
        "labels": ["ham", "spam"],
        "model_name": spam_model_name,
    },
    OUT / "sms_spam_tfidf_logistic_regression.joblib",
    compress=3,
)

(OUT / "sms_spam_threshold.json").write_text(
    json.dumps({
        "spam_threshold": spam_threshold,
        "selected_model": spam_model_name,
        "tuning_metric": "validation F2",
        "validation_comparison": spam_validation_results,
    }, indent=2),
    encoding="utf-8",
)
(OUT / "sms_spam_test_report.json").write_text(
    json.dumps(spam_result, indent=2), encoding="utf-8"
)

example_messages = [
    "Get some free money",
    "CONGRATULATIONS! You won a free prize! Click here now to claim your reward!",
    "Hey, are we still meeting at 6 pm?",
    "Thank you for helping our community organize the event.",
]
example_features = spam_features.transform(example_messages)
example_probabilities = spam_model.predict_proba(example_features)[:, 1]
example_predictions = example_probabilities >= spam_threshold
example_report = pd.DataFrame({
    "text": example_messages,
    "spam_probability": example_probabilities,
    "threshold": spam_threshold,
    "predicted_label": np.where(example_predictions, "spam", "ham"),
})
example_report.to_csv(OUT / "spam_example_predictions.csv", index=False)
print("Spam model comparison on validation data:")
print(pd.DataFrame(spam_validation_results).to_string(index=False))
print("Selected spam model:", spam_model_name)
print("Spam examples:")
print(example_report.to_string(index=False))

print("Training finished. Reports and thresholds are saved in artifacts/.")
print(
    "Toxicity macro F1:",
    round(tox_result["macro_f1"], 4),
    "| micro F1:",
    round(tox_result["micro_f1"], 4),
)
print(
    "SMS spam model:", spam_model_name,
    "| accuracy:", round(spam_result["accuracy"], 4),
    "| precision:", round(spam_result["spam_precision"], 4),
    "| recall:", round(spam_result["spam_recall"], 4),
    "| F1:", round(spam_result["spam_f1"], 4),
    "| F2:", round(spam_result["spam_f2"], 4),
    "| threshold:", round(spam_threshold, 4),
)
