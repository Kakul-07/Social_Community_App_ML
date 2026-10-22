from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException

from app.schemas.moderation import PredictionRequest

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"

TOXICITY_MODEL_PATH = ARTIFACTS / "toxicity_word_char_tfidf.joblib"
SPAM_MODEL_PATH = ARTIFACTS / "sms_spam_tfidf_logistic_regression.joblib"

app = FastAPI(
    title="Smart Community Comment Moderation API",
    version="1.0.0",
    description="Detects toxic comments and spam using one common endpoint.",
)

toxicity_bundle = (
    joblib.load(TOXICITY_MODEL_PATH) if TOXICITY_MODEL_PATH.is_file() else None
)
spam_bundle = (
    joblib.load(SPAM_MODEL_PATH) if SPAM_MODEL_PATH.is_file() else None
)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "toxicity_model_loaded": toxicity_bundle is not None,
        "spam_model_loaded": spam_bundle is not None,
    }


@app.post("/api/v1/moderation/predict")
def moderate_comment(request: PredictionRequest):
    text = request.text.strip()

    if not text:
        raise HTTPException(
            status_code=422,
            detail="Comment must not be blank.",
        )

    if toxicity_bundle is None or spam_bundle is None:
        raise HTTPException(
            status_code=503,
            detail="One or both models are missing. Train the models first.",
        )

    toxicity_features = toxicity_bundle["features"].transform([text])
    toxicity_probabilities = toxicity_bundle["model"].predict_proba(
        toxicity_features
    )[0]

    toxicity_thresholds = toxicity_bundle["thresholds"]
    toxicity_labels = toxicity_bundle["labels"]

    toxicity_results = {
        label: {
            "probability": round(float(probability), 6),
            "detected": bool(
                float(probability) >= float(toxicity_thresholds[label])
            ),
        }
        for label, probability in zip(
            toxicity_labels, toxicity_probabilities
        )
    }

    detected_toxicity_labels = [
        label
        for label, result in toxicity_results.items()
        if result["detected"]
    ]

    spam_features = spam_bundle["features"].transform([text])
    spam_probability = float(
        spam_bundle["model"].predict_proba(spam_features)[0, 1]
    )
    spam_threshold = float(spam_bundle["threshold"])
    spam_detected = spam_probability >= spam_threshold

    review_reasons = []
    if detected_toxicity_labels:
        review_reasons.append("TOXICITY_DETECTED")
    if spam_detected:
        review_reasons.append("SPAM_DETECTED")
    if not review_reasons:
        review_reasons.append("NO_HARMFUL_LABELS_DETECTED")

    requires_review = bool(detected_toxicity_labels or spam_detected)

    return {
        "comment": text,
        "toxicity": {
            "detected": bool(detected_toxicity_labels),
            "labels": {
                label: result["detected"]
                for label, result in toxicity_results.items()
            },
            "probabilities": {
                label: result["probability"]
                for label, result in toxicity_results.items()
            },
            "detected_labels": detected_toxicity_labels,
        },
        "spam": {
            "detected": bool(spam_detected),
            "probability": round(spam_probability, 6),
            "threshold": round(spam_threshold, 6),
        },
        "review": requires_review,
        "status": "REVIEW" if requires_review else "PASSED",
        "reason": review_reasons,
    }
