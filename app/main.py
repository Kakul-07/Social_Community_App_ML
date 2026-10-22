from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
TOXICITY_MODEL_PATH = ARTIFACTS / "toxicity_word_char_tfidf.joblib"
SPAM_MODEL_PATH = ARTIFACTS / "sms_spam_tfidf_logistic_regression.joblib"

app = FastAPI(
    title="Smart Community Moderation API",
    version="1.1.0",
    description="One prediction endpoint for spam and toxicity detection.",
)

class PredictionRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)
    text: str = Field(..., min_length=1, max_length=20000, description="Community comment to moderate")

# Load model artifacts once when the service starts.
toxicity_bundle = joblib.load(TOXICITY_MODEL_PATH) if TOXICITY_MODEL_PATH.is_file() else None
spam_bundle = joblib.load(SPAM_MODEL_PATH) if SPAM_MODEL_PATH.is_file() else None

@app.get("/health")
def health():
    return {
        "status": "ok" if toxicity_bundle is not None and spam_bundle is not None else "degraded",
        "toxicity_model_loaded": toxicity_bundle is not None,
        "spam_model_loaded": spam_bundle is not None,
    }

@app.post("/predict")
def predict(request: PredictionRequest):
    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=422, detail="Comment must not be blank.")
    if toxicity_bundle is None or spam_bundle is None:
        raise HTTPException(
            status_code=503,
            detail="Model artifacts are missing. Run notebook 03 to train and save both models in the artifacts/ folder.",
        )

    try:
        # Predict toxicity labels. The notebook saves features, model, thresholds and labels.
        tox_x = toxicity_bundle["features"].transform([text])
        tox_probs = toxicity_bundle["model"].predict_proba(tox_x)[0]
        tox_labels = toxicity_bundle["labels"]
        tox_thresholds = toxicity_bundle["thresholds"]
        tox_probabilities = {label: round(float(prob), 6) for label, prob in zip(tox_labels, tox_probs)}
        tox_detected_labels = [
            label for label, prob in tox_probabilities.items()
            if prob >= float(tox_thresholds[label])
        ]

        # Find the spam class explicitly instead of assuming class index 1 is spam.
        spam_x = spam_bundle["features"].transform([text])
        spam_model = spam_bundle["model"]
        spam_probs = spam_model.predict_proba(spam_x)[0]
        classes = list(spam_model.classes_)
        spam_index = next((i for i, cls in enumerate(classes) if str(cls).lower() in {"1", "spam"}), None)
        if spam_index is None:
            spam_index = 1 if len(classes) > 1 else 0
        spam_probability = float(spam_probs[spam_index])
        spam_threshold = float(spam_bundle["threshold"])
        spam_detected = spam_probability >= spam_threshold

        reasons = []
        if tox_detected_labels:
            reasons.append("TOXICITY_DETECTED")
        if spam_detected:
            reasons.append("SPAM_DETECTED")
        if not reasons:
            reasons.append("NO_HARMFUL_LABELS_DETECTED")
        needs_review = bool(tox_detected_labels or spam_detected)

        return {
            "comment": text,
            "toxicity": {
                "detected": bool(tox_detected_labels),
                "labels": {label: label in tox_detected_labels for label in tox_labels},
                "probabilities": tox_probabilities,
                "detected_labels": tox_detected_labels,
            },
            "spam": {
                "detected": spam_detected,
                "probability": round(spam_probability, 6),
                "threshold": round(spam_threshold, 6),
            },
            "review": needs_review,
            "status": "REVIEW" if needs_review else "PASSED",
            "reason": reasons,
        }
    except KeyError as exc:
        raise HTTPException(status_code=500, detail=f"Model artifact format mismatch: missing key {exc}.") from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc
