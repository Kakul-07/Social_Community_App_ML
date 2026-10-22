from pathlib import Path
from typing import Any

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

# Project layout:
# project-root/
#   app/main.py
#   artifacts/<trained model files>
PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

TOXICITY_MODEL_PATH = ARTIFACTS_DIR / "toxicity_word_char_tfidf.joblib"
SPAM_MODEL_PATH = ARTIFACTS_DIR / "sms_spam_tfidf_logistic_regression.joblib"

app = FastAPI(
    title="Smart Community Moderation API",
    version="2.0.0",
    description="Health/status routes and one POST endpoint for toxicity and spam predictions.",
)


class PredictionRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)
    text: str = Field(
        ...,
        min_length=1,
        max_length=20000,
        description="Community comment to check for toxicity and spam.",
    )


def load_artifact(path: Path) -> Any | None:
    """Load an artifact if it exists; let the service start so / can explain status."""
    if not path.is_file():
        return None
    try:
        return joblib.load(path)
    except Exception as exc:
        print(f"Could not load model artifact {path.name}: {exc}")
        return None


# Load models once at startup. Raw training CSVs are not needed for inference.
toxicity_bundle = load_artifact(TOXICITY_MODEL_PATH)
spam_bundle = load_artifact(SPAM_MODEL_PATH)


def model_status() -> dict:
    toxicity_loaded = isinstance(toxicity_bundle, dict) and all(
        key in toxicity_bundle for key in ("features", "model", "labels", "thresholds")
    )
    spam_loaded = isinstance(spam_bundle, dict) and all(
        key in spam_bundle for key in ("features", "model", "threshold")
    )
    return {
        "status": "ready" if toxicity_loaded and spam_loaded else "degraded",
        "toxicity_model_loaded": toxicity_loaded,
        "spam_model_loaded": spam_loaded,
        "prediction_ready": toxicity_loaded and spam_loaded,
    }


@app.get("/")
def root():
    """Friendly landing page for opening the base Render URL in a browser."""
    state = model_status()
    return {
        "application": "Smart Community Moderation API",
        "status": state["status"],
        "message": (
            "API is running and ready for predictions."
            if state["prediction_ready"]
            else "API process is running, but one or more trained model artifacts are missing or invalid."
        ),
        "endpoints": {
            "this_status": "/",
            "health": "/health",
            "prediction": {
                "path": "/predict",
                "method": "POST",
                "example_request": {"text": "Thanks for helping me."},
            },
            "documentation": "/docs",
            "openapi_schema": "/openapi.json",
        },
        "models": {
            "toxicity_artifact_exists": TOXICITY_MODEL_PATH.is_file(),
            "spam_artifact_exists": SPAM_MODEL_PATH.is_file(),
            **state,
        },
    }


@app.get("/health")
def health():
    """Render health check; returns 200 while API process is alive."""
    state = model_status()
    return {
        "service": "Smart Community Moderation API",
        **state,
    }


@app.post("/predict")
def predict(request: PredictionRequest):
    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=422, detail="Comment must not be blank.")

    state = model_status()
    if not state["prediction_ready"]:
        raise HTTPException(
            status_code=503,
            detail=(
                "Prediction models are not ready. Ensure these files are deployed "
                "and match the training bundle format: "
                "artifacts/toxicity_word_char_tfidf.joblib and "
                "artifacts/sms_spam_tfidf_logistic_regression.joblib."
            ),
        )

    try:
        # Toxicity bundle is expected to contain features, model, labels, thresholds.
        tox_x = toxicity_bundle["features"].transform([text])
        tox_probs = toxicity_bundle["model"].predict_proba(tox_x)[0]
        tox_labels = list(toxicity_bundle["labels"])
        tox_thresholds = toxicity_bundle["thresholds"]

        tox_probabilities = {
            label: round(float(prob), 6)
            for label, prob in zip(tox_labels, tox_probs)
        }
        tox_detected_labels = [
            label for label, probability in tox_probabilities.items()
            if probability >= float(tox_thresholds.get(label, 0.5))
        ]

        # Spam bundle is expected to contain features, model, threshold.
        spam_x = spam_bundle["features"].transform([text])
        spam_model = spam_bundle["model"]
        spam_probs = spam_model.predict_proba(spam_x)[0]
        classes = list(spam_model.classes_)
        spam_index = next(
            (
                i for i, cls in enumerate(classes)
                if str(cls).strip().lower() in {"1", "spam", "true"}
            ),
            None,
        )
        if spam_index is None:
            if len(classes) < 2:
                raise ValueError(f"Spam model has unexpected classes: {classes}")
            spam_index = 1

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
                "detected_labels": tox_detected_labels,
                "labels": {label: label in tox_detected_labels for label in tox_labels},
                "probabilities": tox_probabilities,
            },
            "spam": {
                "detected": bool(spam_detected),
                "probability": round(spam_probability, 6),
                "threshold": round(spam_threshold, 6),
            },
            "review": needs_review,
            "status": "REVIEW" if needs_review else "PASSED",
            "reason": reasons,
        }
    except HTTPException:
        raise
    except Exception as exc:
        # Keep internal paths/stack traces out of the public API response.
        print(f"Prediction error: {type(exc).__name__}: {exc}")
        raise HTTPException(
            status_code=500,
            detail="Prediction failed because the deployed model artifact format is incompatible.",
        ) from exc
