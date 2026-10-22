# Smart Community ML — Simple App-Based Deployment


## Dataset paths
- `data/raw/jigsaw/train.csv`
- `data/raw/jigsaw/test.csv`
- `data/raw/jigsaw/test_labels.csv`
- `data/raw/jigsaw/sample_submission.csv`
- `data/raw/sms/SMSSpamCollection`

The names follow the uploaded FastAPI example; despite the filenames, this notebook's models are Multinomial Naive Bayes baselines.

## Install and run
From the project root:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```
Swagger: `http://127.0.0.1:8000/docs`
Health: `http://127.0.0.1:8000/health`

## One prediction endpoint
`POST /predict`
```json
{"text":"Thanks for helping me."}
```

## Deployment
For a standard Python web service on Render, use:
- Build command: `pip install -r requirements.txt`
- Start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
- Health check path: `/health`

## Streamlit configuration

