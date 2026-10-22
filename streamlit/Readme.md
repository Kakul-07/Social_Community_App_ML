# Smart Community ML — Streamlit Interface

This application provides a web interface for the
Smart Community machine learning moderation API.

## Features

- Analyze community comments
- Detect potentially toxic language
- Display toxicity label probabilities
- Detect potential spam
- Display moderation status
- View the complete API response

## Technology Stack

- Python
- Streamlit
- Requests
- FastAPI
- Scikit-learn

## ML API

Base URL:

https://group-03-social-community-app-xuoc.onrender.com

Prediction endpoint:

POST /api/v1/moderation/predict

Health endpoint:

GET /health

## Run Locally

From the repository root:

```bash
python -m pip install -r streamlit/requirements.txt
streamlit run streamlit/streamlit_app.py