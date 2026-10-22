# Smart Community ML

### Machine Learning Service for Comment Moderation

Smart Community ML is a machine learning component of the Smart Community
application. It analyzes user-generated text and helps identify potentially
toxic, abusive, or spam-related comments before they are published.

The project combines text preprocessing, exploratory data analysis (EDA),
feature engineering, machine learning, model evaluation, and a FastAPI
service to provide moderation predictions.

The goal is to provide a practical, explainable, and maintainable machine
learning pipeline that can be integrated with the main Smart Community
Spring Boot backend.

---

## Table of Contents

1. Project Overview
2. Objectives
3. Features
4. Technology Stack
5. Project Architecture
6. Project Structure
7. Datasets
8. Machine Learning Workflow
9. Data Preparation
10. Exploratory Data Analysis
11. Feature Engineering
12. Model Training
13. Model Evaluation
14. Model Artifacts
15. API Documentation
16. API Response Format
17. Installation and Setup
18. Running the Project
19. Testing the API
20. Spring Boot Integration
21. Configuration
22. Git and Dataset Management
23. Limitations
24. Future Improvements
25. Contributors

---

# 1. Project Overview

Online communities allow users to communicate through comments, posts,
discussions, and messages.

However, user-generated content can contain:

- Toxic language
- Insults
- Threats
- Obscene language
- Identity-based hate
- Spam messages

Manually reviewing every comment can be difficult as the community grows.

Smart Community ML addresses this problem by using machine learning to
analyze submitted text and generate moderation predictions.

The system contains two classification components:

1. **Toxicity Classification**
   - Identifies potentially toxic or abusive language.
   - Predicts multiple toxicity categories.

2. **Spam Classification**
   - Identifies text that resembles spam.
   - Produces a spam probability and a predicted classification.

The results are combined into a single moderation decision.

A comment is assigned one of two statuses:

- `PASSED`: Neither classifier detected a condition that triggers review.
- `REVIEW`: At least one classifier detected a condition that triggers review.

These predictions support moderation decisions. They do not guarantee that
a comment is harmless or that a flagged comment necessarily violates
community rules.

---

# 2. Objectives

The main objectives of this project are:

- Build a machine learning pipeline for text moderation.
- Work with real-world text classification datasets.
- Perform data cleaning and preprocessing.
- Explore datasets using exploratory data analysis.
- Convert text into numerical features using TF-IDF.
- Train machine learning classification models.
- Evaluate models using appropriate classification metrics.
- Save trained models for reuse.
- Expose predictions through a REST API.
- Combine toxicity and spam predictions into one moderation response.
- Support integration with a Spring Boot backend.
- Organize the project into understandable and maintainable components.

---

# 3. Features

## 3.1 Data Processing

- Load raw datasets.
- Inspect dataset structure and labels.
- Clean text data.
- Handle missing and invalid text.
- Prepare a unified dataset.
- Create training, validation, and test datasets where applicable.

## 3.2 Exploratory Data Analysis

- Inspect dataset dimensions and columns.
- Examine class distributions.
- Identify missing values.
- Analyze text lengths.
- Investigate class imbalance.
- Visualize relevant dataset characteristics.

## 3.3 Toxicity Classification

The toxicity classifier predicts the following categories:

| Label | Description |
|---|---|
| `toxic` | General toxic language |
| `severe_toxic` | Severely toxic language |
| `obscene` | Potentially obscene language |
| `threat` | Potentially threatening language |
| `insult` | Insulting language |
| `identity_hate` | Identity-based hateful language |

A single comment may be associated with more than one category.

## 3.4 Spam Classification

The spam classifier determines whether a message resembles spam.

It returns:

- A spam probability.
- A predicted spam classification.
- A Boolean spam detection result used by the combined moderation endpoint.

## 3.5 Combined Moderation

The API combines both classification results.

The final moderation logic is:

- If any toxicity label is detected, return `REVIEW`.
- If spam is detected, return `REVIEW`.
- Otherwise, return `PASSED`.

This allows the backend to consume one consistent moderation response.

## 3.6 FAST API

The FastAPI application provides:

- A health-check endpoint.
- A combined moderation prediction endpoint.
- JSON request and response formats.
- Input validation.
- Model loading from saved artifacts.

---

# 4. Technology Stack

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Pandas | Dataset loading and manipulation |
| NumPy | Numerical operations |
| Matplotlib | Data visualization |
| Seaborn | Exploratory data visualization |
| Scikit-learn | Machine learning and evaluation |
| TF-IDF | Text feature extraction |
| Logistic Regression | Text classification |
| OneVsRestClassifier | Multi-label toxicity classification |
| Joblib | Saving and loading trained models |
| Jupyter Notebook | Interactive exploration and learning |
| FastAPI | REST API development |
| Pydantic | Request validation |
| Uvicorn | Running the API server |
| Pytest | Automated testing |

The exact dependencies are defined in `requirements.txt`.

