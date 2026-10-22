import os
import requests
import streamlit as st


DEFAULT_API_URL = (
    "https://group-03-social-community-app-xuoc.onrender.com"
)

PREDICTION_ENDPOINT = "/api/v1/moderation/predict"
HEALTH_ENDPOINT = "/health"



st.set_page_config(
    page_title="Smart Community ML",
    layout="centered",
)

st.title(" Smart Community")
st.header("AI Comment Moderation")

st.write(
    "Analyze community comments for potentially harmful "
    "language and spam using machine learning."
)



with st.sidebar:
    st.subheader("API Configuration")

    api_url = st.text_input(
        "ML API Base URL",
        value=os.getenv(
            "ML_API_BASE_URL",
            DEFAULT_API_URL,
        ),
    ).strip().rstrip("/")

    st.caption(
        "Predictions are processed by the FastAPI service "
        "deployed on Render."
    )

    if st.button("Check API Health"):
        try:
            response = requests.get(
                f"{api_url}{HEALTH_ENDPOINT}",
                timeout=30,
            )

            st.write(
                f"HTTP status: {response.status_code}"
            )

            try:
                st.json(response.json())
            except ValueError:
                st.code(response.text)

        except requests.RequestException as exc:
            st.error(f"API connection failed: {exc}")

comment = st.text_area(
    "Enter a community comment",
    placeholder=(
        "Example: Thank you for helping with "
        "the community event."
    ),
    height=150,
    max_chars=10000,
)

if st.button(
    "Analyze Comment",
    type="primary",
    use_container_width=True,
):

    if not api_url:
        st.error("Please enter the API base URL.")

    elif not comment.strip():
        st.warning("Please enter a comment.")

    else:

        payload = {
            "text": comment.strip()
        }

        prediction_url = (
            f"{api_url}{PREDICTION_ENDPOINT}"
        )

        try:
            with st.spinner(
                "Analyzing comment..."
            ):
                response = requests.post(
                    prediction_url,
                    json=payload,
                    timeout=60,
                )

            if not response.ok:

                st.error(
                    f"API returned HTTP "
                    f"{response.status_code}"
                )

                try:
                    st.json(response.json())
                except ValueError:
                    st.code(response.text)

            else:

                result = response.json()

                # Overall moderation status
                status = result.get(
                    "status",
                    "UNKNOWN",
                )

                if status == "REVIEW":
                    st.error(
                        "Comment flagged for review"
                    )

                elif status == "PASSED":
                    st.success(
                        " No configured toxicity "
                        "or spam threshold was triggered"
                    )

                else:
                    st.info(
                        f"Moderation status: {status}"
                    )

                # Result sections
                toxicity = result.get(
                    "toxicity",
                    {},
                )

                spam = result.get(
                    "spam",
                    {},
                )

                col1, col2 = st.columns(2)

                with col1:
                    st.metric(
                        "Toxicity detected",
                        (
                            "Yes"
                            if toxicity.get("detected")
                            else "No"
                        ),
                    )

                with col2:
                    st.metric(
                        "Spam detected",
                        (
                            "Yes"
                            if spam.get("detected")
                            else "No"
                        ),
                    )

                # Toxicity labels
                st.subheader("Toxicity Analysis")

                detected_labels = toxicity.get(
                    "detected_labels",
                    [],
                )

                if detected_labels:
                    st.write(
                        "Detected labels: "
                        + ", ".join(detected_labels)
                    )
                else:
                    st.write(
                        "No toxicity labels crossed "
                        "their configured thresholds."
                    )

                probabilities = toxicity.get(
                    "probabilities",
                    {},
                )

                if probabilities:

                    st.write(
                        "Toxicity label probabilities"
                    )

                    for label, probability in (
                        probabilities.items()
                    ):

                        probability = float(
                            probability
                        )

                        st.progress(
                            max(
                                0.0,
                                min(1.0, probability),
                            ),
                            text=(
                                f"{label}: "
                                f"{probability:.1%}"
                            ),
                        )

                # Spam probability
                st.subheader("Spam Analysis")

                spam_probability = spam.get(
                    "probability"
                )

                if spam_probability is not None:

                    spam_probability = float(
                        spam_probability
                    )

                    st.progress(
                        max(
                            0.0,
                            min(1.0, spam_probability),
                        ),
                        text=(
                            "Spam probability: "
                            f"{spam_probability:.1%}"
                        ),
                    )

                reasons = result.get(
                    "reason",
                    [],
                )

                if reasons:

                    st.subheader(
                        "Moderation Reasons"
                    )

                    for reason in reasons:
                        st.write(f"- {reason}")
                        
                with st.expander(
                    "View complete API response"
                ):
                    st.json(result)

        except requests.RequestException as exc:

            st.error(
                f"Could not connect to the ML API: {exc}"
            )

        except (ValueError, TypeError) as exc:

            st.error(
                f"Could not process the API response: {exc}"
            )

st.divider()
