import os

import requests
import streamlit as st

st.set_page_config(page_title="Smart Community AI Moderation", page_icon="🛡️", layout="centered")
st.title("🛡️ Smart Community — Comment Moderation")
st.caption("This tool supports moderator review; predictions should not be the sole basis for punishment.")

# Configure ML_API_URL in Streamlit Community Cloud → App settings → Secrets.
# The API URL is intentionally never displayed in the page.
try:
    API_BASE_URL = str(st.secrets.get("ML_API_URL", "")).strip().rstrip("/")
except Exception:
    API_BASE_URL = ""
API_BASE_URL = API_BASE_URL or os.getenv("ML_API_URL", "").strip().rstrip("/")

text_value = st.text_area("Enter a community comment", height=150, max_chars=20000, placeholder="Type a comment to check for toxicity or spam...")

if st.button("Analyze comment", type="primary", use_container_width=True):
    if not text_value.strip():
        st.warning("Please enter a comment first.")
    elif not API_BASE_URL:
        st.error("The moderation API is not configured. Add ML_API_URL in the app's Secrets settings and restart the app.")
    else:
        try:
            with st.spinner("Analyzing comment..."):
                response = requests.post(
                    f"{API_BASE_URL}/predict",
                    json={"text": text_value.strip()},
                    timeout=(10, 120),
                )
            if not response.ok:
                try:
                    detail = response.json().get("detail", "The API request failed.")
                except ValueError:
                    detail = "The API request failed. Please try again later."
                st.error(f"Prediction failed (HTTP {response.status_code}): {detail}")
            else:
                result = response.json()
                if result.get("status") == "REVIEW":
                    st.warning("Flagged for human moderator review.")
                else:
                    st.success("No configured toxicity or spam threshold was crossed.")

                st.subheader("Moderation result")
                st.metric("Decision", result.get("status", "UNKNOWN"))

                toxicity = result.get("toxicity", {})
                spam = result.get("spam", {})
                left, right = st.columns(2)
                with left:
                    st.markdown("**Toxicity**")
                    st.write("Detected:", "Yes" if toxicity.get("detected") else "No")
                    labels = toxicity.get("detected_labels", [])
                    st.write("Detected labels:", ", ".join(labels) if labels else "None")
                    if toxicity.get("probabilities"):
                        with st.expander("Toxicity probabilities"):
                            st.json(toxicity["probabilities"])
                with right:
                    st.markdown("**Spam**")
                    st.write("Detected:", "Yes" if spam.get("detected") else "No")
                    st.write("Spam probability:", spam.get("probability", "N/A"))
                    st.write("Threshold:", spam.get("threshold", "N/A"))

                if result.get("reason"):
                    st.caption("Reason: " + ", ".join(result["reason"]))
        except requests.Timeout:
            st.error("The moderation API took too long to respond. Please try again.")
        except requests.RequestException:
            st.error("Could not connect to the moderation API. Check the deployment status and try again.")
