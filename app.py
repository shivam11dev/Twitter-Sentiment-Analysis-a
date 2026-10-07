import streamlit as st
import joblib

# Page settings
st.set_page_config(page_title="Twitter Sentiment", page_icon="🕊️")
st.title("Twitter Sentiment Analysis")

# Load the trained model
try:
    model = joblib.load("models/sentiment.pkl")
except Exception:
    st.error("⚠️ Model not found. Please run: python src/train.py first to train and save the model.")
    st.stop()

# Text area for user input
tweet = st.text_area("Enter tweet or sentence here:", height=150)

# Button to analyze
if st.button("Analyze"):
    if not tweet.strip():
        st.warning("Please type something first.")
    else:
        # Predict label
        label = model.predict([tweet])[0]

        # (optional) predict probabilities
        try:
            probas = dict(zip(model.classes_, model.predict_proba([tweet])[0]))
        except Exception:
            probas = None

        # Show result nicely
        if label == 'positive':
            st.success("Prediction: Positive ✅")
        elif label == 'negative':
            st.error("Prediction: Negative ❌")
        else:
            st.info("Prediction: Neutral ⚪")

        # Show probabilities if available
        if probas:
            st.write("Class probabilities:", probas)
