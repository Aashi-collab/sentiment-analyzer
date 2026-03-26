import streamlit as st
import joblib

# Page config
st.set_page_config(page_title="Sentiment Analyzer", page_icon="💬", layout="centered")

# Load pipeline
pipeline = joblib.load('pipeline (2).pkl')

# Custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #0E1117;
        color: white;
    }
    .stTextArea textarea {
        border-radius: 10px;
        font-size: 16px;
    }
    .stButton button {
        background-color: #FF4B4B;
        color: white;
        border-radius: 10px;
        height: 50px;
        width: 100%;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.title("💬 AI Emotion Analyzer")
st.write("Analyze emotions in text using Machine Learning 🚀")

# Input box
user_input = st.text_area("Enter your text here:", height=150)

# Button
if st.button("Analyze Emotion"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text")
    else:
        prediction = pipeline.predict([user_input])[0]

        emotion_map = {
            0: "Sadness 😢",
            1: "Joy 😊",
            2: "Love ❤️",
            3: "Anger 😡",
            4: "Fear 😨",
            5: "Surprise 😲"
        }

        result = emotion_map[prediction]

        st.success(f"Detected Emotion: {result}")

col1, col2 = st.columns(2)

with col1:
    st.write("### ✍️ Input")

with col2:
    st.write("### 🎯 Result")

    proba = pipeline.predict_proba([user_input])[0]
confidence = max(proba)

st.info(f"Confidence: {confidence:.2f}")

st.sidebar.title("About")
st.sidebar.write("This is an AI-based Emotion Analyzer using NLP & ML.")
st.sidebar.write("Built with ❤️ using Streamlit")