import streamlit as st
import pandas as pd
import string
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="centered"
)

# -----------------------------
# Custom Styling
# -----------------------------
st.markdown("""
    <style>
        .main {
            background-color: #f5f7fa;
        }
        .title {
            text-align: center;
            color: #2C3E50;
            font-size: 36px;
            font-weight: bold;
        }
        .subtitle {
            text-align: center;
            color: #7F8C8D;
            font-size: 18px;
        }
        .result-box {
            padding: 15px;
            border-radius: 10px;
            text-align: center;
            font-size: 20px;
            font-weight: bold;
        }
    </style>
""", unsafe_allow_html=True)

# -----------------------------
# NLTK Setup
# -----------------------------
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# -----------------------------
# Text Cleaning Function
# -----------------------------
def clean_text(text):
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

# -----------------------------
# Load Dataset
# -----------------------------
@st.cache_data
def load_data():
    data = pd.read_csv("fake_news.csv")
    data['cleaned_text'] = data['text'].apply(clean_text)
    return data

data = load_data()

# -----------------------------
# Train Model
# -----------------------------
@st.cache_resource
def train_model():
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(data['cleaned_text'])
    y = data['label']

    model = LogisticRegression(class_weight='balanced', max_iter=1000)
    model.fit(X, y)

    return model, vectorizer

model, vectorizer = train_model()

# -----------------------------
# UI Layout
# -----------------------------
st.markdown("<div class='title'>📰 Fake News Detection System</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Detect whether a news article is Real or Fake using AI</div>", unsafe_allow_html=True)

st.write("")
news_input = st.text_area("✍ Enter News Article:", height=150)

# -----------------------------
# Prediction Logic
# -----------------------------
if st.button("🔍 Check Credibility"):
    if news_input.strip() == "":
        st.warning("⚠ Please enter some news text.")
    else:
        cleaned = clean_text(news_input)
        vectorized = vectorizer.transform([cleaned])

        # ✅ Improved Prediction Logic
        proba = model.predict_proba(vectorized)[0]
        prediction = model.classes_[proba.argmax()]
        probability = proba.max()

        st.write("")

        if prediction == "FAKE":
            st.markdown(
                "<div class='result-box' style='background-color:#FFCDD2; color:#C62828;'>🚨 FAKE NEWS DETECTED</div>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                "<div class='result-box' style='background-color:#C8E6C9; color:#2E7D32;'>✅ REAL NEWS</div>",
                unsafe_allow_html=True
            )

        st.write("")
        st.info(f"📊 Confidence Score: {probability*100:.2f}%")

# -----------------------------
# Footer
# -----------------------------
st.write("")
st.markdown(
    "<div style='text-align:center; color:gray; font-size:14px;'>Built using NLP + Machine Learning</div>",
    unsafe_allow_html=True
)