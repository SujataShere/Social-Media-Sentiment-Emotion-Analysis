import streamlit as st
import pandas as pd
import joblib
import re
import string
import nltk

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Social Media Sentiment & Emotion Analysis",
    page_icon="💬",
    layout="wide"
)


# =========================================================
# NLTK SETUP
# =========================================================

@st.cache_resource
def download_nltk():

    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)
    nltk.download("stopwords", quiet=True)
    nltk.download("wordnet", quiet=True)
    nltk.download("omw-1.4", quiet=True)


download_nltk()


# =========================================================
# LOAD MODELS
# =========================================================

@st.cache_resource
def load_models():

    sentiment_tfidf = joblib.load(
        "sentiment_tfidf.pkl"
    )

    sentiment_model = joblib.load(
        "sentiment_model.pkl"
    )

    emotion_tfidf = joblib.load(
        "emotion_tfidf.pkl"
    )

    emotion_model = joblib.load(
        "emotion_model.pkl"
    )

    return (
        sentiment_tfidf,
        sentiment_model,
        emotion_tfidf,
        emotion_model
    )


try:

    (
        sentiment_tfidf,
        sentiment_model,
        emotion_tfidf,
        emotion_model
    ) = load_models()

    models_loaded = True

except Exception as e:

    models_loaded = False

    st.error(
        "Model files could not be loaded."
    )

    st.info(
        "Make sure these four files are in the same folder as app.py:\n\n"
        "sentiment_tfidf.pkl\n"
        "sentiment_model.pkl\n"
        "emotion_tfidf.pkl\n"
        "emotion_model.pkl"
    )

    st.stop()


# =========================================================
# TEXT CLEANING
# =========================================================

stop_words = set(
    stopwords.words("english")
)

lemmatizer = WordNetLemmatizer()


def clean_text(text):

    text = str(text)

    # Convert to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(
        r"http\S+|www\S+|https\S+",
        "",
        text
    )

    # Remove mentions
    text = re.sub(
        r"@\w+",
        "",
        text
    )

    # Remove hashtags symbol but keep word
    text = re.sub(
        r"#",
        "",
        text
    )

    # Remove numbers
    text = re.sub(
        r"\d+",
        "",
        text
    )

    # Remove punctuation
    text = text.translate(
        str.maketrans(
            "",
            "",
            string.punctuation
        )
    )

    # Tokenization
    tokens = word_tokenize(text)

    # Remove stopwords
    tokens = [
        word
        for word in tokens
        if word not in stop_words
    ]

    # Lemmatization
    tokens = [
        lemmatizer.lemmatize(word)
        for word in tokens
    ]

    return " ".join(tokens)


# =========================================================
# SENTIMENT PREDICTION
# =========================================================

def predict_sentiment(text):

    cleaned = clean_text(text)

    vector = sentiment_tfidf.transform(
        [cleaned]
    )

    prediction = sentiment_model.predict(
        vector
    )[0]

    probabilities = sentiment_model.predict_proba(
        vector
    )[0]

    confidence = max(probabilities)

    return prediction, confidence


# =========================================================
# EMOTION PREDICTION
# =========================================================

def predict_emotion(text):

    cleaned = clean_text(text)

    vector = emotion_tfidf.transform(
        [cleaned]
    )

    prediction = emotion_model.predict(
        vector
    )[0]

    probabilities = emotion_model.predict_proba(
        vector
    )[0]

    confidence = max(probabilities)

    return prediction, confidence


# =========================================================
# HEADER
# =========================================================

st.title(
    "💬 Social Media Sentiment & Emotion Analysis"
)

st.write(
    "Analyze social media text using "
    "TF-IDF, Logistic Regression and NLP preprocessing."
)

st.divider()


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title(
    "📌 Project Information"
)

st.sidebar.write(
    """
    **Project:** Social Media Sentiment & Emotion Analysis

    **Techniques:**
    - NLP
    - NLTK
    - TF-IDF
    - Logistic Regression
    - Sentiment Classification
    - Emotion Classification
    """
)

st.sidebar.success(
    "Models loaded successfully!"
)


# =========================================================
# TEXT INPUT
# =========================================================

st.subheader(
    "✍️ Enter Social Media Text"
)

text = st.text_area(
    "Enter a tweet, post or comment:",
    placeholder="Example: I am extremely happy today because I got my dream job!",
    height=150
)


# =========================================================
# ANALYZE BUTTON
# =========================================================

if st.button(
    "🔍 Analyze Text",
    type="primary"
):

    if not text.strip():

        st.warning(
            "Please enter some text first."
        )

    else:

        # Prediction
        sentiment, sentiment_confidence = predict_sentiment(
            text
        )

        emotion, emotion_confidence = predict_emotion(
            text
        )

        # =================================================
        # RESULTS
        # =================================================

        st.subheader(
            "📊 Analysis Result"
        )

        col1, col2 = st.columns(2)

        # Sentiment
        with col1:

            st.markdown(
                "### 😊 Sentiment"
            )

            if sentiment == "Positive":

                st.success(
                    f"Positive"
                )

            else:

                st.error(
                    f"Negative"
                )

            st.metric(
                "Confidence",
                f"{sentiment_confidence * 100:.2f}%"
            )


        # Emotion
        with col2:

            st.markdown(
                "### 🎭 Emotion"
            )

            st.info(
                f"{emotion}"
            )

            st.metric(
                "Confidence",
                f"{emotion_confidence * 100:.2f}%"
            )


        # =================================================
        # CLEANED TEXT
        # =================================================

        st.divider()

        st.subheader(
            "🧹 Processed Text"
        )

        cleaned = clean_text(text)

        st.code(
            cleaned
        )


        # =================================================
        # RESULT TABLE
        # =================================================

        result = pd.DataFrame({

            "Text": [text],

            "Sentiment": [sentiment],

            "Sentiment Confidence": [
                round(
                    sentiment_confidence,
                    4
                )
            ],

            "Emotion": [emotion],

            "Emotion Confidence": [
                round(
                    emotion_confidence,
                    4
                )
            ]

        })

        st.subheader(
            "📋 Prediction Details"
        )

        st.dataframe(
            result,
            use_container_width=True
        )


# =========================================================
# EXAMPLE TEXTS
# =========================================================

st.divider()

st.subheader(
    "💡 Try These Examples"
)

examples = [

    "I am extremely happy today because I got my dream job!",

    "I am scared about what will happen tomorrow.",

    "I hate this service.",

    "I love this product, it is amazing!"

]

for example in examples:

    st.write(
        "•",
        example
    )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "Social Media Sentiment & Emotion Analysis | "
    "Python • NLTK • TF-IDF • Scikit-learn • Streamlit"
)
