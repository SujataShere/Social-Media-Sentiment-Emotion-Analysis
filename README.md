💬 Social Media Sentiment & Emotion Analysis
A Natural Language Processing (NLP) web application that analyzes social media text and predicts both sentiment and emotion using TF-IDF and Logistic Regression.
The application is built with Python, NLTK, Scikit-learn, Joblib, Pandas, and Streamlit.

---

📌 Project Overview
This project accepts a tweet, post, or comment as input and performs:
1. Text preprocessing
2. TF-IDF feature transformation
3. Sentiment classification
4. Emotion classification
5. Confidence-score calculation
6. Display of the cleaned text and prediction details
The Streamlit application loads separate TF-IDF vectorizers and trained models for sentiment and emotion prediction.

---

✨ Features
- 😊 Sentiment Classification
  - Positive
  - Negative
- 🎭 Emotion Classification
  - Predicts the emotion learned by the trained emotion model
- 🧹 NLP Text Preprocessing
  - Converts text to lowercase
  - Removes URLs
  - Removes mentions
  - Removes the # symbol while keeping hashtag words
  - Removes numbers
  - Removes punctuation
  - Tokenizes text
  - Removes English stopwords
  - Performs lemmatization
- 📊 Confidence Scores
  - Displays sentiment confidence
  - Displays emotion confidence
- 📋 Prediction Details
  - Shows the original text
  - Predicted sentiment
  - Sentiment confidence
  - Predicted emotion
  - Emotion confidence
- 🖥️ Interactive Streamlit UI
  - Text input area
  - Analyze button
  - Results dashboard
  - Example texts

---

🛠️ Technologies Used
Technology	Purpose
Python	Programming language
Pandas	Data handling and result table
NLTK	NLP preprocessing
Scikit-learn	TF-IDF and Logistic Regression models
Joblib	Saving and loading trained models
Streamlit	Web application interface
Regular Expressions	Text cleaning

---

🔄 Project Workflow
Social Media Text
       ↓
Text Cleaning
       ↓
Lowercase
       ↓
Remove URLs / Mentions / Numbers / Punctuation
       ↓
Tokenization
       ↓
Stopword Removal
       ↓
Lemmatization
       ↓
TF-IDF Transformation
       ↓
┌───────────────────────┬───────────────────────┐
│ Sentiment TF-IDF      │ Emotion TF-IDF        │
↓                       ↓
Sentiment Model         Emotion Model
↓                       ↓
Positive / Negative     Predicted Emotion
└───────────────────────┴───────────────────────┘
       ↓
Confidence Scores
       ↓
Streamlit Results

---

📂 Project Structure
Social-Media-Sentiment-Emotion-Analysis/
│
├── app.py
├── Social Media Sentiment & Emotion.ipynb
│
├── sentiment_tfidf.pkl
├── sentiment_model.pkl
├── emotion_tfidf.pkl
├── emotion_model.pkl
│
├── requirements.txt
└── README.md
The four .pkl files are required by the Streamlit application. The app loads them from the same folder as app.py.

---

🧠 NLP Preprocessing
The application uses NLTK to prepare the input text before prediction.
The preprocessing pipeline includes:
Raw Text
   ↓
Lowercase
   ↓
Remove URLs
   ↓
Remove @mentions
   ↓
Remove # symbol
   ↓
Remove numbers
   ↓
Remove punctuation
   ↓
Tokenization
   ↓
Stopword Removal
   ↓
Lemmatization
   ↓
Cleaned Text
For example:
I finally completed my project, and I feel so proud of myself!
can be processed into:
finally completed project feel proud

---

🤖 Machine Learning Models
Sentiment Model
The application uses:
- TF-IDF vectorization
- Logistic Regression
- Binary sentiment prediction:
  - Positive
  - Negative
Emotion Model
The emotion pipeline also uses:
- TF-IDF vectorization
- Logistic Regression
- Emotion prediction based on the trained model
Both models return a prediction and a probability-based confidence score.

---

📊 Example
Input:
I finally completed my project, and I feel so proud of myself!
Example application result:
Sentiment: Positive
Sentiment Confidence: 94.95%

Emotion: Joy
Emotion Confidence: 99.65%
The project demonstration shows the processed text as:
finally completed project feel proud

---

🖥️ Application Screens
The Streamlit interface contains:
- Project information sidebar
- Social media text input
- Analyze Text button
- Sentiment result
- Emotion result
- Confidence scores
- Processed text
- Prediction details table
- Example inputs