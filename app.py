import streamlit as st
import pickle
import re
import nltk
import os

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ---------------- NLTK SETUP ----------------
# Download only if not available (prevents re-download every run)
nltk_data_path = os.path.join(os.getcwd(), "nltk_data")
if not os.path.exists(nltk_data_path):
    nltk.download('stopwords')
    nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# ---------------- TEXT CLEANING FUNCTION ----------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z]", " ", text)  # remove numbers & punctuation
    words = text.split()
    words = [w for w in words if w not in stop_words]
    words = [lemmatizer.lemmatize(w) for w in words]
    return " ".join(words)

# ---------------- LOAD MODEL & VECTORIZER ----------------
try:
    model = pickle.load(open("model.pkl", "rb"))
    vectorizer = pickle.load(open("tfidf.pkl", "rb"))
except:
    st.error("Model files not found. Please ensure .pkl files are in the same folder.")
    st.stop()

# ---------------- STREAMLIT UI ----------------
st.set_page_config(page_title="Food Review", page_icon="📝")

st.title(" Naive Bayes Food Review Classifier")
st.write("Enter a Food review to predict whether it is Liked or Not.")

review = st.text_area("Enter your review :")

if st.button("Analyze Food Review"):

    if review.strip() == "":
        st.warning("⚠ Please enter a review text.")
    else:
        review_clean = clean_text(review)
        vector_input = vectorizer.transform([review_clean])
        prediction = model.predict(vector_input)

        if prediction[0] == 1:
            st.success("😊 Positive Review")
        else:
            st.error("😠 Negative Review")

        # Optional: Show probability
        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(vector_input)
            st.write(f"Confidence: {round(max(prob[0])*100, 2)}%")
