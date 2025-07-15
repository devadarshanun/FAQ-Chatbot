
import pandas as pd
import re
import string
import nltk
import pickle
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download NLTK assets
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

# === Text Cleaning ===
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(f"[{re.escape(string.punctuation)}]", '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [stemmer.stem(word) for word in tokens]
    return " ".join(tokens)

# === Load & Train Model ===
@st.cache_data(show_spinner=False)
def train_model():
    df = pd.read_csv("faq_dataset_100k_improved.csv")
    df.dropna(inplace=True)
    df['cleaned_question'] = df['question'].apply(clean_text)

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df['cleaned_question'])

    os.makedirs("models", exist_ok=True)
    with open("models/tfidf_vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)

    with open("models/faq_model.pkl", "wb") as f:
        pickle.dump((X, df), f)

    return vectorizer, X, df

# === Chatbot Logic ===
def get_response(query, vectorizer, X, df, top_n=1):
    cleaned = clean_text(query)
    vec = vectorizer.transform([cleaned])
    scores = cosine_similarity(vec, X)
    top_indices = scores.argsort()[0][-top_n:][::-1]
    return df.iloc[top_indices[0]]['answer']

# === Main Interface ===
def main():
    st.set_page_config(page_title="AI FAQ Chatbot", layout="centered")
    st.title("🤖 AI-Powered FAQ Chatbot")
    st.markdown("Ask me anything related to our services and get instant answers!")

    with st.spinner("Training or loading model..."):
        vectorizer, X, df = train_model()

    user_input = st.text_input("Enter your question:")

    if user_input:
        response = get_response(user_input, vectorizer, X, df)
        st.success(f"**Answer:** {response}")

if __name__ == "__main__":
    main()
