import streamlit as st
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Load the dataset
column_names = ["exp", "id", "time", "query", "id-name", "comments"]
df = pd.read_csv("C://Users//DELL//Desktop//sentiment//sentiment.csv", encoding="latin1", names=column_names)

# Preprocessing
def preprocess_text(text):
    pr = r'[^\w\s]'
    text = re.sub(pr, ' ', text)
    pe = r'http'
    text = re.sub(pe, ' ', text)
    pe1 = "[0-9]"
    text = re.sub(pe1, ' ', text)
    
    stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in text.split() if word not in stop_words])
    
    porter_stemmer = PorterStemmer()
    text = ' '.join([porter_stemmer.stem(word) for word in word_tokenize(text)])
    
    return text

df['processed_comments'] = df['comments'].apply(preprocess_text)

# Split data
X = df['processed_comments']
Y = df['exp']

# TF-IDF Vectorization
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(X)

# Model training
lr = LogisticRegression()
lr.fit(tfidf_matrix, Y)

def predict_sentiment(text):
    text = preprocess_text(text)
    text_count = tfidf_vectorizer.transform([text])
    prediction = lr.predict(text_count)[0]
    if prediction == 0:
        result = "Your feedback is Negative"
    else:
        result = "Your feedback is Positive"
    return result
# Streamlit UI
st.title('Sentiment analysis on text')

text_input = st.text_input('Enter your text here:')
if st.button('Predict'):
    prediction = predict_sentiment(text_input)
    st.write('Predicted Sentiment:', prediction)
