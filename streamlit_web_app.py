import streamlit as st
import joblib as jb
from sklearn.feature_extraction.text import TfidfVectorizer

global model 
model=jb.load("./ML_Models/randomForestModel1.joblib")
global vectorizer 
vectorizer=jb.load("./ML_Models/vectorizer.joblib")

def analyse(text):
    text = vectorizer.transform([text])
    analysis=model.predict_proba(text)
    analysis=analysis.item(1)
    if analysis>0.5:
        st.title("Sentiment: Positive")
    elif analysis==0.5:
        st.title("Sentiment: Neutral")
    else:
        st.title("Sentiment: Negative")

def sentiment_analysis_webapp():
    st.title("Social Media Sentiment Analysis")

    text=st.text_area("Enter the Text for Sentiment Analysis:")
    if st.button("Analyse Sentiment"):
        analyse(text)


sentiment_analysis_webapp()