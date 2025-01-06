import streamlit as st
import pickle
from nltk.corpus import stopwords
import nltk
from nltk.stem import PorterStemmer
import re
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
nltk.download('stopwords')

with open('Default risk Analysis/log_reg.pkl', 'rb') as file:
    model = pickle.load(file)

model_path = 'Sentiment Analysis/my_model.keras'
sent_model = load_model(model_path)

with open('Sentiment Analysis/tokenizer.pkl', 'rb') as file:
    tokenizer = pickle.load(file)


def process_inputs(SEX: str, MARRIAGE: str, EDUCATION: str):
    if MARRIAGE == 'married':
        MARRIAGE = 1
    elif MARRIAGE == 'single':
        MARRIAGE = 2
    else:
        MARRIAGE = 3

    if SEX == 'male':
        SEX = 2
    elif SEX == 'female':
        SEX = 1
    else:
        SEX = 3

    if EDUCATION == 'graduate':
        EDUCATION = 1
    elif EDUCATION == 'school':
        EDUCATION = 2
    elif EDUCATION == 'university':
        EDUCATION = 3
    else:
        EDUCATION = 4

    return SEX, MARRIAGE, EDUCATION


def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    tokens = text.split()

    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]

    stemmer = PorterStemmer()
    tokens = [stemmer.stem(token) for token in tokens]

    return ' '.join(tokens)


st.title("Credit Score and Sentiment Analysis")

st.subheader("Default Risk Analysis")

AGR = st.text_input("Enter your Age in years")
SEX = st.selectbox("Gender:", ["male", "female", "other"])
EDUCATION = st.selectbox("Education Level:", ["graduate", "school", "university", "others"])
MARRIAGE = st.selectbox("Marital Status:", ["married", "single", "others"])
PAY_0 = st.number_input("PAY_0 (Last month payment status):", value=0)
PAY_2 = st.number_input("PAY_2 (Two months ago payment status):", value=0)
PAY_3 = st.number_input("PAY_3 (Three months ago payment status):", value=0)
PAY_4 = st.number_input("PAY_4 (Four months ago payment status):", value=0)
PAY_5 = st.number_input("PAY_5 (Five months ago payment status):", value=0)
PAY_6 = st.number_input("PAY_6 (Six months ago payment status):", value=0)

st.subheader("Sentiment Analysis")

st.text_input("Enter the twitter handle link")
text = st.text_area("Enter your recent twitter post ")
st.subheader("Spending to income ratio")

income = st.number_input('Enter your income in Thousands', value=1)
spending = st.number_input('Enter your spending in Thousands', value=1)

if st.button("Predict Default Risk"):
    try:
        SEX, MARRIAGE, EDUCATION = process_inputs(SEX, MARRIAGE, EDUCATION)
        features = [
            SEX,
            EDUCATION,
            MARRIAGE,
            PAY_0,
            PAY_2,
            PAY_3,
            PAY_4,
            PAY_5,
            PAY_6
        ]

        probabilities = model.predict_proba([features])
        default_probability = probabilities[0][1] * 100
    
        preprocessed_text = preprocess_text(text)
        sequence = tokenizer.texts_to_sequences([preprocessed_text])
        max_length = 80
        padded_sequence = pad_sequences(sequence, maxlen=max_length, padding='post', truncating='post')

        sentiment_result = sent_model.predict(padded_sequence)
        sentiment_probability = sentiment_result[0][0]
        sentiment_label = 'positive' if sentiment_probability > 0.5 else 'negative'

        negative_sentiment_probability = 1 - sentiment_probability
        
        sptoincom = (spending/income)*100

        defualt_risk = (default_probability+negative_sentiment_probability+sptoincom)/3

        st.write(f"The Default risk of your credit is {defualt_risk}%")
        
    except Exception as e:
        st.error(f"Error: {str(e)}")
