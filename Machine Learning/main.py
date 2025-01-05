from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
from nltk.corpus import stopwords
import nltk
from nltk.stem import PorterStemmer
import re
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
nltk.download('stopwords')



with open('F:/CreditScore/Machine Learning/Default risk Analysis/log_reg.pkl', 'rb') as file:
    model = pickle.load(file)

app = FastAPI()


class PredictionRequest(BaseModel):
    SEX: str
    EDUCATION: str
    MARRIAGE: str
    PAY_0: float
    PAY_2: float
    PAY_3: float
    PAY_4: float
    PAY_5: float
    PAY_6: float

def process_inputs(SEX: str, MARRIAGE: str, EDUCATION:str):
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
    
    if EDUCATION =='graduate':
        EDUCATION = 1
    elif EDUCATION == 'school':
        EDUCATION = 2
    elif EDUCATION == 'university':
        EDUCATION = 3
    else:
        EDUCATION = 4
    

    return SEX, MARRIAGE



def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    text = text.lower()
    tokens = text.split()

    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]

    stemmer = PorterStemmer()
    tokens = [stemmer.stem(token) for token in tokens]

    return ' '.join(tokens)

@app.post("/predict")
def predict(request: PredictionRequest):
    try:
        SEX, MARRIAGE, EDUCATION = process_inputs(request.SEX, request.MARRIAGE, request.EDUCATION)
        
        features = [
            SEX,
            EDUCATION,
            MARRIAGE,
            request.PAY_0,
            request.PAY_2,
            request.PAY_3,
            request.PAY_4,
            request.PAY_5,
            request.PAY_6
        ]
        
        probabilities = model.predict_proba([features])
        default_probability = probabilities[0][1] 
        
        return {"default_probability": default_probability * 100}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error: {str(e)}")

class TextRequest(BaseModel):
    text: str



model_path = 'F:/CreditScore/Machine Learning/Sentiment Analysis/my_model.keras'
sent_model = load_model(model_path)

with open('F:/CreditScore/Machine Learning/Sentiment Analysis/tokenizer.pkl', 'rb') as file:
    tokenizer = pickle.load(file)


@app.post('/sentiment')
def sentiment(request: TextRequest):
    try:
        preprocessed_text = preprocess_text(request.text)
        sequence = tokenizer.texts_to_sequences([preprocessed_text])
        max_length = 80 
        padded_sequence = pad_sequences(sequence, maxlen=max_length, padding='post', truncating='post')

        sentiment_result = sent_model.predict(padded_sequence)
        sentiment_probability = sentiment_result[0][0]
        sentiment_label = 'positive' if sentiment_probability > 0.5 else 'negative'

        return {"probability": sentiment_probability, "sentiment": sentiment_label}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error: {str(e)}")

        
    