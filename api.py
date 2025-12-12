# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import streamlit as st
import pickle
import tensorflow as tf
from tensorflow import keras
import nltk
import re
from tensorflow.keras.preprocessing.sequence import pad_sequences   # ✅ ADD THIS LINE
import joblib

model = tf.keras.models.load_model('sentiment_model.h5')
vectorize = joblib.load('tokenizer.pkl')

nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

def clean_txt(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

max_len = 50

st.title('Twitter Sentiment Analysis Using LSTM')

user_input = st.text_area('Enter A Tweet')

if st.button('predict'):
    if user_input == '':
        st.warning('Please Enter A Message')
    else:
        cleaned = clean_txt(user_input)
        seq = vectorize.texts_to_sequences([cleaned])
        padded = pad_sequences(seq, maxlen=max_len, padding='post', truncating='post')
        prediction = model.predict(padded)[0][0]

        if prediction > 0.5:
            st.success('Positive Tweet')
        else:
            st.error('Negative Tweet')


# %%
