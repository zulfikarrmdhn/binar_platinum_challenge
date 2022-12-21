import streamlit as st
import requests

def call_api(text, path):
    url = f"http://127.0.0.1:1234/{path}/v1"

    data_payload = {
        "text":text
    }

    response = requests.post(url,json=data_payload)
    result = response.json()
    return result

st.title('Sentiment Analysis in Bahasa Indonesia (Platinum Challenge Data Science Binar Academy)')

option = st.selectbox(
    'Pilih model yang akan Anda gunakan.',
    ('ANN', 'LSTM'))
text = st.text_input('Masukkan kalimat dalam Bahasa Indonesia')

if text:
    if option == 'ANN':
        path = 'ann_text'
        result = call_api(text,path)
    else:
        path = "lstm_text"
        result = call_api(text,path)
    st.write("your result: ", result)