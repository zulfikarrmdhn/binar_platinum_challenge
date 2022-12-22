import streamlit as st
import requests

def call_api(text, path):
    url = f"http://127.0.0.1:5555/{path}/v1"

    data_payload = {
        "text":text
    }

    response = requests.post(url,json=data_payload)
    result = response.json()
    print(type(data_payload))
    return result

def upload_csv(uploaded_file, path):
    url = f"http://127.0.0.1:5555/{path}/v1"
    response = requests.post(url,files={'data.csv': uploaded_file})
    result = response.json()
    return result

st.title('Sentiment Analysis in Bahasa Indonesia (Platinum Challenge Data Science Binar Academy)')

option = st.selectbox(
    'Pilih model yang akan akan Anda gunakan.',
    ('ANN', 'LSTM'))

text = st.text_input('Masukkan kalimat dalam bahasa indonesia')

if text:
    if option == 'ANN':
        path = 'ann_text'
        result = call_api(text,path)
    else:
        path = "lstm_text"
        result = call_api(text,path)
    st.write("your result: ", result)

uploaded_file = st.file_uploader("Choose a CSV file")

if uploaded_file:
    if option == 'ANN':
        path = 'ann_file'
        result = upload_csv(uploaded_file,path)
    else:
        path = "lstm_file"
        result = upload_csv(uploaded_file,path)
    st.write("your result: ", result)