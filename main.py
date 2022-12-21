from flask import Flask, request, jsonify
from preprocessing.text_processing import TextProcessing
from inference.predict import PredictSentiment
import pandas as pd

TP = TextProcessing()
predict_model = PredictSentiment()
app = Flask(__name__)

def mapping_result(result_prediction):
    if result_prediction == 0:
        return "neutral"
    elif  result_prediction == 1:
        return "positive"
    else:
        return "negative"

@app.route("/ann_text/v1", methods=['POST'])
def ann_text():
    text = request.get_json()['text']
    df_sent = pd.DataFrame(text, columns=['sent_text'])
    df_sent['sent_text'] = df_sent['sent_text'].apply(TP.cleansing)
    df_sent['sent_text'] = df_sent['sent_text'].apply(TP.get_bow)
    Sentiment = predict_model.ann_predict(df_sent['sent_text'])[0]
    Sentiment = mapping_result(Sentiment)

    return_text = {
        "Sent Text : ": text,
        "Sentiment : ": Sentiment
    }

    return jsonify(return_text)

@app.route("/ann_file/v1", methods=['POST'])
def ann_file():
    file = request.files["file"]
    df = pd.read_csv(file,encoding='latin-1')
    df['Tweet'] = df['Tweet'].apply(TP.cleansing)
    df['Tweet'] = df['Tweet'].apply(TP.get_bow)
    df['Sentiment'] = predict_model.ann_predict(df['Tweet'])
    df['Sentiment'] = df['Sentiment'].apply(mapping_result)

    Tweet = df['Tweet'].to_list()
    Sentiment = df['Sentiment'].to_list()

    return_text = {
        "Tweet     : ": Tweet,
        "Sentiment : ": Sentiment
    }

    return jsonify(return_text)

@app.route("/lstm_text/v1", methods=['POST'])
def lstm_text():
    text = request.get_json()['text']
    df_sent = pd.DataFrame(text, columns=['sent_text'])
    df_sent['sent_text'] = df_sent['sent_text'].apply(TP.cleansing)
    df_sent['sent_text'] = df_sent['sent_text'].apply(TP.get_token)
    Sentiment = predict_model.lstm_predict(df_sent['sent_text'])[0]
    Sentiment = mapping_result(Sentiment)
    
    return_text = {
        "Sent Text : ": text,
        "Sentiment : ": Sentiment
    }

    return jsonify(return_text)

@app.route("/lstm_file/v1", methods=['POST'])
def lstm_file():
    file = request.files["file"]
    df = pd.read_csv(file,encoding='latin-1')
    df['Tweet'] = df['Tweet'].apply(TP.cleansing)
    df['Tweet'] = df['Tweet'].apply(TP.get_token)
    df['Sentiment'] = predict_model.lstm_predict(df['Tweet'])
    df['Sentiment'] = df['Sentiment'].apply(mapping_result)

    Tweet = df['Tweet'].to_list()
    Sentiment = df['Sentiment'].to_list()

    return_text = {
        "Predict Sentiment Analysis for Data Tweet Succes!!!"
        "Tweet     : ": Tweet,
        "Sentiment : ": Sentiment
    }

    return jsonify(return_text)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=1234, debug=True)