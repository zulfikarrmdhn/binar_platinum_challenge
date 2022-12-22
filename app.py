from flask import Flask, request, jsonify
from preprocessing.text_processing import TextProcessing
from inference.predict import PredictSentiment
import pandas as pd
import sqlite3

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

@app.route("/ann_text/v1", methods=["POST"])
def ann_text():
    text = request.get_json()['text']
    clean_text,bow = TP.get_bow(text)
    result_prediction = predict_model.ann_predict(bow)
    result_prediction = mapping_result(result_prediction)

    conn =  sqlite3.connect('platinum_challenge.db', check_same_thread=False)
    conn.execute("CREATE TABLE IF NOT EXISTS ann_text (sent_text, predict_result)")
    query_text = "INSERT INTO ann_text (sent_text, predict_result) values(?, ? )"
    val = (clean_text, result_prediction)
    conn.execute(query_text, val)
    clean_text_file = pd.read_sql_query("SELECT * FROM ann_text", conn)
    clean_text_file.to_csv("ann_text.csv")
    conn.commit()
    conn.close()

    return jsonify({"text":clean_text, "result_sentiment":result_prediction})

@app.route("/lstm_text/v1", methods=["POST"])
def lstm_text():
    text = request.get_json()['text']
    clean_text,token = TP.get_token(text)
    result_prediction = predict_model.lstm_predict(token)
    result_prediction = mapping_result(result_prediction)

    conn =  sqlite3.connect('platinum_challenge.db', check_same_thread=False)
    conn.execute("CREATE TABLE IF NOT EXISTS lstm_text (sent_text, predict_result)")
    query_text = "INSERT INTO lstm_text (sent_text, predict_result) values(?, ? )"
    val = (clean_text, result_prediction)
    conn.execute(query_text, val)
    clean_text_file = pd.read_sql_query("SELECT * FROM lstm_text", conn)
    clean_text_file.to_csv("lstm_text.csv")
    conn.commit()
    conn.close()

    return jsonify({"text":clean_text, "result_sentiment":result_prediction})

@app.route("/ann_file/v1", methods=["POST"])
def ann_file():
    file = request.files["file"]
    df = (pd.read_csv(file, encoding="latin-1"))
    list_result = []

    for i,row in df.iterrows():
        clean_text,bow = TP.get_bow(row['Tweet'])
        result_prediction = predict_model.ann_predict(bow)
        result_prediction = mapping_result(result_prediction)
        list_result.append(result_prediction)

    df['predict_result'] = list_result

    ann_df = df[['Tweet', 'predict_result']].copy()

    conn =  sqlite3.connect('platinum_challenge.db', check_same_thread=False)
    ann_df.to_sql("ann_file", con=conn, index=False, if_exists='append')
    ann_file = pd.read_sql_query("SELECT * FROM ann_file", conn)
    ann_file.to_csv("ann_file.csv")
    conn.close()

    Tweet = ann_df.Tweet.to_list()
    Sentiment = ann_df.predict_result.to_list()

    return_text = {
        "Tweet     : ": Tweet,
        "Sentiment : ": Sentiment
    }

    return jsonify(return_text)

@app.route("/lstm_file/v1", methods=["POST"])
def lstm_file():
    file = request.files["file"]
    df = (pd.read_csv(file, encoding="latin-1"))
    list_result = []

    for i,row in df.iterrows():
        clean_text,token = TP.get_token(row['Tweet'])
        result_prediction = predict_model.lstm_predict(token)
        result_prediction = mapping_result(result_prediction)
        list_result.append(result_prediction)

    df['predict_result'] = list_result

    lstm_df = df[['Tweet', 'predict_result']].copy()

    conn =  sqlite3.connect('platinum_challenge.db', check_same_thread=False)
    lstm_df.to_sql("lstm_file", con=conn, index=False, if_exists='append')
    lstm_file = pd.read_sql_query("SELECT * FROM lstm_file", conn)
    lstm_file.to_csv("lstm_file.csv")
    conn.close()

    Tweet = lstm_df.Tweet.to_list()
    Sentiment = lstm_df.predict_result.to_list()

    return_text = {
        "Tweet     : ": Tweet,
        "Sentiment : ": Sentiment
    }

    return jsonify(return_text)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5555, debug=True)