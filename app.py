from flask import Flask, request, jsonify
from flasgger import Swagger, LazyString, LazyJSONEncoder, swag_from
from preprocessing.text_processing import TextProcessing
from inference.predict import PredictSentiment
import pandas as pd
import sqlite3

TP = TextProcessing()
predict_model = PredictSentiment()
app = Flask(__name__)
app.json_encoder = LazyJSONEncoder

swagger_template = dict(
info = {
    'title': LazyString(lambda: 'Platinum Challenge Data Science Binar Academy'),
    'version': LazyString(lambda: '1'),
    'description': LazyString(lambda: 'Sentiment Analysis in Bahasa Indonesia with ANN and LSTM model'),
    },
    host = LazyString(lambda: request.host)
)

swagger_config = {
    "headers": [],
    "specs": [
        {
            "endpoint": 'docs',
            "route": '/docs.json',
            "rule_filter": lambda rule: True,
            "model_filter": lambda tag: True,
        }
    ],
    "static_url_path": "/flasgger_static",
    "swagger_ui": True,
    "specs_route": "/docs/"
}

swagger = Swagger(app, template=swagger_template,             
                  config=swagger_config)

def mapping_result(result_prediction):
    if result_prediction == 0:
        return "neutral"
    elif  result_prediction == 1:
        return "positive"
    else:
        return "negative"

@swag_from("docs/text_input.yml", methods=['POST'])
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
    conn.commit()
    conn.close()

    return jsonify({"text":clean_text, "result_sentiment":result_prediction})

@swag_from("docs/text_input.yml", methods=['POST'])
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
    conn.commit()
    conn.close()

    return jsonify({"text":clean_text, "result_sentiment":result_prediction})

@swag_from("docs/upload_file.yml", methods=['POST'])
@app.route("/ann_file/v1", methods=["POST"])
def ann_file():
    file = request.files["file"]
    df = pd.read_csv(file, encoding="latin-1")
    list_result = []

    for i,row in df.iterrows():
        clean_text,bow = TP.get_bow(row['Tweet'])
        result_prediction = predict_model.ann_predict(bow)
        result_prediction = mapping_result(result_prediction)
        list_result.append(result_prediction)

    df['clean_tweet'] = clean_text
    df['predict_result'] = list_result

    ann_df = df[['clean_tweet', 'predict_result']].copy()

    conn =  sqlite3.connect('platinum_challenge.db', check_same_thread=False)
    ann_df.to_sql("ann_file", con=conn, index=False, if_exists='append')
    conn.close()

    return jsonify({"message": "File successfully uploaded"})

@swag_from("docs/upload_file.yml", methods=['POST'])
@app.route("/lstm_file/v1", methods=["POST"])
def lstm_file():
    file = request.files["file"]
    df = pd.read_csv(file, encoding="utf-8")
    list_result = []

    for i,row in df.iterrows():
        clean_text,token = TP.get_token(row['Tweet'])
        result_prediction = predict_model.lstm_predict(token)
        result_prediction = mapping_result(result_prediction)
        list_result.append(result_prediction)

    df['clean_tweet'] = clean_text
    df['predict_result'] = list_result

    lstm_df = df[['clean_tweet', 'predict_result']].copy()

    conn =  sqlite3.connect('platinum_challenge.db', check_same_thread=False)
    lstm_df.to_sql("lstm_file", con=conn, index=False, if_exists='append')
    conn.close()

    return jsonify({"message": "File successfully uploaded"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5555, debug=True)