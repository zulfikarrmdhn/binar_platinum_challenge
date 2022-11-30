from flask import Flask, request, jsonify
import re
import string
import pandas as pd
from flasgger import Swagger, LazyString, LazyJSONEncoder, swag_from
import sqlite3
from datetime import datetime
import matplotlib.pyplot as plt
from cleansing import cleansing

app = Flask(__name__)
app.json_encoder = LazyJSONEncoder

swagger_template = dict(
info = {
    'title': LazyString(lambda: 'API for Sentiment Analysis'),
    'version': LazyString(lambda: '1'),
    'description': LazyString(lambda: 'Platinum Challenge Data Science Binar Academy'),
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

conn = sqlite3.connect("platinum_challenge.db",check_same_thread=False)
#conn.execute("CREATE TABLE cleansing_json (input_kotor varchar(255), output_bersih varchar(50));")

@swag_from("swagger_text.yml", methods=['POST'])
@app.route("/ann_text/v1", methods=['POST'])
def ann_text():
    s = request.get_json()
    return jsonify(s)

@swag_from("swagger_file.yml", methods=['POST'])
@app.route("/ann_file/v1", methods=['POST'])
def ann_file():
    file = request.files["file"]
    df = pd.read_csv(file,encoding='latin-1')
    #print(df.head(20))
    current_datetime = str(datetime.now())
    df.to_sql("ann_file"+current_datetime, conn)
    #return df.to_html()
    return jsonify({"say":"success"})
    
@swag_from("swagger_text.yml", methods=['POST'])
@app.route("/lstm_text/v1", methods=['POST'])
def lstm_text():
    s = request.get_json()
    return jsonify(s)

@swag_from("swagger_file.yml", methods=['POST'])
@app.route("/lstm_file/v1", methods=['POST'])
def lstm_file():
    file = request.files["file"]
    df = pd.read_csv(file,encoding='latin-1')
    #print(df.head(20))
    current_datetime = str(datetime.now())
    df.to_sql("lstm_file"+current_datetime, conn)
    #return df.to_html()
    return jsonify({"say":"success"})
    
if __name__ == "__main__":
    app.run(port=4444, debug=True)