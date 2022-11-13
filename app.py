from flask import Flask, request, jsonify
import re
import string
import pandas as pd
from flasgger import Swagger, LazyString, LazyJSONEncoder, swag_from
import sqlite3
from datetime import datetime
import matplotlib.pyplot as plt

app = Flask(__name__)
app.json_encoder = LazyJSONEncoder

swagger_template = dict(
info = {
    'title': LazyString(lambda: 'Membuat Cleansing API untuk CSV dan JSON'),
    'version': LazyString(lambda: '1'),
    'description': LazyString(lambda: 'Gold Challenge'),
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

conn = sqlite3.connect("challenge_gold.db",check_same_thread=False)
#conn.execute("CREATE TABLE cleansing_json (input_kotor varchar(255),  output_bersih varchar(50));")


swagger = Swagger(app, template=swagger_template,             
                  config=swagger_config)



@swag_from("swagger_text.yml", methods=['POST'])
@app.route("/clean_text/v1", methods=['POST'])
def remove_punct_post():
    s = request.get_json()
    return jsonify(s)

@swag_from("swagger_file.yml", methods=['POST'])
@app.route("/clean_csv/v1", methods=['POST'])
def remove_punct_csv():
    file = request.files.get('file')
    df = pd.read_csv(file,encoding='latin')
    #print(df.head(20))
    current_datetime = str(datetime.now())
    df.to_sql("uploadtable"+current_datetime, conn)
    #return df.to_html()
    return jsonify({"say":"success"})
    

if __name__ == "__main__":
    app.run(port=4444, debug=True)