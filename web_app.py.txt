import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import f1_score, classification_report
from flask import Flask, abort, jsonify, request, send_file
from flask import url_for, render_template
from jinja2 import Template
import nltk

from icd_class import Tokenization, MatrixConverter

model = pickle.load(open("model_pipe.pkl", "rb"))
icd_codes = pickle.load(open("icd_codes.pkl", "rb"))
nltk.download('punkt')

app = Flask(__name__)

@app.route("/")
def show_condition():
    condition_list = open("./conditions.txt").read().splitlines()
    return render_template('condition.html', conditions=condition_list)            

@app.route("/test")
def show_test_result():
    X_test = pd.read_csv('X_test.csv', header=None, squeeze=True)
    Y_test = pd.read_csv('Y_test.csv')
    predict_test = model.predict(X_test)
    classification = classification_report(Y_test, predict_test)
    test_f1 = f1_score(Y_test, predict_test, average='micro')
    template = Template('test f1 score {{ name }}')
    return template.render(name=test_f1)

@app.route("/result", methods = ['POST'])
def show_code_result():
    selection = request.form.get("selection")
    free_text = request.form.get("free_text")
    if selection != None:
        series = pd.Series(selection)
    elif free_text != None:
        series = pd.Series(free_text)
    results = model.predict(series)
    codes = [[icd_codes[i] for i, r in enumerate(result) if r == 1] for result in results][0] #assume only one row 
    return render_template('code_prediction.html', condition=series[0], code=codes)

@app.route("/upload", methods=['GET', 'POST'])
def upload_file():
#http://flask.pocoo.org/docs/0.12/patterns/fileuploads/
    if request.method == 'POST':
        try:
            csvfile = request.files['file']
            series = pd.read_csv(csvfile, header=None, squeeze=True)
            results = model.predict(series)
            codes = [[icd_codes[i] for i, r in enumerate(result) if r == 1] for result in results] #assume only one row 
            df = pd.DataFrame({'condition': series, 'icd9_codes': codes})
            df.to_csv('prediction_result.csv', index=False)
            return render_template('downloads.html')
        except Exception as e:
            return str(e) 

@app.route('/return-files/')
def return_files_tut():
#https://pythonprogramming.net/flask-send-file-tutorial/
    try:
        return send_file('./prediction_result.csv', attachment_filename='prediction_result.csv')
    except Exception as e:
        return str(e)

if __name__ == "__main__":
    app.run(host="0.0.0.0")
