import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import f1_score, classification_report
from flask import Flask, abort, jsonify, request
from flask import url_for, render_template
from jinja2 import Template

from icd_class import Tokenization, MatrixConverter

model = pickle.load(open("model_pipe.pkl", "rb"))
icd_codes = pickle.load(open("icd_codes.pkl", "rb"))

s = pd.Series('Gross hematuria. Normal renal ultrasound including the bladder.')
results = model.predict(s)
codes = [[icd_codes[i] for i, r in enumerate(result) if r == 1] for result in results][0]
codes

print(codes)
