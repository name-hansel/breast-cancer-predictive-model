import seaborn as sns
from flask import request, send_file, jsonify
import flask
import json
from flask_cors import CORS

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

app = flask.Flask(__name__)
app.config["DEBUG"] = True
CORS(app)


def trainModel():
    df = pd.read_csv("data.csv")
    # Remove the unnamed 32nd column which has no values
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    X = df.drop(['id'], axis=1)
    X = X.drop(['diagnosis'], axis=1)
    y = df['diagnosis']
    X_mean = X.iloc[:, 0:10]

    global X_test, lr

    X_train, X_test, y_train, y_test = train_test_split(
        X_mean, y, random_state=42, test_size=0.1)
    lr = LogisticRegression(solver="lbfgs", max_iter=10000)
    lr.fit(X_train, y_train)
    return round(lr.score(X_test, y_test)*100, 2)


def prediction(data):
    custom = [data]
    customData = pd.DataFrame(custom, columns=list(X_test.columns))
    return(lr.predict(customData))


@app.route('/trainModel', methods=['GET'])
def train():
    return str(trainModel())


@app.route('/predict', methods=['POST'])
def predict():
    data = json.loads(request.data)
    result = prediction(data).tolist()
    return jsonify(result[0])


app.run()
