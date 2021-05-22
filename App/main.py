from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from flask import Flask, render_template, request, jsonify
import json
from flask_cors import CORS

# Libraries for ML
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Visualization
import matplotlib as plt
from flask import request
import pandas as pd
import seaborn as sns
from io import BytesIO
import base64
plt.use('Agg')

app = Flask(__name__)
CORS(app)

# API routes -> for training model and prediction
# Get dataset for global use
df = pd.read_csv("data.csv")


def train(model_name):
    global df
    # Remove the unnamed 32nd column which has no values
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    X = df.drop(['id'], axis=1)
    X = X.drop(['diagnosis'], axis=1)
    y = df['diagnosis']
    X_mean = X.iloc[:, 0:10]

    global X_test, model

    X_train, X_test, y_train, y_test = train_test_split(
        X_mean, y, random_state=42, test_size=0.1)

    if(model_name == 'logistic-regression'):
        model = LogisticRegression(solver="lbfgs", max_iter=10000)
    elif(model_name == 'decision-tree'):
        model = DecisionTreeClassifier(criterion='entropy', random_state=0)
    elif(model_name == 'random-forest'):
        model = RandomForestClassifier(
            n_estimators=10, criterion='entropy', random_state=0)

    model.fit(X_train, y_train)
    return round(model.score(X_test, y_test)*100, 2)


@app.route('/trainModel/<model_name>', methods=['GET'])
def trainModel(model_name):
    return str(train(model_name))

# Prediction


def predict(data):
    custom = [data]
    customData = pd.DataFrame(custom, columns=list(X_test.columns))
    return(model.predict(customData))


@app.route('/predict/<model_name>', methods=['POST'])
def predictModel(model_name):
    data = json.loads(request.data)
    result = predict(data).tolist()
    return jsonify(result[0])

# App routes


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/login')
def login():
    return render_template('login.html')


@app.route('/signup')
def signup():
    return render_template('signup.html')


@app.route('/logistic-regression')
def showLogisticRegression():
    return render_template('logistic-regression.html')


@app.route('/decision-tree')
def showDecisionTree():
    return render_template('decision-tree.html')


@app.route('/random-forest')
def showRandomForest():
    return render_template('random-forest.html')


@app.route('/visualization')
def showVisualization():
    return render_template('visual.html')


@app.route('/getPlot')
def getPlot():
    # Get features
    feature1 = request.args.get('feature-1')
    feature2 = request.args.get('feature-2')

    global df
    g = sns.scatterplot(x=feature1, y=feature2, hue='diagnosis', data=df)

    img = BytesIO()
    g.figure.savefig(img, format='png')
    img.seek(0)

    image = base64.b64encode(img.getvalue())
    image = image.decode('utf8')

    g.get_figure().clf()
    return jsonify(image)


if __name__ == '__main__':
    app.run(debug=True)
