from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib as plt
from flask_cors import CORS
import json
import flask
from flask import request, send_file, jsonify
import seaborn as sns
plt.use('Agg')

app = flask.Flask(__name__)
app.config["DEBUG"] = True
CORS(app)


def getPlot(feature1, feature2):
    # Get dataset
    df = pd.read_csv("data.csv")
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    X = df.drop(['id'], axis=1)

    # Select only features and output
    X1 = X[[feature1, feature2, 'diagnosis']]
    print(feature1,feature2)

    # Plot
    g = sns.scatterplot(x=feature1, y=feature2,
                        hue='diagnosis', data=X1)
    handles, labels = g.get_legend_handles_labels()
    g.legend(handles[:2], labels[:2])

    # Save plot
    g.figure.savefig('output.png')
    return


def trainModel():
    # Read dataset
    df = pd.read_csv("data.csv")
    # Remove the unnamed 32nd column which has no values
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

    # Remove id and diagnosis columns
    X = df.drop(['id'], axis=1)
    X = X.drop(['diagnosis'], axis=1)
    y = df['diagnosis']

    # Select only mean features
    X_mean = X.iloc[:, 0:10]

    global X_test, lr

    # Split into training and testing dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X_mean, y, random_state=42, test_size=0.1)
    lr = LogisticRegression(solver="lbfgs", max_iter=10000)

    # Fit to model
    lr.fit(X_train, y_train)

    # Return accuracy
    return round(lr.score(X_test, y_test)*100, 2)


def prediction(data):
    # Get the user-input data
    custom = [data]

    # Make the user-input data into a dataframe
    customData = pd.DataFrame(custom, columns=list(X_test.columns))

    # Predict using the custom dataframe
    return(lr.predict(customData))


@app.route('/trainModel', methods=['GET'])
def train():
    return str(trainModel())


@app.route('/predict', methods=['POST'])
def predict():
    # Get the user-input data in JSON form
    data = json.loads(request.data)

    # Call predict function, and convert the predicted class to a list
    result = prediction(data).tolist()

    # Return the predicted class
    return jsonify(result[0])


@app.route('/dataVisualization', methods=['GET'])
def dataVisualization():
    # Get the features for the plot from the url query
    feature1 = request.args.get('feature1')
    feature2 = request.args.get('feature2')

    # Save plot
    getPlot(feature1, feature2)

    # Send the plot
    return send_file('output.png', mimetype='image/gif')


app.run()
