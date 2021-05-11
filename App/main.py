from flask import Flask, render_template

# Visualization
import matplotlib as plt
from flask import request
import pandas as pd
import seaborn as sns
from io import BytesIO
import base64
plt.use('Agg')

app = Flask(__name__)

# Get dataset for global use
df = pd.read_csv("data.csv")


@app.route('/')
def hello_world():
    return render_template('home.html')


@app.route('/login/')
def login():
    return render_template('login.html')


@app.route('/signup/')
def signup():
    return render_template('signup.html')


@app.route('/model1/')
def model1():
    return render_template('model1.html')


@app.route('/model2/')
def model2():
    return render_template('model2.html')


@app.route('/model3/')
def model3():
    return render_template('model3.html')


@app.route('/visualization/getImage')
def visualizationGetImage():
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

    return render_template('visual.html', image=True, img=image)


@app.route('/visualization')
def visualization():
    return render_template('visual.html', image=False)


if __name__ == '__main__':
    app.run(debug=True)
