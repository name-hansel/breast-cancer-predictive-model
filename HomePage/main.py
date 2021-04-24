from flask import Flask, render_template
app = Flask(__name__)


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

@app.route('/visual/')
def visual():
    return render_template('visual.html')

if __name__ == '__main__':
    app.run()


