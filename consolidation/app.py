import flask
import tensorflow
import sklearn
import pandas as pd
import numpy as np
from tensorflow import keras
from pickle import load
from load import *

model = init()

scaler = load(open('model\scaler.pkl', 'rb'))
print("Awesome, your scaler has been loaded from disk! Cool beans!")

app = flask.Flask(__name__, template_folder='templates')

@app.route('/', methods=['GET', 'POST'])
def main():
    global model 
    if flask.request.method == 'GET':
        return(flask.render_template('main.html'))
    if flask.request.method == 'POST':
        R_Weight = flask.request.form['R_Weight']
        R_Height = flask.request.form['R_Height']
        R_Age = flask.request.form['R_Age']
        B_Weight = flask.request.form['B_Weight']
        B_Height = flask.request.form['B_Height']
        B_Age = flask.request.form['B_Age']
        RPrev = 2.130049
        BPrev = 1.756650
        BStreak = 0.643350
        RStreak = 0.748768
        input_variables = pd.DataFrame([[BPrev, BStreak, B_Age,B_Height,B_Weight,RPrev, RStreak, R_Age,R_Height,R_Weight]],
                                    columns=['BPrev','BStreak','B_Age','B_Height','B_Weight','RPrev','RStreak','R_Age','R_Height','R_Weight'],
                                    dtype=float)
        input_scaled = scaler.transform(input_variables)
        prediction = model.predict(input_scaled)[0][0]
        if np.round(prediction) == 0:
            prediction = "Blue"
        else:
            prediction = "Red"   
        # print(type(prediction))
        # prediction = np.round(prediction)
        return flask.render_template('main.html',
                                      original_input={'R_Weight' :R_Weight,
                                                    'R_Height' :R_Height,
                                                    'R_Age' :R_Age,
                                                    'B_Weight' :B_Weight,
                                                    'B_Height' :B_Height,
                                                    'B_Age' :B_Age
                                                     },
                                     result=prediction
                                     )

@app.route('/logisticregression')
def logreg():
    return(flask.render_template('LogisticRegression.html'))

if __name__ == '__main__':
    app.run(debug=True)