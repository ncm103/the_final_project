import flask
import tensorflow
import pandas as pd
from tensorflow import keras
from load import *

global model 

model = init()

# Use pickle to load in the pre-trained model.
# with open(f'tf_model3/saved_model.pb', 'rb') as f:
#     model = keras.models.load_model(f)

# with open(f'model/model.json', 'r') as f:
#     model = json.load(f)

app = flask.Flask(__name__, template_folder='templates')
# def index_view():
#     return render_template('main.html')

@app.route('/', methods=['GET', 'POST'])
def main():
    if flask.request.method == 'GET':
        return(flask.render_template('main.html'))
    if flask.request.method == 'POST':
        R_Weight = flask.request.form['R_Weight']
        R_Height = flask.request.form['R_Height']
        R_Age = flask.request.form['R_Age']
        B_Weight = flask.request.form['B_Weight']
        B_Height = flask.request.form['B_Height']
        B_Age = flask.request.form['B_Age']
        RPrev = 0
        BPrev = 0
        BStreak = 0
        RStreak = 0

        input_variables = pd.DataFrame([[BPrev, BStreak, B_Age,B_Height,B_Weight,RPrev, RStreak, R_Age,R_Height,R_Weight]],
                                       columns=['BPrev','BStreak','B_Age','B_Height','B_Weight','RPrev','RStreak','R_Age','R_Height','R_Weight'],
                                       dtype=float)
                                    #    index=['input'])
        prediction = model.predict(input_variables)[0]
        return flask.render_template('main.html',
                                     original_input={'R_Weight' :R_Weight,
                                                    'R_Height' :R_Height,
                                                    'R_Age' :R_Age,
                                                    'B_Weight' :B_Weight,
                                                    'B_Height' :B_Height,
                                                    'B_Age' :B_Age
                                                     },
                                     result=prediction,
                                     )
        print(prediction)

if __name__ == '__main__':
    app.run(debug=True)