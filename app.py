from flask_cors import CORS
import flask
from flask import request, render_template
import joblib
import numpy as np

model = joblib.load('MLmodel.ml')

app = flask.Flask(__name__)
app.config["DEBUG"] = True


@app.route('/')
def home():
    return render_template("page1.html")


@app.route('/predict', methods=['POST', 'GET'])
def predict():
    int_features = [int(x) for x in request.form.values()]
    int_features = [np.array(int_features)]
    print(int_features)
    result = model.predict(int_features)
    result = result[0]
    return render_template("page1.html", pred=f"You have the chances of {result} for admission")


app.run(debug=True)
