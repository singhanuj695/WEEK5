import numpy as np
from flask import Flask, request, render_template
import pickle

# Create flask app
Flask_app = Flask(__name__)
model = pickle.load(open("iris_model.pkl", "rb"))

@Flask_app.route("/")
def Home():
    return render_template("home.html")

@Flask_app.route("/predict", methods = ["POST"])
def predict():
    float_features = [float(x) for x in request.form.values()]
    features = [np.array(float_features)]
    prediction = model.predict(features)
    return render_template("home.html", prediction_text = "The flower species is {}".format(prediction))

if __name__ == "__main__":
    Flask_app.run(debug=True)
