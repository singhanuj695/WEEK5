import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
import pickle

# Create flask app
app_api = Flask(__name__)
model = pickle.load(open("iris_model.pkl", "rb"))

@app_api.route("/", methods = ['GET', 'POST'])
def Home():
    if request.method == 'GET':
        data = 'Testing with API'
        return jsonify({'data':data})

@app_api.route("/predict/", methods = ['GET', 'POST'])
def predict():
    sepal_length = request.args.get("sepal_length")
    sepal_width = request.args.get("sepal_width")
    petal_length = request.args.get("petal_length")
    petal_width = request.args.get("petal_width")
    test_df = pd.DataFrame({'sepal.length': [sepal_length], 'sepal.width':[sepal_width], 'petal.length':[petal_length], 'petal.width': [petal_width]})
    prediction = model.predict(test_df)
    return jsonify({'Iris Flower':str(prediction)})

if __name__ == "__main__":
    app_api.run(debug=True)