from django.shortcuts import render
import pickle
from flask import Flask, request, app, jsonify, url_for,render_template
import numpy as np
import pandas as pd

app=Flask(__name__)
##Load the model
regmodel=pickle.load(open('regmodel.pkl','rb'))
scaler=pickle.load(open('scaling.pkl','rb'))
@app.route('/')


def home():
    return render_template('home.html')

@app.route('/predict_api',methods=['POST'])

def predict_api():
    dataset=request.json['dataset']
    print(dataset)
    print(np.array(list(dataset.values())).reshape(1,-1))
    newdata=scaler.transform(print(np.array(list(dataset.values())).reshape(1,-1)))
    output=regmodel.predict(newdata)
    print(output[0])
    return jsonify(output[0])

if __name__ == "__main_":
            app.run(debug=True)