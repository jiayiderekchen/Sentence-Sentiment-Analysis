from flask import Flask, render_template,request,jsonify
import requests
import pickle
import numpy as np
import sklearn
from main import predict_pipeline


app = Flask(__name__)
@app.route('/',methods=['GET'])
def Home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = str(request.form['sentence'])
    try:
        sample = data['text']
    except KeyError:
        return jsonify({'error': 'No text sent'})

    sample = [sample]
    predictions = predict_pipeline(sample)
    try:
        result = jsonify(predictions[0])
    except TypeError as e:
        result = jsonify({'error': str(e)})
    return render_template('index.html',prediction_text=result)

if __name__ == '__main__':
    app.run(debug=True)