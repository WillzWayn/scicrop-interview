import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from scicrop import *

app = Flask(__name__)
model1_C = pickle.load(open('static/modelos/model_1_FindCrop.sav', 'rb'))
model2 = pickle.load(open('static/modelos/model2_P.sav', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/production',methods=['POST'])
def production():
    '''
    For rendering results on HTML GUI
    '''
    features = [float(x) for x in request.form.values()]
    
    final = np.array(features)

    result = model2.predict([final])

    return render_template('index.html', prediction_text='Sua Produção agricula será de: {}'.format(result[0]))


@app.route('/crop_predict', methods=['POST'])
def crop_predict():
    '''
    For rendering results on HTML GUI
    '''
    features = [float(x) for x in request.form.values()]
    
    final = np.array(features)
    
    prediction = model1_C.predict([final])
    
    return render_template('index.html', prediction_text='Você deve cultivar: {}'.format(cropIntToName(prediction[0])))



if __name__ == "__main__":
    app.run(debug=True)
