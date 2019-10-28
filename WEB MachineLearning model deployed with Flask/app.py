import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from scicrop import *

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/production',methods=['POST'])
def production():
    '''
    For rendering results on HTML GUI
    '''
    try:
        # Features contem Area, Crop, GDP
        features = [x for x in request.form.values()]
        
        #np array com as informações de Area e GDP
        predictValues = np.array([float(features[0])])
        
        # Puxando as informações do dict do crop para usar aquela regLinear !
        result = model2[int(features[1])].predict([predictValues])
        
        return render_template('index.html', prediction_text='Sua Produção agricula será de: {:.2f}'.format(abs(result[0][0])))
        
    except KeyError:
        return render_template('index.html', prediction_text='Desculpe mas esse cultivo que deseja prever a produção ainda não foi mapeado. Caso deseje desenvolver um trabalho nessa área, entre em contato conosco')


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
