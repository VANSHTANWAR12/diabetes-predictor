from flask import Flask,request,jsonify,render_template

import pickle
import numpy
import pandas as pd
from sklearn.preprocessing import StandardScaler

application=Flask(__name__)
app=application

##impport ridge regressor and standardscaler pickle

model=pickle.load(open('models/regressor.pkl','rb'))
standard_scaler=pickle.load(open('models/scaler.pkl','rb'))


@app.route('/')
def index():
    return render_template('about.html')

@app.route('/predict',methods=['GET','POST'])
def predict():
    if request.method=='POST':
        Pregnancies=float(request.form.get('Pregnancies'))
        Glucose=float(request.form.get('Glucose'))
        BloodPressure=float(request.form.get('BloodPressure'))
        SkinThickness=float(request.form.get('SkinThickness'))
        Insulin=float(request.form.get('Insulin'))
        BMI=float(request.form.get('BMI'))
        DiabetesPedigreeFunction=float(request.form.get('DiabetesPedigreeFunction'))
        Age = float(request.form.get('Age'))



        new_data = [[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]]
        new_data_scaled = standard_scaler.transform(new_data)
        result=model.predict(new_data_scaled)
        return render_template('index.html',results=[result[0]])

    else:
        return render_template('index.html')
    
if __name__=="__main__":
    app.run(host="0.0.0.0",debug=True)