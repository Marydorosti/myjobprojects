# -*- coding: utf-8 -*-
"""
Created on Sat Jun 19 12:58:06 2021

@author: m.dorosti
"""

# Dependencies
from flask import Flask, request, jsonify
import joblib
import traceback
import pandas as pd
import numpy as np
import sys
#import argparse
#import parser
import decimal

# Your API definition
#app = Flask(__name__)

from flask import Blueprint
forecastingPredict_bp = Blueprint('forecastingPredict', __name__)



@forecastingPredict_bp.route('/Predict', methods=['POST'])
#@app.route('/Predict', methods=['POST'])
def Predict():
   # if model:
        try:
            #json_ = request.json
            
            #CustomerName = request.args['CustomerName']
            #CustomerName = request.args.get('CustomerName')
            myjson = request.json
            myjson2=pd.DataFrame(request.json)
               #CustomerName=myjson2.CustomerName
               
            json_=myjson['Array']
            CustomerName=myjson['CustomerName']
            forecast_out=10
               
               
            from sklearn.preprocessing import MinMaxScaler
            sc = MinMaxScaler(feature_range = (0, 1))
            #inputs = sc.transform(inputs)
            
            print(json_)
            query = np.array(pd.DataFrame(json_))
            
            query = sc.fit_transform(query)
            #sc = sc.fit(query)
            #query=sc.transform(query)
            
            model = joblib.load('C:/Users/paya8/Desktop/GLOBAL PYTHON SERVICE/ForecastingModels/'+str(CustomerName)+"ForecastModel.pkl") # Load "model.pkl"
            
            prediction = model.predict(query)
            
            my_file = open('C:/Users/paya8/Desktop/GLOBAL PYTHON SERVICE/ForecastingModels/'+str(CustomerName)+"MIN&MAX.txt", "r")
            prediction=np.array(prediction)
            
            scale= my_file.readlines()
            #scale=str(scale)
            pred=prediction*int(float(scale[0]))+(int(float(scale[1]))-int(float(scale[0])))
            
            #pred=prediction*decimal.Decimal(scale[0])+(decimal.Decimal(scale[1])-decimal.Decimal(scale[0]))
            #pred=prediction*scale[0]+(scale[1]-scale[0])
            #pred = sc.inverse_transform(prediction.reshape(-1,1))
            #pred = sc.inverse_transform(prediction)
            #prediction=sc.inverse_transform(prediction)
            prediction=list(pred)
            
            #query = query.reindex(columns=model_columns, fill_value=0)
            #a=np.array(df)
            #x = df.iloc[:,0:6]
            #a=np.array(x)
 
            #prediction = list(XG.predict(query))
            #return str(scale[0])

            return jsonify({'prediction': str(prediction)})

        except:

            return jsonify({'trace': traceback.format_exc()})
   # else:
        #print ('Train the model first')
       # return ('No model here to use')

