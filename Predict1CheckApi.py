# -*- coding: utf-8 -*-
"""
Created on Mon Jun 28 08:43:19 2021

@author: m.dorosti
"""
# For Predict IsDishonor
# Dependencies
from flask import Flask, request, jsonify
import joblib
import traceback
import pandas as pd
import numpy as np
from flask import Flask, request, render_template, session, redirect
from sklearn.impute import SimpleImputer
import skl2onnx
import onnx
import sklearn
from sklearn.linear_model import LogisticRegression
import numpy
import onnxruntime as rt
from skl2onnx.common.data_types import FloatTensorType
from skl2onnx import convert_sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

import sys



from flask import Blueprint
Predict1CheckApi_bp = Blueprint('Predict1CheckApi', __name__)



# Your API definition
app = Flask(__name__)
def preprocess_query(json_,CustomerName):
    #my_file = open(str(CustomerName)+"2removedFeatures.txt.txt", "r")
    my_file = open(str('D:/Dorosti/100.20python Service/GLOBAL PYTHON SERVICE/CheckModels/'+CustomerName)+"1bestFeatures.txt", "r")
    #to_drop = my_file.readline()
    to_drop=my_file.read().splitlines()
    #to_drop=to_drop[0:len(to_drop)-1]
    
    #to_drop
    #to_drop=[]
    '''for element in to_drop1:
        #textfile. write(str(element) + "\n")
        to_drop.append(element)'''
    
    #to_drop=list(to_drop)
    s=pd.DataFrame(json_)
    s=s.iloc[:,3:]
    p=type(s)
     
    #query1=s.drop(s[to_drop], axis=1)
    s=s[to_drop]
    #s.columns=to_drop
    #query3=s.drop(columns=['PrsType','IsDishonor'])
    #f=query3.columns
    #query3=s
    m=len(s.columns)
    #query3=query3.iloc[:,0:m]
    r=s.iloc[:,0:m].columns
    f=s.columns
    s=s.iloc[:,0:m].values
    
    #query3=s.iloc[:,0:m].values
    #D=query3.shape
    
    #query = np.array(query3.values)
    #query = np.array(query3.values)
    #X1=np.array(X1)
    #j=query.shape
    return to_drop,s,f,m,p,to_drop,r
    #return to_drop
    
    #print(content_list)
    
    
    
#@app.route('/form',methods = ['GET'])
#def form():
   # return render_template('form.html')




@Predict1CheckApi_bp.route('/Predict1CheckApi', methods = ['POST', 'GET'])

#@app.route('/predict1',  methods = ['POST', 'GET'])
def Predict1CheckApi():
    if request.method == 'GET':
        #return f"The URL /predict is accessed directly. Try going to '/form' to submit form"
        return render_template('form.html')
        #return render_template('form.html')
    if request.method == 'POST':
      #if model:
        try:
            #json_ = request.json
            myjson = request.json
            myjson2=pd.DataFrame(request.json)
               #CustomerName=myjson2.CustomerName
               
            '''json_=myjson['Array']
            CustomerName=myjson['CustomerName']'''
            json_=request.json
            CustomerName="Aramesh"
            
            #CustomerName = request.args['CustomerName']
            #CustomerName = request.args.get('CustomerName')
            #query,to_drop=preprocess_query(json_)
            to_drop,s,f,m,p,to_drop,r=preprocess_query(json_,CustomerName)
            
           
           
            #r=query3.shape
            #r=r[1]
            
            from sklearn.preprocessing import MinMaxScaler
            sc = MinMaxScaler(feature_range = (0, 1))
            
            
            
            
            sc = sc.fit(s)
            
            s=sc.transform(s)
            
            '''imp = SimpleImputer(missing_values=np.nan, strategy='mean')
            query=imp.fit_transform(np.array(s).reshape(-1,1))'''
            
            
            #model = joblib.load(str(CustomerName)+'2model.pkl')
            #s=np.array(s)
            #import onnx

           # model = onnx.load('C:/Users/paya8/Desktop/GLOBAL PYTHON SERVICE/CheckModels/'+str(CustomerName)+'2model.onnx')
            sess = rt.InferenceSession('D:/Dorosti/100.20python Service/GLOBAL PYTHON SERVICE/CheckModels/'+str(CustomerName)+'1model.onnx')
            input_name = sess.get_inputs()[0].name
            label_name = sess.get_outputs()[0].name
            pred_onx = sess.run([label_name],
                    {input_name: s.astype(numpy.float32)})[0]
           # pred_onx = sess.run([label_name],
                    #{input_name: s})[0]
            #j=s.astype(numpy.float32)
            #s=np.array(s)
            #file1=open("1.txt","w")
            #for element in j:
                
              #file1.write(f"{element}\n")
            
            #prediction=model.predict(s.reshape(1,-1))
            
            
            #prediction=model.predict(query2)
            
            
            
            #prediction=list(prediction)
            
            #return jsonify({'prediction': str(j)})
            return jsonify({'prediction': str(pred_onx )})
            #return  '{} {} '.format(pred_onx,p)
        

        except:

            return jsonify({'trace': traceback.format_exc()})
    #else:
        #print ('Train the model first')
        #return ('No model here to use')

'''if __name__ == '__main__':
    try:
        port = int(sys.argv[1]) # This is for a command-line input
    except:
        port = 16944 # If you don't provide any port the port will be set to 12345

    #XG = joblib.load("model.pkl") # Load "model.pkl"
    #model=tf.keras.models.load_model('check_model.sav')
    #model = joblib.load('LRaramesh1.pkl')
    #model = joblib.load(str(CustomrName)+'2model.pkl')
    #model = joblib.load('1model.pkl')
    
    
    print ('Model loaded')
    #model_columns = joblib.load("model_columns.pkl") # Load "model_columns.pkl"
    #print ('Model columns loaded')

    app.run(port=port, debug=True)'''