# -*- coding: utf-8 -*-
"""
Created on Mon Jul  5 11:51:12 2021

@author: m.dorosti
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Jul  4 16:45:14 2021

@author: m.dorosti
"""

from flask import Flask, request, jsonify
import joblib
import traceback
import pandas as pd
import numpy as np
import pyodbc

import json

from sklearn.impute import SimpleImputer


from flask import Blueprint
letter_recommendation_train_bp = Blueprint('letter_recommendation_train', __name__)


#app = Flask(__name__,template_folder='template')
def unique(list1):
         x = np.array(list1)
         print(np.unique(x))
         a=list(np.unique(x))
         return a

def preprocess_data(json_,CustomerName):
      df = pd.DataFrame(json_)
      m=pd.DataFrame(df['ReceiverUser'])
      m.to_csv('D:/Dorosti/100.20python Service/GLOBAL PYTHON SERVICE/LetterRecommendModels/'+str( CustomerName)+'RU.csv')
      from sklearn import preprocessing
      from sklearn.linear_model import LogisticRegression
      from sklearn.model_selection import train_test_split
      from sklearn.preprocessing import StandardScaler
      # label_encoder object knows how to understand word labels. 
      label_encoder = preprocessing.LabelEncoder()
      # Encode labels in column 'Country'. 
      df['Sender']= label_encoder.fit_transform(df['Sender'].astype(str))
      df['Class']= label_encoder.fit_transform(df['Class'])
      df['NodeType']= label_encoder.fit_transform(df['NodeType'].astype(str))
      df['OwnerWorkgroup']= label_encoder.fit_transform(df['OwnerWorkgroup'].astype(str))
      df['OwnerUser']= label_encoder.fit_transform(df['OwnerUser'].astype(str))
      df['ReceiverUser']= label_encoder.fit_transform(df['ReceiverUser'].astype(str))
      df['SenderWorkgroup']= label_encoder.fit_transform(df['SenderWorkgroup'].astype(str))
      df['SenderPost']= label_encoder.fit_transform(df['SenderPost'].astype(str))
      df=df.drop(columns=['WaitTime2','IsCopy'])
      #df=df.drop(columns=['IsPrivate','IsTask'])
      df.drop(["IsPrivate","IsTask"],axis=1,inplace=True)
      df=(df-df.min())/(df.max()-df.min())
      df.fillna(df.mean(), inplace=True)
     
      imp = SimpleImputer(missing_values=np.nan, strategy='mean')
      y = df.ReceiverUser.copy().values
      
      X=df.drop("ReceiverUser",axis=1).copy().values
      X=imp.fit_transform(X)
      
      
      lab_enc = preprocessing.LabelEncoder()
      encoded1 = lab_enc.fit_transform(y)
      encoded=imp.fit_transform(encoded1.reshape(-1,1))
     

      xtrain,xtest,ytrain,ytest = train_test_split(X,encoded,random_state=500)
      
      a=unique(encoded1)
      #f=[m.ReceiverUser.unique]
      m=pd.read_csv('D:/Dorosti/100.20python Service/GLOBAL PYTHON SERVICE/LetterRecommendModels/'+str( CustomerName)+'RU.csv')
      f=unique(m.ReceiverUser)
      df3 = pd.DataFrame(zip(f, a),
               columns =['Name', 'code'])
      #df3.to_csv(str(CustomerName)+'.csv')
      #D:/Dorosti/100.20python Service/GLOBAL PYTHON SERVICE/similarLettersModels
      #df3.to_csv('C:/Users/paya8/Desktop/GLOBAL PYTHON SERVICE/LetterRecommendModels/'+str( CustomerName)+'.csv')
      df3.to_csv('D:/Dorosti/100.20python Service/GLOBAL PYTHON SERVICE/LetterRecommendModels/'+str( CustomerName)+'.csv')
     
      return xtrain,xtest,ytrain,ytest ,f,a
@letter_recommendation_train_bp.route('/letter_recommendation_train',methods=['POST'])
#@app.route('/train', methods=['POST'])
def letter_recommendation_train():
    #if model:
        try :
               myjson = request.json
               #myjson2=pd.DataFrame(request.json)
               #CustomerName=myjson2.CustomerName
               
               #json_=myjson['Array']
               ConnectionString=myjson['ConnectionString']
               #json_=request.json
               CustomerName=myjson['CustomerName']
               #CustomerName="PAYA2"
               s=ConnectionString.split(';')
               a=s[0].split('=')[1]
               b=s[1].split('=')[1]
               c=s[2].split('=')[1]
               d=s[3].split('=')[1]
               m='Sql Server'
               conn=pyodbc.connect(Driver=m,
                                   Server=a,
                                   Database=b,
                                   UID=c,
                                   PWD=d
                                   )
               
               
               '''conn=pyodbc.connect('Driver={Sql Server};'
                                   'Server=192.168.100.17\\SQL2019;'
                                   'Database=PayaAfzarPendarData;'
                                  'UID=sa;'
                                   'PWD=PAYA+master;'
                                   )'''
               
               
               with open('D://Dorosti//100.20python Service//GLOBAL PYTHON SERVICE//LetterRecommendModels//letterRecommendationQuery.txt' ,'r')as file:
                   query=file.read()
                   
                   
               cursor = conn.cursor()   

           
               GF=pd.read_sql_query(query,conn) 
                  
               
               json_=GF
               
               import joblib
               from xgboost import XGBClassifier
               xtrain,xtest,ytrain,ytest,f ,a=preprocess_data(json_,CustomerName)
               xgb_clf = XGBClassifier(objective="multi:softmax",n_estimators=100)
               xgb_clf.fit(xtrain,ytrain,verbose=True)
               
               
               from sklearn.metrics import accuracy_score
               xgb_test_accuracy=xgb_clf.score(xtest,ytest)
               xgb_train_accuracy=xgb_clf.score(xtrain,ytrain)
               
               #joblib.dump(xgb_clf,'C:/Users/paya8/Desktop/GLOBAL PYTHON SERVICE/LetterRecommendModels/'+str( CustomerName)+'xgb_clf.pkl')
               joblib.dump(xgb_clf,'D:/Dorosti/100.20python Service/GLOBAL PYTHON SERVICE/LetterRecommendModels/'+str( CustomerName)+'xgb_clf.pkl')
               #print("Model dumped!") 
               #xgb_clf = joblib.load('xgb_clf.pkl') 
               #m=xtrain.shape
               #return  (str(f))
               #return  (str("Model dumped!"))
               #return jsonify({'RESULT': str("Model dumped!")})
           
               return jsonify({'RESULT': str("Model dumped")})
           
               #return jsonify({'RESULT': str(json_)})
               #return (CustomerName         
               
        except:

               return jsonify({'trace': traceback.format_exc()})
    #else:
       # print ('Train the model first')
       # return ('No model here to use')



               
               
               

