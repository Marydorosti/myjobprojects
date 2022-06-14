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
#import sys
import json
from flask import Flask, request, render_template, session, redirect
from sklearn.impute import SimpleImputer

from flask import Blueprint
letter_recommendation_predict_bp = Blueprint('letter_recommendation_predict', __name__)

#letter_recommendation_predict

#app = Flask(__name__,template_folder='template')
#df5=pd.read_csv('5.csv')

def preprocess_data(json_):
      df = pd.DataFrame(json_)
      from sklearn import preprocessing
      # label_encoder object knows how to understand word labels. 
      label_encoder = preprocessing.LabelEncoder()
      # Encode labels in column 'Country'. 
      df['Sender']= label_encoder.fit_transform(df['Sender'].astype(str))
      df['Class']= label_encoder.fit_transform(df['Class'])
      df['NodeType']= label_encoder.fit_transform(df['NodeType'].astype(str))
      df['OwnerWorkgroup']= label_encoder.fit_transform(df['OwnerWorkgroup'].astype(str))
      df['OwnerUser']= label_encoder.fit_transform(df['OwnerUser'].astype(str))
      #df['ReceiverUser']= label_encoder.fit_transform(df['ReceiverUser'].astype(str))
      df['SenderWorkgroup']= label_encoder.fit_transform(df['SenderWorkgroup'].astype(str))
      df['SenderPost']= label_encoder.fit_transform(df['SenderPost'].astype(str))
      #HasBody
      df['HasBody']= label_encoder.fit_transform(df['HasBody'].astype(str))
      df['HasAttachment']= label_encoder.fit_transform(df['HasAttachment'].astype(str))
      df['IsCopy']= label_encoder.fit_transform(df['IsCopy'].astype(str))
      df['NodeType']= label_encoder.fit_transform(df['NodeType'].astype(str))
      df['IsOwner']= label_encoder.fit_transform(df['IsOwner'].astype(str))
      df['HasCustomDisplayName']= label_encoder.fit_transform(df['HasCustomDisplayName'].astype(str))
      df['IsSent']= label_encoder.fit_transform(df['IsSent'].astype(str))
      df['IsCopy']= label_encoder.fit_transform(df['IsCopy'].astype(str))
      
      
      
      from sklearn.preprocessing import MinMaxScaler
      sc = MinMaxScaler(feature_range = (0, 1))
      df=df.drop(columns=['WaitTime2','IsCopy'])
      df.drop(["IsPrivate","IsTask"],axis=1,inplace=True)
      df=(df-df.min())/(df.max()-df.min())
      df.fillna(df.mean(), inplace=True)
      imp = SimpleImputer(missing_values=np.nan, strategy='mean')
            
      #query = np.array(pd.DataFrame(json_))
      sc = sc.fit(df)
      query=sc.transform(df)
      #query=imp.fit_transform(query.reshape(-1,1))
      return query,df
  

@letter_recommendation_predict_bp.route('/letter_recommendation_predict',methods=['POST'])
#@app.route('/predict', methods=['POST'])
def letter_recommendation_predict():
    #if model:
        try :
               #json_ = request.json
               #CustomerName = request.args['CustomerName']
               #ustomerName = request.args.get('CustomerName')
               
               import pandas as pd
               myjson = request.json
               #myjson2=pd.DataFrame(request.json)
               #CustomerName=myjson2.CustomerName
               #CustomerName="Test3"
               
               json_=myjson['Array']
               #json_=request.json
               #json_=myjson
               CustomerName=myjson['CustomerName']
               #CustomerName="Aramesh"             
               #a_json = json.loads(json_)
               #import joblib
               #import traceback
               import pandas as pd
               #import datetime
               import numpy as np            
               query,df=preprocess_data(json_)        
               #df5=pd.read_csv('C:/Users/paya8/Desktop/GLOBAL PYTHON SERVICE/LetterRecommendModels/'+str( CustomerName)+'.csv')
               df5=pd.read_csv('D:/Dorosti/100.20python Service/GLOBAL PYTHON SERVICE/LetterRecommendModels/'+str( CustomerName)+'.csv')
               
               #df5=df5.astype({"Name":int})
               #model = joblib.load('C:/Users/paya8/Desktop/GLOBAL PYTHON SERVICE/LetterRecommendModels/'+str(CustomerName)+'xgb_clf.pkl')
               model = joblib.load('D:/Dorosti/100.20python Service/GLOBAL PYTHON SERVICE/LetterRecommendModels/'+str(CustomerName)+'xgb_clf.pkl')
               
            
               prediction=model.predict_proba(query.reshape(1,-1))
               
               #prediction=model.predict_proba(query)
               
               topk = np.argsort(prediction,axis=1)[:,-4:]
               #topk2=list(topk)
               #z=[topk[0][2],topk[0][1],topk[0][0]]
               m=[df5.Name[topk[0][3]],df5.Name[topk[0][2]],df5.Name[topk[0][1]],df5.Name[topk[0][0]]]
               #prediction=list(pred)
               #m=json_
               #m=[0,371]
               #m=[df['NodeType'][0]]
               #m=[m[0]]
               
            
               #query = query.reindex(columns=model_columns, fill_value=0)
               #a=np.array(df)
               #x = df.iloc[:,0:6]
               #a=np.array(x)
 
               #prediction = list(XG.predict(query))
               #return str(scale[0])

               return jsonify({'recommendation': str(m)})
               
               #return  (str(m))        
               
        except:

               return jsonify({'trace': traceback.format_exc()})
    #else:
        #print ('Train the model first')
        #return ('No model here to use')


               
               
               

