# -*- coding: utf-8 -*-
"""
Created on Mon Jul 11 11:59:19 2022

@author: m.dorosti
"""

import pandas as pd
import numpy as np
import pickle
import fasttext
import heapq
import numpy
import traceback
from flask import Flask, request, jsonify
from parsivar import Tokenizer
from parsivar import Normalizer

from flask import Blueprint

from flask import Blueprint
SearchEnginePredict_bp = Blueprint('SearchEnginePredict', __name__)

@SearchEnginePredict_bp.route('/SearchEnginePredict',methods=['POST']) 

def SearchEnginePredict():
  try:  
    json_=request.json['word']
    model1=fasttext.load_model('D:/Dorosti/100.20python Service/GLOBAL PYTHON SERVICE/similarLettersModels/''SearchEngineWord_VectorsModern.bin')
    predicted_vector=model1[json_]
     #df2=df2.iloc[:,5:7]
    filename ='D:/Dorosti/100.20python Service/GLOBAL PYTHON SERVICE/similarLettersModels/''wordClustering_modelModern.sav'
    kmeans_model = pickle.load(open(filename, 'rb'))
    df = pd.read_csv('D:/Dorosti/100.20python Service/GLOBAL PYTHON SERVICE/similarLettersModels/''wordClusterNumberModern.csv')
    df5=pd.read_csv('D:/Dorosti/100.20python Service/GLOBAL PYTHON SERVICE/similarLettersModels/''WordsAvgVectorModern')
    f=kmeans_model.predict(predicted_vector.reshape(1,-1).astype(float))
    rslt_df = df.loc[df['ClusterName'] == f[0]]   
    from scipy import spatial
    #XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
    S=[]
    print(len(df5))
    for i in range(0, len(df5)):
       S.append(df5.iloc[i,1:])
    k=[]
    for i in rslt_df.iloc[:,1]:
        cosine_similarity = 1 - spatial.distance.cosine(S[i], predicted_vector)
        k.append(cosine_similarity)

    a = numpy.array(k)
    j=heapq.nlargest(1000, range(len(a)), a.take)
    print(j)
    pi=[]  
    for i in j:
          #print(df2.iloc[rslt_df.iloc[i].Number,:])
          pi.append(df.iloc[rslt_df.iloc[i].Number,3])
          #pi.append(df.iloc[rslt_df.iloc[i].UId,:])
            
          
          
    #return prin.to_json(orient="records") 
    #return  jsonify({'similar':str(prin)})
    return  jsonify({"RESULT":str(pi)})        
    #return df2.iloc[rslt_df.iloc[i].Number,:].to_json(orient="records") 

    
  except:

             return jsonify({'trace': traceback.format_exc()})
    
    
    
    
    
    
    
    
    
    