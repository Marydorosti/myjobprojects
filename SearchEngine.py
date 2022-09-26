
from flask import Flask, request, jsonify
#import joblib
import traceback
import pandas as pd
import numpy as np
#import sys
import json
#from flask import Flask, request, render_template, session, redirect
from sklearn.cluster import KMeans
import pickle
from parsivar import Tokenizer
from parsivar import Normalizer
import fasttext
import pyodbc
import re

from flask import Blueprint
SearchEngineTrain_bp = Blueprint('SearchEngineTrain', __name__)

@SearchEngineTrain_bp.route('/SearchEngineTrain',methods=['POST'])
def SearchEngineTrain():
    
    
    
 try:  

    my_Normalizer = Normalizer()    
    conn=pyodbc.connect('Driver={Sql Server};'
                                   'Server=192.168.100.17\\SQL2019;'
                                   'Database=Modern_Master;'
                                   'UID=sa;'
                                   'PWD=PAYA+master;'
                                   )
    
     
    with open('D://Dorosti//PayaSoft.BI//PythonCodes//similarLettersModels//SearchEngineQuery.txt' ,'r')as file:
                   query=file.read()
                   
                   
    cursor = conn.cursor()  
    cursor.close()      
    GF=pd.read_sql_query(query,conn)
    df=pd.DataFrame(GF)
    df.dropna()
    RANDOM_SEED = 42
    np.random.seed(RANDOM_SEED)
    f=list(df['DscGds'])

    for i in range(0,len(f)):
      f[i]=re.sub(r'[A-Za-z0-9-]',r'',f[i])
      f[i]=my_Normalizer.normalize(f[i])
   
#print(df.head()) 
    print(f[0])  
    with open('readme.txt', 'w', encoding="utf-8") as g:
      for line in f:
        g.write(line)
        g.write('\n')
        
    with open('readme.txt', 'r',encoding="utf-8") as file:
       text = file.read()
        
    model1 = fasttext.train_unsupervised('readme.txt', model='skipgram',dim=50,ws=5,epoch=500)
    word_vector=[]
    for i in range(0,len(f)):
    
        word_vector.append(model1[f[i]])
    vectors=pd.DataFrame(word_vector)
    vectors.to_csv('D:/Dorosti/100.20python Service/GLOBAL PYTHON SERVICE/similarLettersModels/''WordsAvgVectorModern')
    
     
     
  
    
################*KmeansAlgorithm*###################################

    kmeans = KMeans(
        init="random",
        n_clusters=4,
        n_init=10,
        max_iter=2000,
        random_state=42 )


    kmeans.fit(word_vector)

    labels=kmeans.labels_

    a=[]
    for i in range(0,len(labels)):
        a.append(i)
   
    df6 = pd.DataFrame({'Number' : a,'ClusterName' : labels,'UId':GF.iloc[0:len(a),0]}, columns=['Number','ClusterName','UId'])
    #'C:/Users/paya8/Desktop/GLOBAL PYTHON SERVICE/similarLettersModels/'+str(CustomerName)+'.csv'
    df6.to_csv('D:/Dorosti/100.20python Service/GLOBAL PYTHON SERVICE/similarLettersModels/''wordClusterNumberModern.csv')


    #filename = 'Clustering_model.sav'
    filename='D:/Dorosti/100.20python Service/GLOBAL PYTHON SERVICE/similarLettersModels/''wordClustering_modelModern.sav'
    pickle.dump(kmeans, open(filename, 'wb'))

    #model1.save_model("Word_Vectors.bin")
    model1.save_model('D:/Dorosti/100.20python Service/GLOBAL PYTHON SERVICE/similarLettersModels/''SearchEngineWord_VectorsModern.bin')

    
    return jsonify({"RESULT":str("Model Dumped")})
    #return "null"

 
 except:
     
     return jsonify({'trace': traceback.format_exc()})
     
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    