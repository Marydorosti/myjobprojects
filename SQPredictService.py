# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 08:18:39 2022

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
#import joblib
#import traceback
#import pandas as pd
#import numpy as np
#import sys
#import json
#from flask import Flask, request, render_template, session, redirect
#from sklearn.cluster import KMeans




from flask import Blueprint
SQPredictService_bp = Blueprint('SQPredictService', __name__)

def preprocess_sentence(sentence):
    
    df = pd.read_csv('C:/Users/paya8/Desktop/GLOBAL PYTHON SERVICE/similarQuestionsModels/ClusterNumber.csv')
    df2=pd.read_excel('C:/Users/paya8/Desktop/GLOBAL PYTHON SERVICE/similarQuestionsModels/Porsesh & Pasokh (11) (1).xlsx')
    df5=pd.read_csv('C:/Users/paya8/Desktop/GLOBAL PYTHON SERVICE/similarQuestionsModels/S.csv')
    df2=df2.iloc[:,5:7]
    filename = 'C:/Users/paya8/Desktop/GLOBAL PYTHON SERVICE/similarQuestionsModels/Clustering_model.sav'
    kmeans_model = pickle.load(open(filename, 'rb'))
    stop_words=['؟','حلی','حل','راه','وجود','دارد','چیست','می خواهیم','سلام','اً','شرکت','می','های','خواهشمند','فوق','حضور','یافت','پیرو','جناب','فرمائید','شماره','است','این','تشکر','موضوع','نام','خدا','نام خدا','،',':  موضوع','و احترام','بسمه تعالی','به نام خدا','احترا ما','احترام','آقا','خانم','باشد','از','به','که','در','با','با سلام',' و ','.','(',')','،',':',';','با احترام','+','محترم','','1','2','3','4','5','6','7','8','9','0','?','_','__','/','//','-','{}','احترا م']
    my_normalizer = Normalizer()
    from parsivar import Tokenizer
    my_tokenizer = Tokenizer()
    def preprocess(f):
                  for i in range(0,len(f)):
                       deleteWords =stop_words
                       for word in deleteWords:
                         f= f.replace(word,"")
                  #my_normalizer = Normalizer()
                  #from parsivar import Tokenizer
                  #my_tokenizer = Tokenizer()
                 # gen_docs=my_tokenizer.tokenize_words(my_normalizer.normalize(f))
                  return f
    sentence=preprocess(sentence)

    #sentence = input("Enter a sentence: ")
    words = sentence.split()
    model1 = fasttext.load_model("C:/Users/paya8/Desktop/GLOBAL PYTHON SERVICE/similarQuestionsModels/Word_Vectors.bin")
    #words = sentence.split()
    
    #stop_words=[''سلام','اً','شرکت','می','های','خواهشمند','فوق','حضور','یافت','پیرو','جناب','فرمائید','شماره','است','این','تشکر','موضوع','نام','خدا','نام خدا','،',':  موضوع','و احترام','بسمه تعالی','به نام خدا','احترا ما','احترام','آقا','خانم','باشد','از','به','که','در','با','با سلام',' و ','.','(',')','،',':',';','با احترام','+','محترم','','1','2','3','4','5','6','7','8','9','0','?','_','__','/','//','-','؟','.','{}','احترا ]
    import statistics
    
    new_vectors = []
    for i in words:
        
        new_vectors.append(model1[i])
    return new_vectors,df5,df2,df






@SQPredictService_bp.route('/SQPredictService',methods=['POST'])

def SQPredictService():
  # load the model from disk
  try:
    #jsonString = request.decode("utf-8") 
    #jStr = json.loads(jsonString)
    json_=request.json
    #json_= json_.decode("utf-8")
    sentence= json_ 
    
    #stop_words=['؟','حلی','حل','راه','وجود','دارد','چیست','می خواهیم','سلام','اً','شرکت','می','های','خواهشمند','فوق','حضور','یافت','پیرو','جناب','فرمائید','شماره','است','این','تشکر','موضوع','نام','خدا','نام خدا','،',':  موضوع','و احترام','بسمه تعالی','به نام خدا','احترا ما','احترام','آقا','خانم','باشد','از','به','که','در','با','با سلام',' و ','.','(',')','،',':',';','با احترام','+','محترم','','1','2','3','4','5','6','7','8','9','0','?','_','__','/','//','-','{}','احترا م']
               #my_normalizer = Normalizer()
               #from parsivar import Tokenizer
               #my_tokenizer = Tokenizer()
               
               
    #df = pd.read_csv('C:/Users/paya8/Desktop/GLOBAL PYTHON SERVICE/similarQuestionsModels/ClusterNumber.csv')
    #df2=pd.read_excel('C:/Users/paya8/Desktop/GLOBAL PYTHON SERVICE/similarQuestionsModels/Porsesh & Pasokh (11) (1).xlsx')
    #df5=pd.read_csv('C:/Users/paya8/Desktop/GLOBAL PYTHON SERVICE/similarQuestionsModels/S.csv')
    #df2=df2.iloc[:,5:7]
    filename = 'C:/Users/paya8/Desktop/GLOBAL PYTHON SERVICE/similarQuestionsModels/Clustering_model.sav'
    kmeans_model = pickle.load(open(filename, 'rb'))
    vectors,df5,df2,df=preprocess_sentence(sentence)
    h=[]
    sum=0
    for i in range(0,len(vectors)):
  
      p=np.array(vectors[i])
      sum=sum+p
    
    h.append(sum/len(vectors))

    f=kmeans_model.predict(h)
    print(f)
    S=[]
    print(len(df5))
    for i in range(0, len(df5)):
       S.append(df5.iloc[i,1:])

    print(len(S))
    print(len(S[0]))
    print(f)
    rslt_df = df.loc[df['ClusterName'] == f[0]]

    from scipy import spatial
    k=[]
    for i in rslt_df.iloc[:,0]:
        cosine_similarity = 1 - spatial.distance.cosine(S[i], h[0])
        k.append(cosine_similarity)

    a = numpy.array(k)
    j=heapq.nlargest(3, range(len(a)), a.take)
    print(j)
    prin=[]  
    for i in j:
          #print(df2.iloc[rslt_df.iloc[i].Number,:])
          prin.append(df2.iloc[rslt_df.iloc[i].Number,:])
            
          
          
    #return prin.to_json(orient="records") 
    return  jsonify({'similar':str(prin)})    
    #return df2.iloc[rslt_df.iloc[i].Number,:].to_json(orient="records") 

    
  except:

             return jsonify({'trace': traceback.format_exc()})
    
    
    
    
    
    
    
    
    
    
    
    
    
    