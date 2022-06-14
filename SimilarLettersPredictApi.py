# -*- coding: utf-8 -*-
"""
Created on Sat Aug 21 12:49:49 2021

@author: Msii
"""
from flask import Flask, request, jsonify
import joblib
import traceback
import pandas as pd
import numpy as np
import sys
import json
from flask import Flask, request, render_template, session, redirect
from sklearn.feature_extraction.text import TfidfVectorizer
import gensim
from gensim.corpora import Dictionary
import parsivar
from parsivar import Tokenizer
from parsivar import Normalizer
import numpy
import itertools
import pandas as pd

from gensim.matutils import jaccard
from gensim.models import LdaModel
from gensim.test.utils import datapath

from flask import Blueprint
SimilarLettersPredictApi_bp = Blueprint('SimilarLettersPredictApi', __name__)




#app=Flask(__name__)


@SimilarLettersPredictApi_bp.route('/SimilarLettersPredictApi',methods=['POST'])



#@app.route('/predict',methods=['POST'])
def SimilarLettersPredictApi():
    try :
               #json_ = request.json
               
               #CustomerName = request.args['CustomerName']
               #CustomerName = request.args.get('CustomerName')
               #forecast_out = request.args['forecast_out']
              # forecast_out = request.args.get('forecast_out',type=int,default=10)
               
               myjson = request.json
               #myjson2=pd.DataFrame(request.json)
               #CustomerName=myjson2.CustomerName
               
               json_=myjson["array"]
               #json_=myjson['Array']
               #json_=myjson[0]
               #json_=myjson.Array
               CustomerName=myjson["customerName"]
               #CustomerName=myjson[1]
               
               #CustomerName = request.args['CustomerName']
               #json_=request.args['Array']
               #json_=json_[0]
               
               
               
               json_=request.json
               CustomerName="Test"
              
               
               
               d=json_[0]["body"]

               #d=pd.DataFrame(json_, index=[0])
               #d=pd.DataFrame(json_)
               #df.columns =['Name', 'Body', 'Type']
               #value_counts = df['Type'].value_counts()

               # Select the values where the count is less than 3 (or 5 if you like)
               #to_remove = value_counts[value_counts <= 20].index

               # Keep rows where the city column is not in to_remove
               #df = df[~df.Type.isin(to_remove)]
               stop_words=['سلام','اً','شرکت','می','های','خواهشمند','فوق','حضور','یافت','پیرو','جناب','فرمائید','شماره','است','این','تشکر','موضوع','نام','خدا','نام خدا','،',':  موضوع','و احترام','بسمه تعالی','به نام خدا','احترا ما','احترام','آقا','خانم','باشد','از','به','که','در','با','با سلام',' و ','.','(',')','،',':',';','با احترام','+','محترم','','1','2','3','4','5','6','7','8','9','0','?','_','__','/','//','-','{}','احترا م']
               my_normalizer = Normalizer()
               from parsivar import Tokenizer
               my_tokenizer = Tokenizer()
               
               #f=str(d['body'])
               f=d
               
               def preprocess(f):
                  for i in range(0,len(f)):
                       deleteWords =stop_words
                       for word in deleteWords:
                         f= f.replace(word,"")
                  my_normalizer = Normalizer()
                  from parsivar import Tokenizer
                  my_tokenizer = Tokenizer()
                  gen_docs=my_tokenizer.tokenize_words(my_normalizer.normalize(f))
                  return gen_docs
               NewLetter=preprocess(f)
               
               #temp_file = datapath('C:/Users/paya8/Desktop/GLOBAL PYTHON SERVICE/similarLettersModel/'+str(CustomerName)+'similarLettermodel')
               #D:/Dorosti/100.20python Service/GLOBAL PYTHON SERVICE/similarLettersModels
               #MYmodel = LdaModel.load('C:/Users/Msii/Desktop/similarLettters/SavedModel/'+str(CustomerName)+'model')
               
               #MYmodel = LdaModel.load('C:/Users/paya8/Desktop/GLOBAL PYTHON SERVICE/similarLettersModels/'+str(CustomerName)+'similarLettermodel')
               MYmodel = LdaModel.load('D:/Dorosti/100.20python Service/GLOBAL PYTHON SERVICE/similarLettersModels/'+str(CustomerName)+'similarLettermodel')
    
    
              #MYmodel=LdaModel.load(temp_file)
              

               def get_most_likely_topic(doc):
                  bow = MYmodel.id2word.doc2bow(doc)
                  topics, probabilities = zip(*MYmodel.get_document_topics(bow))
                  max_p = max(probabilities)
                  topic = topics[probabilities.index(max_p)]
                  return topic
              
               newtopic=get_most_likely_topic(NewLetter)
               #df=pd.read_csv('C:/Users/paya8/Desktop/GLOBAL PYTHON SERVICE/similarLettersModels/'+str(CustomerName)+'.csv')
               df=pd.read_csv('D:/Dorosti/100.20python Service/GLOBAL PYTHON SERVICE/similarLettersModels/'+str(CustomerName)+'.csv')
               z=df['topic'].tolist()
               h=df['doc'].tolist()
               s=df['uID'].tolist()
               topics=[]
               #p=df.iloc(df['topic']==newtopic,'doc')
               #p=df.query('topic'==str(newtopic))['doc']
               p=df[df['topic']==newtopic]['doc']
               m=df[df['topic']==newtopic]['uID']
               #for i, t in enumerate(df):
               p=p.tolist()
               m=m.tolist()
                      
                      # if (df['topic'].iloc==newtopic):
                      #topics.append(df.loc(df['topic']==newtopic,'doc').iloc[0])
               weight=[]
               for i in range(0,len(p)):
               
    
                  bow1, bow2 = NewLetter,p[i]
                  distance = jaccard(bow1, bow2)
              
                  weight.append(1/(distance+1))
                  #weight.append((distance+1))
               
                 # No documentatio
               import heapq
               import numpy
               a = numpy.array(weight)
               j=heapq.nlargest(10, range(len(a)), a.take)
               H=[]
               for z in j:
                   H.append(int(m[z]))
                   

               
               return jsonify({"RESULT": str(H)})
               #return jsonify({"RESULT": str(h[5])}) 
               #return jsonify({"RESULT":str(json_)})
               #return json.dumps({"RESULT":"hello"})

               
               
               
               
               
              

               
               
    except:

               return jsonify({'trace': traceback.format_exc()})
    #else:
       # print ('Train the model first')
       # re
    
    
    
    
    
    
    
    
    
    
    
    
    
'''if __name__ == '__main__':
    
    
    try:
        port = int(sys.argv[1]) # This is for a command-line input
    except:
        port = 19445 # If you don't provide any port the port will be set to 12345

    
    #model = joblib.load('xgb.pkl')
    #model = joblib.load('xgb2 .pkl')
    

app.run(port=port, debug=True)'''
    