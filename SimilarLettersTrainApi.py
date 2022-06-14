# -*- coding: utf-8 -*-
"""
Created on Sun Aug 22 08:28:52 2021

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
import pyodbc
import unicodecsv as csv
import re

from flask import Blueprint
SimilarLettersTrainApi_bp = Blueprint('SimilarLettersTrainApi', __name__)


#app=Flask(__name__)

'''"DbConnStr": "Data Source=192.168.100.17\\SQL2019;Initial Catalog=PayaAfzarPendarData;User ID=sa;Password=PAYA+master;App=NewPendar;MultipleActiveResultSets=true;Encrypt=True;TrustServerCertificate=True;"'''

'''def preprocess(f):
    for i in range(0,len(f)):
       deleteWords =stop_words
       for word in deleteWords:
         f= f.replace(word,"")
    my_normalizer = Normalizer()
    from parsivar import Tokenizer
    my_tokenizer = Tokenizer()
    gen_docs=my_tokenizer.tokenize_words(my_normalizer.normalize(f))
    return gen_docs'''
    


@SimilarLettersTrainApi_bp.route('/SimilarLettersTrainApi',methods=['POST'])
#@app.route('/train',methods=['POST'])
def SimilarLettersTrainApi():
    try :
               #json_ = request.json
               #print(json_)
               conn=pyodbc.connect('Driver={Sql Server};'
                                   'Server=192.168.100.17\\SQL2019;'
                                   'Database=PayaAfzarPendarData;'
                                   'UID=sa;'
                                   'PWD=PAYA+master;'
                                   )
               
               GF=pd.read_sql_query("SELECT o.Id, \
               o.DisplayName, \
               o.BodyText, \
               c.DisplayName AS Type \
               FROM dbo.AUT_Objects o \
               INNER JOIN dbo.AUT_Classes c \
               ON c.Id = o.Class \
               WHERE o.BaseClass = 2 \
               AND o.BodyText <> '' "  ,conn)
                 
               #gf=list(GF.iloc[:,1])
              
               #CustomerName = request.args['CustomerName']
               #CustomerName = request.args.get("CustomerName")
               
               #myjson = request.json
               #json_=request.json
               #myjson2=pd.DataFrame(request.json)
               #CustomerName=myjson2.CustomerName
               
               #json_=myjson['Array']
               #CustomerName=myjson['CustomerName']
               GF=GF[31:5000]
               CustomerName='Test'
               json_=GF
               #json_=request.json
               df=json_
               df.dropna()
               #df=df.to_csv('encoded.csv')
               #df=pd.read_csv('encoded.csv',encoding='utf-8')
               df=df.to_csv('encoded.csv')
               #encoding='ISO-8859-1'
               with open('encoded.csv','r',encoding='utf-8') as f:
                    df=pd.read_csv(f,error_bad_lines=False)
               
               
               #df=df.iloc[0:1000,:]
               
               #df=pd.DataFrame(json_)
               #df.columns =['Name', 'Body', 'Type']
               
               value_counts = df['Type'].value_counts()
               #value_counts=df.iloc[:,3].value_counts

# Select the values where the count is less than 3 (or 5 if you like)
               to_remove = value_counts[value_counts <= 20].index

# Keep rows where the city column is not in to_remove
               df = df[~df.Type.isin(to_remove)]
               f=list(df['BodyText'])
               stop_words=['سلام','اً','شرکت','می','های','خواهشمند','فوق','حضور','یافت','پیرو','جناب','فرمائید','شماره','است','این','تشکر','موضوع','نام','خدا','نام خدا','،',':  موضوع','و احترام','بسمه تعالی','به نام خدا','احترا ما','احترام','آقا','خانم','باشد','از','به','که','در','با','با سلام',' و ','.','(',')','،',':',';','با احترام','+','محترم','','1','2','3','4','5','6','7','8','9','0','?','_','__','/','//','-','{}','احترا م']
           
               my_normalizer = Normalizer()
               from parsivar import Tokenizer
               my_tokenizer = Tokenizer()
               k=len(df.Type.unique())
               unwanted_chars = "'تشکر','خواهشمند','شرکت','آقای','آقا','خانم','محترم','{}','با سلام','سلام','احترام','با','بسمه تعالی','تعالی','بسمه','احترام',.,-_,[1:9] ,،,:,'| ','!','#','نامه','<>','؟','Body',(and so on),?,1,2,3,4,5,6,7,8,9,۰,۱,۲,۳,۴,۵,۶,۷,۸,۹,'','+',/,\,10"
               #for i in range(0,len(f)):
            
                   #f[i] = f[i].strip(unwanted_chars)
               '''f=[]
               for i in range(0,k):
                   f.append(list(df['BodyText'][df['Type'] == df.Type.unique()[i]]))
               f= [i for i in f[i]] '''
               #f=list(df['BodyText'])    
               #r=len(f[5][3])
               #p=len(f[5][20]
               #f=list(df['BodyText'][df['Type'] == df.Type.unique()[0]])
               #f=list(df['BodyText'][df['Type'] == df.Type.unique()[2]])
               #for i in range(0,len(f)):
                   #for j in range(0,len(f[i])):
                       #return(f[i][j])
               for i in range(0,len(f)):
                   for j in range(0,len(f[i])):
                     deleteWords =stop_words
                     #if type(f[i][j])==str:
                     for word in deleteWords:
                         f[i]= f[i].replace(word,"")
                         
                         #f[i]=f[i].strip(word)
               for i in range(0,len(f)):
                   f[i]=re.sub(r'[A-Za-z0-9]',r'',f[i])
                   
               #gen_docs=[]          
               #for i in range (0,len(f)):
               gen_docs=[my_tokenizer.tokenize_words(my_normalizer.normalize(text)) for text in f]
                 #gen_docs.append([my_tokenizer.tokenize_words(my_normalizer.normalize(text)) for text in f[i]])
               #dictionary=[] 
               #bow_corpus=[]
               #for i in range (0,len(gen_docs)):
               dictionary = gensim.corpora.Dictionary(gen_docs)
                 #dictionary.append(gensim.corpora.Dictionary(gen_docs[i]))
                 #dictionary[i].filter_extremes(no_below=15, no_above=0.5, keep_n=100000) 
               #dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=100000)
               #dictionary.filter_extremes(no_below=1, no_above=0.5, keep_n=100)
               bow_corpus = [dictionary.doc2bow(doc) for doc in gen_docs]
                 #bow_corpus.append([dictionary[i].doc2bow(doc) for doc in gen_docs[i]])
               from gensim.corpora import Dictionary


               #bow_corpus=[i for i in bow_corpus]

               numpy.random.seed(1) # setting random seed to get the same results each time.

               from gensim.models import LdaModel
               MYmodel = LdaModel(bow_corpus, id2word=dictionary, num_topics=50, minimum_probability=1e-8)
               #MYmodel.show_topics()
               import itertools
               from gensim.matutils import jaccard

               def get_most_likely_topic(doc):
                  bow = MYmodel.id2word.doc2bow(doc)
                  topics, probabilities = zip(*MYmodel.get_document_topics(bow))
                  max_p = max(probabilities)
                  topic = topics[probabilities.index(max_p)]
                  return topic
            
               print('id\ttopic\tdoc')
               topics=[]
               number=[]
               top=[]
               doc=[]
               #for s in range(0,len(gen_docs)):
               for i, t in enumerate(gen_docs):
                  topic=get_most_likely_topic(t)
                  number.append(i)
                  top.append(topic)
                  doc.append(t)
    
                  print('%d\t%d\t%s' % (i, get_most_likely_topic(t), ' '.join(t)))
                  #if (topic==newtopic):
                      #topics.append(t)
               s=pd.DataFrame()
               s['number']=number
               s['topic']=top
               s['doc']=doc
               s['uID']=df['Id']
               from gensim.test.utils import datapath
                  
                  
                  
               MYmodel.save('D:/Dorosti/100.20python Service/GLOBAL PYTHON SERVICE/similarLettersModels/'+str(CustomerName)+'similarLettermodel')
               
               #with open('D:/Dorosti/100.20python Service/GLOBAL PYTHON SERVICE/similarLettersModels/'+str(CustomerName)+'.csv',"wb")as g:
                   #writer=csv.writer(g,dialect='excel',encoding='utf-8')
                   #writer.writerows(f)
               
               s.to_csv('D:/Dorosti/100.20python Service/GLOBAL PYTHON SERVICE/similarLettersModels/'+str(CustomerName)+'.csv')

                 

               #return jsonify({"RESULT": str("Model dumped!")})
               return jsonify({"result":str(f[20])})
               #return(gen_docs[5])
               #return jsonify({"RESULT": str(bow_corpus)})
               #return jsonify({"RESULT": str(r)})
               #return(str(len(f[0][5])))


               
               
               
               
               
              

               
               
    except:

               return jsonify({'trace': traceback.format_exc()})
    #else:
       # print ('Train the model first')
       # re
    
    
    
    
    
    
    
    
    
    
    
    
    
'''if __name__ == '__main__':
    
    
    try:
        port = int(sys.argv[1]) # This is for a command-line input
    except:
        port = 19444 # If you don't provide any port the port will be set to 12345

    
    #model = joblib.load('xgb.pkl')
    #model = joblib.load('xgb2 .pkl')
    

app.run(port=port, debug=True)'''
    