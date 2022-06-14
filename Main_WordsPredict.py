# -*- coding: utf-8 -*-
"""
Created on Sun Mar 13 08:28:12 2022

@author: m.dorosti
"""

from flask import Flask, request, jsonify
import joblib
import traceback
import pandas as pd
import numpy as np
from parsivar import Tokenizer
import gensim
from json import dumps
import re
import yake
from parsivar import Normalizer
import os.path

from flask import Blueprint,make_response
Main_WordsPredict_bp = Blueprint('Main_WordsPredict', __name__)



@Main_WordsPredict_bp.route('/Main_WordsPredict',methods=['POST'])

    

def Main_WordsPredict():
     try:
        myjson=request.json
       
        
        json_=myjson['Array']
        #json_=request.json
        df1=pd.DataFrame(json_)
        CustomerName=myjson['CustomerName']
        #CustomerName="Test"
        #'D:/Dorosti/100.20python Service/GLOBAL PYTHON SERVICE/LetterRecommendModels/'+str( CustomerName)+'.csv'
        #CustomerName="Aramesh"
        
        file_exist=os.path.isfile('D:/Dorosti/100.20python Service/GLOBAL PYTHON SERVICE/MainWordModels/'+str( CustomerName)+'MainWords'+'.csv')
        if file_exist==True:
        
          df=pd.read_csv('D:/Dorosti/100.20python Service/GLOBAL PYTHON SERVICE/MainWordModels/'+str( CustomerName)+'MainWords'+'.csv')
        #if df>0:
            
          o=df['Group'].tolist()
          s=[]
        #for i in range (0,len(df)):
          for i in range(0,len(o)):   
            #if df1.Type[0]==df.iloc[i,3]:
              if df1.Type[0]==o[i]:
            #if df1.Type[0]==df.word3[0]:
                
                  s.append(df.iloc[i,1:])
          #p=[x for x in s[0] if x   in h]
                #s.append(df.iloc[i,1:][0])
                #s.append()
        
        P=df1.Body[0]
        #P=re.sub("ی","ي",P)
        P=re.sub("ي","ی"  ,P)
        
        
        
        stop_words=['سلام','اً','ذیل','شرکت','خواهشمند','فوق','حضور','یافت','پیرو','جناب','فرمائید','شماره','لطفا',' این ',' است ','تشکر','موضوع','نام','خدا','نام خدا','،',':  موضوع','و احترام','بسمه تعالی','به نام خدا','احترا ما','احترام','آقا','خانم','باشد','برای','کرد','با سلام','.','(',')','،',':',';','با احترام','بر ای','+',' مهندس ',' آقای ',' می رساند ',' دارم ',' بود ',' گردید ',' کنید ','محترم','',',','احترا م']
        for word in stop_words:
            P= P.replace(word,"")
         #text[i]=re.sub('[^A-Za-z0-9]+','',text[i])
         #f[i]=re.sub(r'[A-Za-z0-9]',r'',f[i])
       
        #P=re.sub(r'[A-Za-z0-9""<<>>+=()-?/\$#@&*!]',r'',P)
        P=re.sub(r'[0-9""<<>>+=()-?/\$#@&*!]',r'',P)
        #P=re.sub(r'(شما'')|("" احترام)|(''محترم'')|(''با احترام'')|(''سلام'')|(''کرد'')|(''نمایید'')|( برای)|( باشد)|(""در"")|("به"")|(''می'')|(می'')',r'',P)
        my_normalizer=Normalizer()
        P=re.sub(r'(""می"")|(""مهندس"")|(""به"")|(""در"")|(""باشد"")|(""برای"")|(""نمایید"")|(""کرد"")|(""سلام"")|(""شما"")|(""با احترام"")|(""محترم"")|(احترام"")|(شما"")|(""وقت"")|(""بخیر"")',r'',P)
        P=my_normalizer.normalize(P)
        
        
        max_ngram_size=1
        deduplication_theshold=0.95
        deduplication_algo='seqm'
        windowSize=5
        R=[P]
        #top=NumberOfKeyWords
        
        for j in R:
            
          if len(j.split())<=3:
              
            NumberOfKeyWords=1
            
          elif 3<len(j.split())<=10:
            
              NumberOfKeyWords=3
          else:
              
              NumberOfKeyWords=6
    
        kw=yake.KeywordExtractor(top=NumberOfKeyWords,n=max_ngram_size,dedupLim=deduplication_theshold,dedupFunc=deduplication_algo,windowsSize= windowSize,features=None)
        KeyWords=kw.extract_keywords(P)
        KeyWords.sort(key=lambda x:x[1])
        r=[]
        for i in range (0,NumberOfKeyWords):
            r.append(KeyWords[i][0])
        #P=P.encode("utf-8")
        my_tokenizer=Tokenizer()
        gen_docs=[my_tokenizer.tokenize_words(P)]
        #gen_docs=gen_docs.encode("utf-8")
        dictionary=gensim.corpora.Dictionary(gen_docs)
        #gen_docs2=[my_tokenizer.tokenize_words(s[0])]
        #dictionary2=gensim.corpora.Dictionary(gen_docs2)
        k=[]
        h=[]
        b=[dictionary]
        #k=[80502787,80502777,80502785]
        for j in range(0,len(gen_docs)):
                
                #if i==dictionary[j]:
                    #k.append(dictionary[j])
                    #.append(dictionary[j])
                    h.append(gen_docs[j])
                
        
            #for j in range(0,len(dictionary)):
        if file_exist==True:
          for i in s[0]:
            for j in R:
                 if i in j.split():     
                    #if i==h[j]:
                
                #if i==h[j]:
                    #k.append(dictionary[j])
                         k.append(i)
                    #h.append()
                
        #p=[x for x in s[0] if x   in h]
        #p=set(s[0]).intersection(h)
        #p=[x for x in s[0] if x   in P.split()]
        
        #j=[s[0],h]
        #value=set(h)-set(s[0])
        #value=set(h)&set(s[0])
        
        #array='{"KeyWords":k}'
        
        #k=s[0]♠
        #if len(k)!=0:
            #return(jsonify({"KeyWords":str(s[0])}))
            #return(make_response(dumps({"KeyWords":k}))) 
            #return(make_response({"KeyWords":dumps(k)}))
        #else:
            
            #return None
            #return(jsonify({'KeyWords':None}))
            #return(jsonify({'KeyWords':str(s[0])}))
            #return(jsonify({"KeyWords":k}))
            #return(make_response(dumps({"KeyWords":str(s[0])}))) 
            #return(make_response(dumps(k)))
        if len(k)>0:    
           A=r+k
        else:
           A=r
            
        #R=k
            
            #return(make_response({"KeyWords":dumps(k)}))
        return(make_response({"KeyWords":dumps(A)})) 
               
     except:
          
          return jsonify({'trace': traceback.format_exc()})
        
        
        
        
        
        