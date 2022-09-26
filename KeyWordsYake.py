# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 16:27:21 2022

@author: m.dorosti
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 14:30:16 2021

@author: m.dorosti
"""

from flask import Flask, request, jsonify
import joblib
import traceback
import pandas as pd
import numpy as np
import sys
import json
from flask import Flask, request, render_template, session, redirect
from statistics import mean
import numpy as np
import parsivar
from parsivar import Normalizer
from hazm import *
import pyodbc
import re
import yake


# Your API definition

from flask import Blueprint
KeyWordsYake_bp = Blueprint('KeyWordsYake', __name__)

with open('PythonPathConfig.txt', 'r', encoding="utf-8") as file:   

        path=file.read()

#path=r'D:/Dorosti/PayaSoft.BI/PythonCodes'




#app = Flask(__name__)

def preprocess_data(json_,CustomerName):
    
    df=pd.DataFrame(json_)
    my_normalizer = Normalizer()
    #df=json_
    #df=pd.read_json('json_')
    df.columns =['Name', 'Body', 'Type']
    #import numpy as np
    value_counts = df['Type'].value_counts()

# Select the values where the count is less than 3 (or 5 if you like)
    to_remove = value_counts[value_counts <= 20].index
    

# Keep rows where the city column is not in to_remove
    df = df[~df.Type.isin(to_remove)]
    
    #d=len(df.Type.value_counts())
    for i in range(0,len(df.Type.unique())-1):
        #df['Body'][df['Type'] == df.Type.unique()[i]].to_csv('D:/Dorosti/100.20python Service/GLOBAL PYTHON SERVICE/MainWordModels'+'a'+str(i)+'.txt')
        df['Body'][df['Type'] == df.Type.unique()[i]].to_csv(path+'/MainWordModels/'+str( CustomerName)+'a'+str(i)+'.txt')
    text=[]
    for i in range(0,len(df.Type.unique())-1):


        with open(path+'/MainWordModels/'+str( CustomerName)+'a'+str(i)+'.txt', 'r', encoding="utf-8") as file: 

         text.append(file.read())
         
    for i in range(0,len(df.Type.unique())-1):
         #text[i]=re.sub("ی","ي",text[i])
         
         text[i]=re.sub("ي","ی"  ,text[i])     
    stop_words=['سلام','اً','ذیل','شرکت','خواهشمند','فوق','حضور','یافت','پیرو','جناب','فرمائید','شماره','است','تشکر','موضوع','نام','خدا','نام خدا','،',':  موضوع','و احترام','بسمه تعالی','به نام خدا','احترا ما','احترام','آقا','خانم','باشد','به','که',' در','با','شد','برای','کرد','با سلام',' و ','.','(',')','،',':',';','با احترام','','بر ای','+','محترم','',',','احترا م']
    for i in range(0,len(df.Type.unique())-1):
       deleteWords =stop_words
       for word in deleteWords:
         text[i]= text[i].replace(word,"")
         #text[i]=re.sub('[^A-Za-z0-9]+','',text[i])
         #f[i]=re.sub(r'[A-Za-z0-9]',r'',f[i])
    for i in range(0,len(df.Type.unique())-1): 
        text[i]=re.sub(r'[A-Za-z0-9""<<>>+=()-?/\$#@&*!]',r'',text[i])
        text[i] = my_normalizer.normalize(text[i])
       
       
        
    
    
    max_ngram_size=1
    deduplication_theshold=0.95
    deduplication_algo='seqm'
    windowSize=3
    #top=NumberOfKeyWords
    NumberOfKeyWords=7
    
    
    kw=yake.KeywordExtractor(top=NumberOfKeyWords,n=max_ngram_size,dedupLim=deduplication_theshold,dedupFunc=deduplication_algo,windowsSize= windowSize,features=None)
    z=[] 
    #for i in range (0,len(df.Type.unique())-1):
        #z.append([])
    
    for i in range(0,len(df.Type.unique())-1):
        
    
    
      KeyWords=kw.extract_keywords(text[i])
    
      KeyWords.sort(key=lambda x:x[1]) 
      z.append(KeyWords)
    #g= sorted(KeyWords,key=lambda x:x[0])   
            
       
    d=len(df.Type.unique())
    #z=0
    m=df.Type.unique()  





      
    return KeyWords,m,d,z
    #return(df)
   

#Data Source=192.168.100.17\SQL2019;Initial Catalog=PayaAfzarPendarData;User ID=sa;Password=PAYA+master;App=NewPendar;MultipleActiveResultSets=true;Encrypt=True;TrustServerCertificate=True;    
#    
@KeyWordsYake_bp.route('/KeyWordsYake',methods=['POST'])
#@app.route('/predict',methods=['POST'])_
def KeyWordsYake():
    try:
        myjson=request.json
        ConnectionString=myjson['ConnectionString']
        s=ConnectionString.split(';')
        '''conn=pyodbc.connect('Driver={Sql Server};'
                                   'Server=192.168.100.17\\SQL2019;'
                                   'Database=PayaAfzarPendarData;'
                                  'UID=sa;'
                                   'PWD=PAYA+master;'
                                   )'''
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
    
               
        '''GF=pd.read_sql_query("SELECT o.Id, \
               o.DisplayName, \
               o.BodyText, \
               c.DisplayName AS Type \
               FROM dbo.AUT_Objects o \
               INNER JOIN dbo.AUT_Classes c \
               ON c.Id = o.Class \
               WHERE o.BaseClass = 2 \
               AND o.BodyText <> '' "  ,conn)'''
        
        
        #with open('D://Dorosti//100.20python Service//GLOBAL PYTHON SERVICE//MainWordModels//MainWordsQuery.txt' ,'r')as file:
        with open(path+'/MainWordModels/MainWordsQuery.txt' ,'r')as file:
                   query=file.read()
                   
                   
        cursor = conn.cursor()   

           
        GF=pd.read_sql_query(query,conn)        
        #json_=GF
        GF=GF.iloc[:,1:]
        
        #json_=request.json
        json_=GF
        
        #CustomerName="Aramesh"
        # myjson=request.json
        
        #json_=myjson['Array']
        df1=pd.DataFrame(json_)
        CustomerName=myjson['CustomerName']
        #CustomerName="raha"
        #CustomerName="Aram"
        #'D:/Dorosti/100.20python Service/GLOBAL PYTHON SERVICE/LetterRecommendModels/'+str( CustomerName)+'.csv'
        
        g,m,d,z=preprocess_data(json_,CustomerName)
        p=[]
        for i in range (0,len(z)):
           p.append([])
           
           
        
        for i in range (0,len(z)):
            for j in range (0,len(z[i])-1):
                p[i].append(z[i][j][0])
                
                
        
        
       
        group=[]
        word1=[]
        word2=[]
        word3=[]
        #for i in range(0,d-1):
        for i in range (0,len(z)):
            group.append(str(m[i]))
            
        s=pd.DataFrame(p) 
        
        
        s['Group']=group
        
          
       
            
        #s.to_csv('D:/Dorosti/100.20python Service/GLOBAL PYTHON SERVICE/MainWordModels/'+str( CustomerName)+'MainWords'+'.csv',encoding="utf-8",index=True)
        s.to_csv(path+'/MainWordModels/'+str( CustomerName)+'MainWords'+'.csv',encoding="utf-8",index=True)
        
        
        #s.to_csv('C:\Users\paya8\Desktop\100.20python Service\GLOBAL PYTHON SERVICE\MainWordModels\'+ str( CustomerName) +'.csv')  
            
        
        
        #return(str(s))
    
        return(jsonify("model Trained"))
        #return(str(uniqueValues))
        
    

    except:
        
        return jsonify({'trace': traceback.format_exc()})   




































