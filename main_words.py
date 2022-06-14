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
from sklearn.feature_extraction.text import TfidfVectorizer


# Your API definition

from flask import Blueprint
main_words_bp = Blueprint('main_words', __name__)

#app = Flask(__name__)

def preprocess_data(json_):
    
    df=pd.DataFrame(json_)
    #df=json_
    #df=pd.read_json('json_')
    df.columns =['Name', 'Body', 'Type']
    #import numpy as np
    value_counts = df['Type'].value_counts()

# Select the values where the count is less than 3 (or 5 if you like)
    to_remove = value_counts[value_counts <= 50].index
    

# Keep rows where the city column is not in to_remove
    df = df[~df.Type.isin(to_remove)]
    
    #d=len(df.Type.value_counts())
    for i in range(0,len(df.Type.unique())-1):
        df['Body'][df['Type'] == df.Type.unique()[i]].to_csv('D:/Dorosti/100.20python Service/GLOBAL PYTHON SERVICE/MainWordModels/'+'a'+str(i)+'.txt')
    text=[]
    for i in range(0,len(df.Type.unique())-1):

#'C:/Users/paya8/Desktop/100.20python Service/GLOBAL PYTHON SERVICE/MainWordModels/'+str( CustomerName)
       with open('D:/Dorosti/100.20python Service/GLOBAL PYTHON SERVICE/MainWordModels/'+'a'+str(i)+'.txt', 'r', encoding="utf-8") as file:

         text.append(file.read())
    stop_words=['سلام','اً','ذیل','شرکت','می','های','خواهشمند','فوق','حضور','یافت','پیرو','جناب','فرمائید','شماره','است','این','تشکر','موضوع','نام','خدا','نام خدا','،',':  موضوع','و احترام','بسمه تعالی','به نام خدا','احترا ما','احترام','آقا','خانم','باشد','از','به','که','در','با','شد','برای','کرد','با سلام',' و ','تشار','.','(',')','،',':',';','با احترام','صل','ص','','بر ای','+','محترم','',',','احترا م']
    for i in range(0,len(df.Type.unique())-1):
       deleteWords =stop_words
       for word in deleteWords:
         text[i]= text[i].replace(word,"")
         #text[i]=re.sub('[^A-Za-z0-9]+','',text[i])
         #f[i]=re.sub(r'[A-Za-z0-9]',r'',f[i])
    for i in range(0,len(df.Type.unique())-1): 
        text[i]=re.sub(r'[A-Za-z0-9""<<>>+=()-?/\$#@&*!]',r'',text[i])
       
        #text[i]=text[i].encode("utf-8")
    for i in range(0,len(df.Type.unique())-1):
         #text[i]=re.sub("ی","ي",text[i])
         
         text[i]=re.sub("ي","ی"  ,text[i])
        
    for i in range(0,len(df.Type.unique())-1):
      
      my_normalizer = Normalizer()
      
      from parsivar import Tokenizer
      my_tokenizer = Tokenizer()
      #text[i] = my_normalizer.normalize(text[i])
      text[i] = my_tokenizer.tokenize_words(my_normalizer.normalize(text[i]))
    import sys
    wordfreq={}
#wordfreq1=[]
#wordfreq=[{}]
    for i in range(0,len(df.Type.unique())-1):


   #words = text[i].split()
        words=text[i]
   #wordfreq=worsfr
   
        unwanted_chars = ".,-_,[1:9] ,،,:,'| ','!','#','نامه','<>','؟','Body',(and so on),?,1,2,3,4,5,6,7,8,9,۰,۱,۲,۳,۴,۵,۶,۷,۸,۹,'','+',/,\,10"
        wordfreq [i]= {}
        for raw_word in words:
           word = raw_word.strip(unwanted_chars)
           if word not in wordfreq[i]:
                wordfreq[i][word] = 0 
           wordfreq[i][word] += 1 
    
    all_values=[]
    max_values=[]
    min_values=[]
    uniqueValues=[]
    std=[]
    mean=[]

    for i in range(0,len(df.Type.unique())-1):
         import numpy as np
         wordfreq[i].pop('')



         all_values.append(wordfreq[i].values())
         max_values.append(max(all_values[i]))
         min_values.append(min(all_values[i]))
         uniqueValues.append(list(set(wordfreq[i].values())))
   
         std.append(np.std(uniqueValues[i]))
         mean.append(np.mean(uniqueValues[i]))
    z=[] 
    for i in range (0,len(df.Type.unique())):
        z.append([])
    #z=[[],[],[],[]]

    for i in range(0,len(df.Type.unique())-1):
      for key, value in dict(wordfreq[i]).items():
        
        
        
        if (min_values[i]+3*mean[i] <value< max_values[i]-2*mean[i]):
            z[i].append((value,str(key)))
            #z.append(q)
    d=len(df.Type.unique())
    #z=0
    m=df.Type.unique() 
    
    '''vectorizer=TfidfVectorizer(max_features=10)
    X=vectorizer.fit_transform(text)
    terms=vectorizer.get_feature_names()'''



      
    return(z,d,m)
    #return(df)
   
   
    
@main_words_bp.route('/main_words',methods=['POST'])
#@app.route('/predict',methods=['POST'])_
def main_words():
    try:
        
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
        
        
        
        GF=GF.iloc[:,1:]
        
        #json_=request.json
        json_=GF
        
        #CustomerName="Aramesh"
        # myjson=request.json
        myjson=request.json
        #json_=myjson['Array']
        df1=pd.DataFrame(json_)
        #CustomerName=myjson['CustomerName']
        CustomerName="Test"
        #CustomerName="Aram"
        #'D:/Dorosti/100.20python Service/GLOBAL PYTHON SERVICE/LetterRecommendModels/'+str( CustomerName)+'.csv'
        
        z,d,m=preprocess_data(json_)
        #d=preprocess_data(json_)
        #return(str(z[-3:][0][1]))
        #return(str(j[-3:][0][1]))
        p=[]
        group=[]
        word1=[]
        word2=[]
        word3=[]
        for i in range(0,d-1):
           #return str((" کلمات کلیدی" + str(m[i]) + ":" + z[i][-3:][0][1],z[i][-3:][1][1],z[i][-3:][2][1]))
           #return jsonify({'کلمات کلیدی': str(m[i]) + ":" + z[i][-3:][0][1],z[i][-3:][1][1],z[i][-3:][2][1])})
           #p.append({str(m[i]), z[i][-3:][0][1],z[i][-3:][1][1],z[i][-3:][2][1]})
           group.append(str(m[i]))
           word1.append(z[i][-3:][0][1])
           word2.append(z[i][-3:][1][1])
           word3.append(z[i][-3:][2][1])
           p.append({str(m[i]), z[i][-3:][0][1],z[i][-3:][1][1],z[i][-3:][2][1]})
           #return jsonify({"کلمات کلیدی گروه"+" "+str(m[i]): ( z[i][-3:][0][1],z[i][-3:][1][1],z[i][-3:][2][1])})
        #for i in range(0,len(p)):
        #s=pd.DataFrame(p,columns=["group","word1","word2","word3"])
        #s=pd.DataFrame(zip*(group,word1,word2,word3),columns=["group","word1","word2","word3"])
        
        
        s=pd.DataFrame({'group':group,'word1':word1,'word2':word2,'word3':word3})
        l=group
        b=word1
        a=word2
        c=word3
        
        
        #C:\Users\paya8\Desktop\100.20python Service\GLOBAL PYTHON SERVICE\MainWordModels
        #encoding="utf-8"
        #columns=["group","word1","word2","word3"]
        #import csv 
        
        #with open('D:/Dorosti/100.20python Service/GLOBAL PYTHON SERVICE/MainWordModels/'+str( CustomerName)+'MainWords'+'.csv','w',encoding="utf-8") as f:
            #write=csv.writer(f)
            #write.writerow(columns)
            #write.writerows(p)
            
            
        s.to_csv('D:/Dorosti/100.20python Service/GLOBAL PYTHON SERVICE/MainWordModels/'+str( CustomerName)+'MainWords'+'.csv',encoding="utf-8",index=True)
        
        
        #s.to_csv('C:\Users\paya8\Desktop\100.20python Service\GLOBAL PYTHON SERVICE\MainWordModels\'+ str( CustomerName) +'.csv')  
            
            
            
        #return(str(p))
        #return(jsonify({'Result':str("model dumped!")}))
        return(jsonify({'Result':str(p)}))
    
        #return(jsonify(p))
        #return(str(uniqueValues))
        
    

    except:
        
        return jsonify({'trace': traceback.format_exc()})
    
    
'''if __name__ == '__main__':
    try:
        port = int(sys.argv[1]) # This is for a command-line input
    except:
     port = 15444 # If you don't provide any port the port will be set to 12345

     app.run(port=port, debug=True)
'''




































