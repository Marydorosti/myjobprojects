# -*- coding: utf-8 -*-
"""
Created on Mon Apr 18 15:32:48 2022

@author: m.dorosti
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Mar  6 15:46:26 2022

@author: m.dorosti
"""

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
similarlettersClustering_bp = Blueprint('similarlettersClustering', __name__)

@similarlettersClustering_bp.route('/similarlettersClustering',methods=['POST'])
def SQClusteringService():
    
    
    
 try:  
     
     
    myjson=request.json 
     
    #json_=myjson['Array']
        #json_=request.json
        
    CustomerName=myjson['CustomerName']
    #CustomerName="ssalam"
    ConnectionString=myjson['ConnectionString']
     
     
    '''conn=pyodbc.connect('Driver={Sql Server};'
                                   'Server=192.168.100.17\\SQL2019;'
                                   'Database=PayaAfzarPendarData;'
                                   'UID=sa;'
                                   'PWD=PAYA+master;'
                                   )'''
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
    
               
    '''GF=pd.read_sql_query("SET TRAN ISOLATION LEVEL READ UNCOMMITTED;\
                         SELECT o.Id, \
               o.DisplayName, \
               o.BodyText, \
               c.DisplayName AS Type \
               FROM dbo.AUT_Objects o \
               INNER JOIN dbo.AUT_Classes c \
               ON c.Id = o.Class \
               WHERE o.BaseClass = 2 \
               AND o.BodyText <> '' "  ,conn)'''
               
    '''GF=pd.read_sql_query("SELECT o.Id, \
               o.DisplayName, \
               o.BodyText, \
               c.DisplayName AS Type \
               FROM dbo.AUT_Objects o \
               INNER JOIN dbo.AUT_Classes c \
               ON c.Id = o.Class \
               WHERE o.BaseClass = 2 \
               AND o.BodyText <> '' "  ,conn) '''      
               
    
                             
    #cursor = conn.cursor()  
     
    with open('D://Dorosti//100.20python Service//GLOBAL PYTHON SERVICE//similarLettersModels//SimilarLettersQuery.txt' ,'r')as file:
                   query=file.read()
                   
                   
    cursor = conn.cursor()  
    cursor.close()
    del cursor

           
    GF=pd.read_sql_query(query,conn)
    
    '''with open('D://Dorosti//100.20python Service//GLOBAL PYTHON SERVICE//MainWordModels//MainWordsQuery.txt' ,'r')as file:
                   query=file.read()
                   
    cursor = conn.cursor()   

           
    GF=pd.read_sql_query(query,conn)'''
                  
               
               #json_=GF
               
    
     
   
    GF=GF[40:1000]
    #CustomerName='Test'
    json_=GF
     
    
    #json_ = request.json
    df=pd.DataFrame(json_)
    df.dropna()
    #df=pd.read_excel('Porsesh & Pasokh (11) (1).xlsx')
    RANDOM_SEED = 42
    np.random.seed(RANDOM_SEED)
    #df=df.iloc[:,1:2]
    #my_normalizer = Normalizer()
    #from parsivar import Tokenizer
    #my_tokenizer = Tokenizer()
    #stop_words=['سلام','اً','شرکت','می','های','خواهشمند','فوق','حضور','یافت','پیرو','جناب','فرمائید','شماره','است','این','تشکر','موضوع','نام','خدا','نام خدا','،',':  موضوع','و احترام','بسمه تعالی','به نام خدا','احترا ما','احترام','آقا','خانم','باشد','از','به','که','در','با','با سلام',' و ','.','(',')','،',':',';','با احترام','+','محترم','','1','2','3','4','5','6','7','8','9','0','?','_','__','/','//','-','؟','.','{}','احترا م']
    #my_normalizer = Normalizer()
    stop_words=['؟','حلی','حل','راه','وجود','دارد','چیست','می خواهیم','سلام','اً','شرکت','می','های','خواهشمند','فوق','حضور','یافت','پیرو','جناب','فرمائید','شماره','است','این','تشکر','موضوع','نام','خدا','نام خدا','،',':  موضوع','و احترام','بسمه تعالی','به نام خدا','احترا ما','احترام','آقا','خانم','باشد','از','به','که','در','با','با سلام',' و ','.','(',')','،',':',';','با احترام','+','محترم','','1','2','3','4','5','6','7','8','9','0','?','_','__','/','//','-','{}','احترا م']
               #my_normalizer = Normalizer()
               #from parsivar import Tokenizer
               #my_tokenizer = Tokenizer()
    #df=pd.DataFrame(json_)
               #df.columns =['Name', 'Body', 'Type']
    value_counts = df['Type'].value_counts()

# Select the values where the count is less than 3 (or 5 if you like)
    to_remove = value_counts[value_counts <= 20].index

# Keep rows where the city column is not in to_remove
    df = df[~df.Type.isin(to_remove)]
    stop_words=['سلام','اً','شرکت','می','های','خواهشمند','فوق','حضور','یافت','پیرو','جناب','فرمائید','شماره','است','این','تشکر','موضوع','نام','خدا','نام خدا','،',':  موضوع','و احترام','بسمه تعالی','به نام خدا','احترا ما','احترام','آقا','خانم','باشد','از','به','که','در','با','با سلام',' و ','.','(',')','،',':',';','با احترام','+','محترم','','1','2','3','4','5','6','7','8','9','0','?','_','__','/','//','-','{}','احترا م']
    my_normalizer = Normalizer()
    from parsivar import Tokenizer
    my_tokenizer = Tokenizer()
               
    #f=list(df['Body'][df['Type'] == df.Type.unique()[0]])
    f=list(df['BodyText'])
               
    for i in range(0,len(f)):
        deleteWords =stop_words
    for word in deleteWords:
            f[i]= f[i].replace(word,"") 
            
     
                         
                         #f[i]=f[i].strip(word)
    for i in range(0,len(f)):
        f[i]=re.sub(r'[A-Za-z0-9]',r'',f[i])        
    #stop_words=[,',',','سلام','اً','شرکت','می','های','خواهشمند','فوق','حضور','یافت','پیرو','جناب','فرمائید','شماره','است','این','تشکر','موضوع','نام','خدا','نام خدا','،',':  موضوع','و احترام','بسمه تعالی','به نام خدا','احترا ما','احترام','آقا','خانم','باشد','از','به','که','در','با','با سلام',' و ','.','(',')','،',':',';','با احترام','+','محترم','','1','2','3','4','5','6','7','8','9','0','?','_','__','/','//','-','؟','.','{}','احترا م']
    #my_normalizer = Normalizer()


    #stop_words=['سلام','اً','شرکت','می','های','خواهشمند','فوق','حضور','یافت','پیرو','جناب','فرمائید','شماره','است','این','تشکر','موضوع','نام','خدا','نام خدا','،',':  موضوع','و احترام','بسمه تعالی','به نام خدا','احترا ما','احترام','آقا','خانم','باشد','از','به','که','در','با','با سلام',' و ','.','(',')','،',':',';','با احترام','+','محترم','','1','2','3','4','5','6','7','8','9','0','?','_','__','/','//','-','؟','.','{}','احترا م
    #stop_words=[']سلام','اً','شرکت','می','های','خواهشمند','فوق','حضور','یافت','پیرو','جناب','فرمائید','شماره','است','این','تشکر','موضوع','نام','خدا','نام خدا','،',':  موضوع','و احترام','بسمه تعالی','به نام خدا','احترا ما','احترام','آقا','خانم','باشد','از','به','که','در','با','با سلام',' و ','.','(',')','،',':',';','با احترام','+','محترم','','1','2','3','4','5','6','7','8','9','0','?','_','__','/','//','-','؟','.','{}','احترا م
    #stop_words=['مس خواهیم',''چیست','وجود','دارد','سلام','اً','شرکت','می','های','خواهشمند','فوق','حضور','یافت','پیرو','جناب','فرمائید','شماره','است','این','تشکر','موضوع','نام','خدا','نام خدا','،',':  موضوع','و احترام','بسمه تعالی','به نام خدا','احترا ما','احترام','آقا','خانم','باشد','از','به','که','در','با','با سلام',' و ','.','(',')','،',':',';','با احترام','+','محترم','','1','2','3','4','5','6','7','8','9','0','?','_','__','/','//','-','؟','.','{}','احترا م']
    #my_normalizer = Normalizer()
    from parsivar import Tokenizer
    my_tokenizer = Tokenizer()
    #m=list(df['سوال'])
    
############################################################################################

    #for i in range(0,len(m)):
                     #deleteWords =stop_words
    #for word in deleteWords:
                      #m[i]= m[i].replace(word,"")
    with open('readme.txt', 'w', encoding="utf-8") as g:
      for line in f:
        g.write(line)
        g.write('\n')
        
    with open('readme.txt', 'r',encoding="utf-8") as file:
       text = file.read()
       
#####################**WordVectorTrain**############################################### 
   
    model1 = fasttext.train_unsupervised('readme.txt', model='skipgram',dim=50,ws=5,epoch=200) 
    wordfreq={}

    #df1=list(df.iloc[:,0])
    df1=list(df['BodyText'])
    for i in range(0,len(df1)):


        words = df1[i].split()
       
   
        unwanted_chars = ".,-_,[1:9] ,،,:,'Body',(and so on),?,1,2,3,4,5,6,7,8,9,۰,۱,۲,۳,۴,۵,۶,۷,۸,۹,'','+',/,\,10"
        wordfreq [i]= {}
        for raw_word in words:
           word = raw_word.strip(unwanted_chars)
           if word not in wordfreq[i]:
                wordfreq[i][word] = 0 
           wordfreq[i][word] += 1 
           
#################################################################################################
        
    all_values=[]
    for i in range(0,len(wordfreq)):
     al_values=[]
     for key, value in wordfreq[i].items():
        
        
        al_values.append(key)  
     all_values.append(al_values)           
        
        

    import statistics
    sentence_vectors=[]
    for i in all_values:
        sent_vector=[]
        c=[]
        for j in i:
      
          sent_vector.append(model1[j])
          c.append(j) 
  
        sentence_vectors.append(sent_vector)         
        
    ave=[]
    for i in range(0,len(sentence_vectors)):
       avg=[]
       sum=0
       for j in range(0,len(sentence_vectors[i])):
    
            avg.append(sentence_vectors[i][j])
       ave.append(avg)         
        
###############*AverageVectors*#######################################       
    l=[]
    for i in range(0,len(ave)):
  
       sum=0
  
       for j in range(0,len(ave[i])):
    
           p=np.array(ave[i][j])
           sum=sum+p
      
       l.append(sum/(len(ave[i])+1))
    #l=[l]  
    ss=[]
    for i in range(0,len(l)):
        if type(l[i])==float:
            #l.pop(i)
            #ss.append(type(l[i]))
            l[i ]= str(l[i])
    S=pd.DataFrame(l)
    
    S.to_csv('D:/Dorosti/100.20python Service/GLOBAL PYTHON SERVICE/similarLettersModels/'+str(CustomerName)+'similarLetterAvgVector')
    
    #S.to_csv('S.csv')
#'D:/Dorosti/100.20python Service/GLOBAL PYTHON SERVICE/similarLettersModels/'+str(CustomerName)+'similarLettermodel'
    
################*KmeansAlgorithm*###################################

    kmeans = KMeans(
        init="random",
        n_clusters=4,
        n_init=10,
        max_iter=2000,
        random_state=42 )


    kmeans.fit(l)

    labels=kmeans.labels_

    a=[]
    for i in range(0,len(labels)):
         a.append(i)
  
  
###############################################################################


    df6 = pd.DataFrame({'Number' : a,'ClusterName' : labels,'UId':GF.iloc[0:len(a),0]}, columns=['Number','ClusterName','UId'])
    #'C:/Users/paya8/Desktop/GLOBAL PYTHON SERVICE/similarLettersModels/'+str(CustomerName)+'.csv'
    df6.to_csv('D:/Dorosti/100.20python Service/GLOBAL PYTHON SERVICE/similarLettersModels/'+str(CustomerName)+'LetterClusterNumber.csv')


    #filename = 'Clustering_model.sav'
    filename='D:/Dorosti/100.20python Service/GLOBAL PYTHON SERVICE/similarLettersModels/'+str(CustomerName)+'LetterClustering_model.sav'
    pickle.dump(kmeans, open(filename, 'wb'))

    #model1.save_model("Word_Vectors.bin")
    model1.save_model('D:/Dorosti/100.20python Service/GLOBAL PYTHON SERVICE/similarLettersModels/'+str(CustomerName)+'LetterWord_Vectors.bin')
    
   
    
    return jsonify({"RESULT":str("Model Dumped")})
    #return "null"

 
 except:
     
     return jsonify({'trace': traceback.format_exc()})
     
      
  
    
    
