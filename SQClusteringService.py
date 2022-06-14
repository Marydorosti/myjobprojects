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



from flask import Blueprint
SQClusteringService_bp = Blueprint('SQClusteringService', __name__)

@SQClusteringService_bp.route('/SQClusteringService',methods=['POST'])
def SQClusteringService():
 try:   
    
    json_ = request.json
    df=pd.DataFrame(json_)
    #df=pd.read_excel('Porsesh & Pasokh (11) (1).xlsx')
    RANDOM_SEED = 42
    np.random.seed(RANDOM_SEED)
    df=df.iloc[:,5:7]
    #my_normalizer = Normalizer()
    #from parsivar import Tokenizer
    #my_tokenizer = Tokenizer()
    #stop_words=['سلام','اً','شرکت','می','های','خواهشمند','فوق','حضور','یافت','پیرو','جناب','فرمائید','شماره','است','این','تشکر','موضوع','نام','خدا','نام خدا','،',':  موضوع','و احترام','بسمه تعالی','به نام خدا','احترا ما','احترام','آقا','خانم','باشد','از','به','که','در','با','با سلام',' و ','.','(',')','،',':',';','با احترام','+','محترم','','1','2','3','4','5','6','7','8','9','0','?','_','__','/','//','-','؟','.','{}','احترا م']
    #my_normalizer = Normalizer()
    stop_words=['؟','حلی','حل','راه','وجود','دارد','چیست','می خواهیم','سلام','اً','شرکت','می','های','خواهشمند','فوق','حضور','یافت','پیرو','جناب','فرمائید','شماره','است','این','تشکر','موضوع','نام','خدا','نام خدا','،',':  موضوع','و احترام','بسمه تعالی','به نام خدا','احترا ما','احترام','آقا','خانم','باشد','از','به','که','در','با','با سلام',' و ','.','(',')','،',':',';','با احترام','+','محترم','','1','2','3','4','5','6','7','8','9','0','?','_','__','/','//','-','{}','احترا م']
               #my_normalizer = Normalizer()
               #from parsivar import Tokenizer
               #my_tokenizer = Tokenizer()
    #stop_words=[,',',','سلام','اً','شرکت','می','های','خواهشمند','فوق','حضور','یافت','پیرو','جناب','فرمائید','شماره','است','این','تشکر','موضوع','نام','خدا','نام خدا','،',':  موضوع','و احترام','بسمه تعالی','به نام خدا','احترا ما','احترام','آقا','خانم','باشد','از','به','که','در','با','با سلام',' و ','.','(',')','،',':',';','با احترام','+','محترم','','1','2','3','4','5','6','7','8','9','0','?','_','__','/','//','-','؟','.','{}','احترا م']
    #my_normalizer = Normalizer()


    #stop_words=['سلام','اً','شرکت','می','های','خواهشمند','فوق','حضور','یافت','پیرو','جناب','فرمائید','شماره','است','این','تشکر','موضوع','نام','خدا','نام خدا','،',':  موضوع','و احترام','بسمه تعالی','به نام خدا','احترا ما','احترام','آقا','خانم','باشد','از','به','که','در','با','با سلام',' و ','.','(',')','،',':',';','با احترام','+','محترم','','1','2','3','4','5','6','7','8','9','0','?','_','__','/','//','-','؟','.','{}','احترا م
    #stop_words=[']سلام','اً','شرکت','می','های','خواهشمند','فوق','حضور','یافت','پیرو','جناب','فرمائید','شماره','است','این','تشکر','موضوع','نام','خدا','نام خدا','،',':  موضوع','و احترام','بسمه تعالی','به نام خدا','احترا ما','احترام','آقا','خانم','باشد','از','به','که','در','با','با سلام',' و ','.','(',')','،',':',';','با احترام','+','محترم','','1','2','3','4','5','6','7','8','9','0','?','_','__','/','//','-','؟','.','{}','احترا م
    #stop_words=['مس خواهیم',''چیست','وجود','دارد','سلام','اً','شرکت','می','های','خواهشمند','فوق','حضور','یافت','پیرو','جناب','فرمائید','شماره','است','این','تشکر','موضوع','نام','خدا','نام خدا','،',':  موضوع','و احترام','بسمه تعالی','به نام خدا','احترا ما','احترام','آقا','خانم','باشد','از','به','که','در','با','با سلام',' و ','.','(',')','،',':',';','با احترام','+','محترم','','1','2','3','4','5','6','7','8','9','0','?','_','__','/','//','-','؟','.','{}','احترا م']
    #my_normalizer = Normalizer()
    from parsivar import Tokenizer
    my_tokenizer = Tokenizer()
    m=list(df['سوال'])
    
############################################################################################

    for i in range(0,len(m)):
                     deleteWords =stop_words
    for word in deleteWords:
                      m[i]= m[i].replace(word,"")
    with open('readme.txt', 'w', encoding="utf-8") as f:
      for line in m:
        f.write(line)
        f.write('\n')
        
    with open('readme.txt', 'r',encoding="utf-8") as file:
       text = file.read()
       
#####################**WordVectorTrain**############################################### 
   
    model1 = fasttext.train_unsupervised('readme.txt', model='skipgram',dim=50,ws=5,epoch=200) 
    wordfreq={}

    df1=list(df.iloc[:,0])
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
    s=[]
    for i in range(0,len(ave)):
  
       sum=0
  
       for j in range(0,len(ave[i])):
    
           p=np.array(ave[i][j])
           sum=sum+p
       s.append(sum/len(ave[i]))        

    S=pd.DataFrame(s)
    S.to_csv('C:/Users/paya8/Desktop/GLOBAL PYTHON SERVICE/similarQuestionsModels/S.csv')
    
    #S.to_csv('S.csv')

    
################*KmeansAlgorithm*###################################

    kmeans = KMeans(
        init="random",
        n_clusters=4,
        n_init=10,
        max_iter=1000,
        random_state=42 )


    kmeans.fit(s)

    labels=kmeans.labels_

    a=[]
    for i in range(0,len(labels)):
         a.append(i)
  
  
###############################################################################


    df6 = pd.DataFrame({'Number' : a,'ClusterName' : labels }, columns=['Number','ClusterName'])
    #'C:/Users/paya8/Desktop/GLOBAL PYTHON SERVICE/similarLettersModels/'+str(CustomerName)+'.csv'
    df6.to_csv('C:/Users/paya8/Desktop/GLOBAL PYTHON SERVICE/similarQuestionsModels/ClusterNumber.csv')


    #filename = 'Clustering_model.sav'
    filename='C:/Users/paya8/Desktop/GLOBAL PYTHON SERVICE/similarQuestionsModels/Clustering_model.sav'
    pickle.dump(kmeans, open(filename, 'wb'))

    #model1.save_model("Word_Vectors.bin")
    model1.save_model("C:/Users/paya8/Desktop/GLOBAL PYTHON SERVICE/similarQuestionsModels/Word_Vectors.bin")
    return jsonify({"RESULT":str("model dumped!!!")})
 
 except:
     
     return jsonify({'trace': traceback.format_exc()})
     
      
  
    
    
