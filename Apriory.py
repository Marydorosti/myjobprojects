# -*- coding: utf-8 -*-
"""
Created on Wed Aug  4 14:36:17 2021

@author: m.dorosti
"""

from flask import Flask, request, jsonify
import joblib
import traceback
import pandas as pd
from flask import Flask, request, render_template, session, redirect
import pyodbc
import numpy as np
import heapq
#from apyori import apriori
from flask import Blueprint
Apriory_bp = Blueprint('Apriory', __name__)
#app = Flask(__name__)


def preprocess_dataset(json_):
    #dataset=pd.DataFrame(json_)
    #*************************Read Data And Create Transactions*************************************
    dataset=json_
    
    dataset=dataset[['IdIvcHdr','IdGds']]
    unique_arr = dataset["IdIvcHdr"].unique()
    dataset= dataset.groupby('IdIvcHdr')
    #l=dataset.get_group(101)
    #l.to_csv('D:/Dorosti/100.20python Service/GLOBAL PYTHON SERVICE/AprioryModels/l.csv')

    
    a=[]
    
    for i in unique_arr:
        a.append(dataset.get_group(i)['IdGds'])
   
      #a.append(dataset[dataset['IdIvcHdr']==i]['IdGds'])
      #columns=['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20']
      
    dataset=pd.DataFrame(a,index=unique_arr)
      #dataset=pd.DataFrame(a,columns=None)
      
    number_records=len(dataset)
    #j=dataset
    #j.to_csv('D:/Dorosti/100.20python Service/GLOBAL PYTHON SERVICE/AprioryModels/j.csv')
    return dataset,number_records, unique_arr

@Apriory_bp.route('/Apriory',methods=['POST'])

#@app.route('/Apriori',  methods = ['POST', 'GET'])
def Apriory():
    if request.method == 'GET':
        #return f"The URL /predict is accessed directly. Try going to '/form' to submit form"
        return render_template('form.html')
        #return render_template('form.html')
    if request.method == 'POST':
         try:
             
             #***********Connect to sql server****************************************
             conn=pyodbc.connect('Driver={Sql Server};'
                                   'Server=192.168.100.17\\SQL2019;'
                                   'Database=Modern_Master;'
                                  'UID=sa;'
                                   'PWD=PAYA+master;'
                                   )
               
               
             with open('D://Dorosti//100.20python Service//GLOBAL PYTHON SERVICE//AprioryModels//AprioryQuery.txt' ,'r')as file:
                   query=file.read()
                   
                   
             cursor = conn.cursor()   

             import pandas as pd
             GF=pd.read_sql_query(query,conn) 
             GF=GF[0:5000]
                  
               
             json_=GF
             CustomerName="maryam"
             dataset,number_records, unique_arr=preprocess_dataset(json_)
             f=len(dataset.columns)
             transactions = []#
             
             for i in range(0, number_records):
    
                  transactions.append([str(dataset.values[i,j]) for j in range(0,f) if str(dataset.values[i,j]) != 'nan' ])
             
             
             from apyori import apriori
             Association_Rules = apriori(transactions=transactions, 
             min_support = 0.05,
             min_confidence = 0.5,
             min_lift = 2,
             min_length = 2,max_length=15)
             

             Results = list(Association_Rules)
             d=pd.DataFrame(transactions)
             d.to_csv('D:/Dorosti/100.20python Service/GLOBAL PYTHON SERVICE/AprioryModels/'+str(CustomerName)+'transactions.csv')
             
             def inspect(output):
                 
                 
                 '''z=[]
                 for i in range (0,len(Results)):
                   b=[]
                   for j in range (0,len(Results[i][2])):
                     d=Results[i][2][j][3]
                     b.append(d)
                   b=np.array(b)   
                 j=heapq.nlargest(2, range(len(b)), b.take)
                 z.append(j)'''
                 
                 '''z=[]
                 for i in range (0,len(output)):
                      b=[]
                      for j in range (0,len(output[i][2])):
                           d=output[i][2][j][3]
                           b.append(d)
                           
                      b=np.array(b)     
                      j=heapq.nlargest(2, range(len(b)), b.take)
                      z.append(j)'''     
                     
                 
                      #z.append(np.argmax(b))
                 
                 
                 
                 #for i in range(0,len(z)):
                 #for i in range(0,len(z)): 
                 #for i in z:
                     
                 
                 #for resul in output:
                     #for j in range (0,len(resul)-1):
                         #l=resul[2][j][3]
                     #d.append(l)
                 
                 #f=[tuple(result[2][j])[3]for result in output  for j in range(0,len(result)-1)]
                 #b=np.argmax(d)
                 #for i in range (0,len(z)):
                     #for s in z[i]:
                         
                 lhs=[] 
                 rhs=[]
                 support=[]
                 confedence=[]
                 lift=[]
                 for i in range(0,len(output)):
                     for j in range (0,len(output[i][2])):
                 
                       #lhs=[tuple(output[i][2][j])[0]]
                       lhs.append(list(output[i][2][j][0]))
                       #lhs.append( x for x in tuple(output[i][2][j])[0])
                       
                       
                 
                 
                 
                       #rhs.append(tuple(output[i][2][j])[1])
                       rhs.append(list(output[i][2][j][1]))
                       
                       support.append(output[i][1])
                 
                       confedence.append(output[i][2][j][2] )
                
                       lift.append(output[i][2][j][3] )
                 
                 return list(zip(lhs,rhs,support,confedence,lift))
                 
             output_dataframe=pd.DataFrame(inspect( Results ),columns=['Left_Hand_Side','Right_Hand_Side','Support','Confident','Lift'],dtype="int")
             output_dataframe=output_dataframe.sort_values('Lift',ascending=False)
             #output_dataframe['Left_Hand_Side']=output_dataframe['Left_Hand_Side'].astype(int)
             #output_dataframe['Right_Hand_Side']=output_dataframe['Right_Hand_Side'].astype(int)
             output_dataframe.to_csv('D:/Dorosti/100.20python Service/GLOBAL PYTHON SERVICE/AprioryModels/'+str(CustomerName)+'output.csv')
             #r=type(output_dataframe.iloc[0:0])
             m=[]
             j=[]
             
             for item in Results:

                 # first index of the inner list
                 # Contains base item and add item
                 pair = item[0]


                 pair=tuple(pair)
                 items = [x for x in pair]
                 rule=items[0]+"---->"+"".join(items[1:])
                 a=items
                 b="Support: " + str(item[1])

                 #third index of the list located at 0th
                 #of the third index of the inner list

                 c="Confidence:" + str(item[2][0][2])
                 d="Lift: " + str(item[2][0][3])
                 e="====================================="
                 m.append(rule)
                 #m.append(a)
                 #m.append(b)
                 #m.append(c)
                 #m.append(d)
                # m.append(e)
                 j.append(item)
             v1=list(output_dataframe.iloc[:,0])
             v2=list(output_dataframe.iloc[:,1])
             m=pd.DataFrame(m)
             m.to_csv('D:/Dorosti/100.20python Service/GLOBAL PYTHON SERVICE/AprioryModels/'+str(CustomerName)+'m.csv')
             v1=list(output_dataframe.iloc[:,0])
             v2=list(output_dataframe.iloc[:,1])
             confident=list(output_dataframe.iloc[:,3])
             lift=list(output_dataframe.iloc[:,4])
             
             conn = pyodbc.connect('Driver={SQL Server};Server=192.168.100.17\sql2019;Database=Modern_Master;uid=sa;pwd=PAYA+master')
             cursor = conn.cursor()
             cursor.execute("DELETE FROM  bi.GdsRuleId  ;")
             cursor.execute("DELETE FROM  dbo.[bi.GdsRuleDetail1]  ;")
             cursor.execute("DELETE FROM  dbo.[bi.GdsRuleDetail2]  ;")
             
             conn.commit()
             for i in range (0,len(v1)):
                  cursor.execute("INSERT INTO  bi.GdsRuleId (GdsRuleId,lift,confident)  VALUES (?,?,?);"  ,i,float(lift[i]),float(confident[i]))
                  conn.commit()
                 
             
             for i in range(0,len(v1)):
                 for j in range(0,len(v1[i])):
                     
                    
           
             
             
                       cursor.execute("INSERT INTO  dbo.[bi.GdsRuleDetail1] (IdHdr,IdGds)  VALUES (?,?);"  ,float(i),float(v1[i][j]))
                       conn.commit()
             for i in range(0,len(v2)):
                 for j in range(0,len(v2[i])): 
                     
                     
                       cursor.execute("INSERT INTO  dbo.[bi.GdsRuleDetail2] (IdHdr,IdGds)  VALUES (?,?);"  ,float(i),float(v2[i][j]))
                       conn.commit()
            
             
             return str(len(Results))
             #return ""
         except:
             
             

            return jsonify({'trace': traceback.format_exc()})
    else:
        print ('Train the model first')
        return ('No model here to use')

