# -*- coding: utf-8 -*-
"""
Created on Tue May 31 14:38:33 2022

@author: m.dorosti
"""

from flask import Flask, request, jsonify
import joblib
import traceback
import pandas as pd
#import numpy as np
#import sys
#import json
#from flask import Flask, request, render_template, session, redirect
#from sklearn.cluster import KMeans
#import pyodbc
#import pickle
#from collections import Counter



from flask import Blueprint
Predict_Association_bp = Blueprint('Predict_Association', __name__)

@Predict_Association_bp.route('/Predict_Association',methods=['POST'])
#@app.route('/predict', methods=['POST'])
def Predict_Association():
    
    try:
        
        
        
       
        
        #************Read executed rules from excel file****************************************
        
        
        CustomerName="maryam"
        df=pd.read_csv('D:/Dorosti/100.20python Service/GLOBAL PYTHON SERVICE/AprioryModels/'+str(CustomerName)+'output.csv')
       
       
       
        
       
        
        
        
        def match(a,b):
            if len(a)==0 or len(b)==0:
                return[]
            if a[0]==b[0]:
                return [a[0]]+match(a[1:],b[1:])
            return max(match(a,b[1:]),match(a[1:],b),key=len)
        
       
        m=request.json
        
        h=[]
       
      
        #**********Select right hand side that have max intersection and min difrence of lenght*********************
        for i in range(0,len(df)):
           
            p=abs(len(set(eval(df.iloc[i,1])))-len(m))-(len(list(set(m)&set(eval(df.iloc[i,1])))))
            h.append(p)
            
        max_value=min(h) 
        i=h.index(max_value)
        if len(list(set(m)&set(eval(df.iloc[i,1])))) !=0:
           recommended_list=df.iloc[i,2]
        else:
            recommended_list=''
            
        
          
        
        
        
        '''s=[]
        #for i,product in enumerate(sorted_rules["Left_Hand_Side"]):
        for i,product in enumerate(df.iloc[:,1]):
            #s.appned(i)
            #intersection_list=list(set.intersection(set(product_id),set(product)))
            s.append(product)
            for j in list(product):
            #if len(intersection_list)!=0:
                
              if j == product_id:
                  
                 recommendation_list.append(list(sorted_rules.iloc[i]["Right_Hand_Side"]))'''
                 
        #return(str(recommended_list))
        return(str(h))
        
        
        
        '''for i in range (0,len(df)):
            
            if len(df.iloc[i,1].intersection(json_))==len(json_):
                   z.append(df.iloc[i,2])
            #else:
            elif len(df.iloc[i,1].intersection(json_))==len(json_)-1:
                   z.append(df.iloc[i,2])
            elif len(df.iloc[i,1].intersection(json_))==len(json_)-2:
                   z.append(df.iloc[i,2])
            elif len(df.iloc[i,1].intersection(json_))==len(json_)-3:
                   z.append(df.iloc[i,2])
                   
             
        s=df.iloc[5,2]      
                
            
        #if len(json_)
        
        return(str(s))'''
      
        
    except:
        
         return jsonify({'trace': traceback.format_exc()})



