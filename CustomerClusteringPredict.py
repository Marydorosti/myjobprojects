# -*- coding: utf-8 -*-
"""
Created on Tue May 24 09:22:41 2022

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

from flask import Blueprint
import pandas as pd
import datetime as dt
CustomerClusteringPredict_bp = Blueprint('CustomerClusteringPredict', __name__)

@CustomerClusteringPredict_bp.route('/CustomerClusteringPredict',methods=['POST'])

def CustomerClusteringPredict():
    
    try:
    
    
        json_=request.json
        json_=pd.DataFrame(json_)
        data=json_
        
        PRESENT = dt.datetime.today()
        
        m = pd.to_datetime(json_.iloc[0,1])
        s=PRESENT-m
        
        #PRESENT = dt.datetime.today()
               #data[['Dt_Effect']]=int(data[['Dt_Effect']])
        data['Dt_Effect'] = pd.to_datetime(data['Dt_Effect'])
               

               #rfm= data.groupby('IdPrsClient').agg({'Dt_Effect': lambda date: (PRESENT - date.max()).days,
                                        #'CountIvc': lambda num: len(num),
                                        #'SumPrice': lambda price: price.sum()})
        rfm= data.groupby('IdPrsClient').agg({'Dt_Effect': lambda date: (PRESENT - date.max()).days,
                                        'CountIvc': lambda num: num.sum(),
                                        'SumPrice': lambda price: price.sum()})
  
  
        rfm.columns=['recency','frequency','monetary']
  
        rfm['recency'] = rfm['recency'].astype(int)

        rfm['r_quartile'] = pd.qcut(rfm['recency'],4, ['1','2','3','4'], duplicates='drop')
               #rfm['f_quartile'] = pd.qcut(rfm['frequency'],4, ['4','3','2','1'], duplicates='drop')
        rfm['f_quartile'] = pd.qcut(rfm['frequency'],4, ['4','3','2'], duplicates='drop')
        rfm['m_quartile'] = pd.qcut(rfm['monetary'],4, ['4','3','2','1'])

               #rfm['RFM_Score'] = rfm.r_quartile.astype(str)+ rfm.f_quartile.astype(str) + rfm.m_quartile.astype(str)
        rfm['RFM_Score'] = rfm.r_quartile.astype(int)+ rfm.f_quartile.astype(int)+ rfm.m_quartile.astype(int)
               
               
               #rfm['RFM_Score'] = rfm.r_quartile.astype(str)+rfm.m_quartile.astype(str)
               #rfm['RFM_Score'] =  rfm.m_quartile.astype(str)
               #p=rfm.head()
               #p=rfm[rfm['RFM_Score']=='121'].sort_values('monetary',ascending=False).head(1)
               #p=rfm[rfm['RFM_Score']]
               #p=[p]
        df=rfm[['RFM_Score']]
                
        quotient_2d = np.array(df).reshape(-1,1)
            
        return quotient_2d,df
        
        
        
        
        
        
        
        
        
        '''rfm= json_.groupby('IdPrsClient').agg({'Dt_Effect': lambda date: PRESENT - json_.iloc[0,1]).days,
                                        'CountIvc': lambda num: len(num),
                                        'SumPrice': lambda price: price.sum()})
        
        
        
        
               #data[['Dt_Effect']]=int(data[['Dt_Effect']])
        #json_['Dt_Effect'] = pd.to_datetime(json_['Dt_Effect'])
        #json_['Dt_Effect'] = PRESENT - json_['Dt_Effect']
        #s=PRESENT - json_['Dt_Effect']
        #s=s.days
        #df=rfm[['RFM_Score']]
                
        quotient_2d = np.array(rfm).reshape(-1,1)
        CustomerName="maryam"
        
        filename ='D:/Dorosti/100.20python Service/GLOBAL PYTHON SERVICE/KmeansMosels/'+str(CustomerName)+'CusromerClustering_model.sav'
        kmeans_model = pickle.load(open(filename, 'rb'))
        f=kmeans_model.predict(quotient_2d)'''
        
        return(str(s.days))
        
        
    except:
        

             return jsonify({'trace': traceback.format_exc()})
        
        
        
        
        
    