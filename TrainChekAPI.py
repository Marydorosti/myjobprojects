# -*- coding: utf-8 -*-
"""
Created on Mon Jul  5 11:51:12 2021

@author: m.dorosti
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Jul  4 16:45:14 2021

@author: m.dorosti
"""

from flask import Flask, request, jsonify
import joblib
import traceback
import pandas as pd
import numpy as np
import sys
import json
import pyodbc
#from flask import Flask, request, render_template, session, redirect


from flask import Blueprint
TrainChekAPI_bp = Blueprint('TrainChekAPI', __name__)




#app = Flask(__name__,template_folder='template')

def preprocess_data(json_,CustomerName):
      train = pd.DataFrame(json_)
      train.astype('float64').dtypes
      from sklearn.impute import SimpleImputer
      
     
      
      mf=train.drop(columns=['IsDishonor'])
      df=train.drop(columns=['IsDeposit'])
      
      df=df.iloc[:,3:]
     
      mf=mf.iloc[:,3:]
      df.fillna(df.mean(), inplace=True)
      
      mf.fillna(mf.mean(), inplace=True)
      
      
      df=(df-df.min())/(df.max()-df.min())
      mf=(mf-mf.min())/(mf.max()-mf.min())
      sd=df.drop(columns=['IsDishonor'])
      fd=mf.drop(columns=['IsDeposit'])
      
      # Create correlation matrix
      corr_matrix = sd.corr().abs()
      corr_matrix1 = fd.corr().abs()

     # Select upper triangle of correlation matrix
      upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
      upper1 = corr_matrix1.where(np.triu(np.ones(corr_matrix1.shape), k=1).astype(np.bool))

    # Find index of feature columns with correlation greater than 0.70
      to_drop = [column for column in upper.columns if any(upper[column] > 0.70)]
      to_drop1 = [column for column in upper1.columns if any(upper1[column] > 0.70)]
      
    # Drop features 
      df1=df.drop(df[to_drop], axis=1)
      #t=df.iloc[:,3:]
      mf1=mf.drop(mf[to_drop1], axis=1)
      
      #np.where(corr.DueDays>0.75)
      df1=df1.drop(columns=['PrsType'])
      #np.where(corr.DueDays>0.75)
      mf1=mf1.drop(columns=['PrsType'])
      m=len(df1.columns)
      s=len(mf1.columns)
      j=df1.columns.tolist()
      k=mf1.columns.tolist()
      H=df1.iloc[:,0:m-1].columns
      B=mf1.iloc[:,0:s-1].columns
      #textfile = open(str(CustomerName)+"1removedFeatures.txt", "w")
      #'D:/Dorosti/100.20python Service/GLOBAL PYTHON SERVICE/AprioryModels/'+str(CustomerName)+'output.csv'
      textfile = open('D:/Dorosti/100.20python Service/GLOBAL PYTHON SERVICE/CheckModels/'+str(CustomerName)+"1bestFeatures.txt", "w")
      #textfile.writelines(to_drop1)
      #for element in to_drop:
      #for element in df1.columns:
      for element in H:
          
        #textfile. writeline(str(element))
        textfile.write(f"{element}\n")
      textfile2 = open('D:/Dorosti/100.20python Service/GLOBAL PYTHON SERVICE/CheckModels/'+str(CustomerName)+"2bestFeatures.txt", "w")
      #textfile = open("2removedFeatures.txt", "w")
      #for element in to_drop1:
      #for element in mf1.columns: 
      for element in B: 
        #textfile. writeline(str(element) )
        textfile2.write(f"{element}\n")
      #N=df1.iloc[:,m-1]
      #M=mf1.iloc[:,s-1]
     #p=df1.iloc[:,0:m-1].columns
      #K=df1.iloc[:,m-1]
      #U=mf1.iloc[:,s-1]
      p=np.array(mf1.iloc[:,0:s-1].values).shape
      
      X1=df1.iloc[:,0:m-1].values
      Y1=df1.iloc[:,m-1].values
      x1=mf1.iloc[:,0:s-1].values
      y1=mf1.iloc[:,s-1].values
      imp = SimpleImputer(missing_values=np.nan, strategy='mean')
      X1=imp.fit_transform(X1)
      x1=imp.fit_transform(x1)
      Y1=imp.fit_transform(Y1.reshape(-1,1))
      y1=imp.fit_transform(y1.reshape(-1,1))
      l=X1.shape
      
      
     
      '''X1=df1.iloc[:,0:m-1]
      Y1=df1.iloc[:,m-1]
      x1=mf1.iloc[:,0:s-1]
      y1=mf1.iloc[:,0:s-1]
      X1 = X1[~pd.isnull(X1)]
      x1 = x1[~pd.isnull(x1)]
      Y1 = Y1[~pd.isnull(Y1)]
      y1 = y1[~pd.isnull(y1)]'''
      
              
      import numpy
# Splitting the dataset into the Training set and Test set
      from sklearn.model_selection import train_test_split
      x_train, x_test, y_train, y_test = train_test_split(X1, Y1, test_size = 0.25)
      x_train1, x_test1, y_train1, y_test1 = train_test_split(x1, y1, test_size = 0.25)
      #x_train1 = x_train1[~numpy.isnan(x_train1)]
      #x_test1 = x_test1[~numpy.isnan(x_test1)]
      #x_test = x_test[~numpy.isnan(x_test)]
      #x_train = x_train[~numpy.isnan(x_train)]
     # x = x[~pd.isnull(x)]
      
      
      
      return x_train,y_train,x_test,y_test,x_train1, x_test1, y_train1, y_test1,to_drop,H,B
      





      
      


     


    

    

      
      
      
      
      
     
      






#@app.route('/form',methods = ['GET'])
#def form():
   # return render_template('form.html')

@TrainChekAPI_bp.route('/TrainChekAPI', methods = ['POST'])

#@app.route('/train', methods = ['POST', 'GET'])
def TrainChekAPI():
    #if request.method == 'GET':
        #return f"The URL /predict is accessed directly. Try going to '/form' to submit form"
        #return render_template('form.html')
        #return render_template('form.html')
    if request.method == 'POST':
    #if model:
        try :
            
            
               '''ConnectionString=myjson['ConnectionString']
               #json_=request.json
               CustomerName=myjson['CustomerName']
               #CustomerName="PAYA2"
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
                                   )'''
               
               
               conn=pyodbc.connect('Driver={Sql Server};'
                                   'Server=192.168.100.17\\SQL2019;'
                                   'Database=Modern_Master;'
                                  'UID=sa;'
                                   'PWD=PAYA+master;'
                                   )
               
               
               with open('D://Dorosti//100.20python Service//GLOBAL PYTHON SERVICE//CheckModels//ChekQuery2.txt' ,'r')as file:
                   query=file.read()
                   
                   
               cursor = conn.cursor() 
               '''while cursor.nextset():
                   try:
                       results=cursor.fetchall()
                       break
                   except pyodbc.ProgrammingError:
                       continue'''
               
               
               
               
               #cursor.execute(query)
               #cursor.execute(query)
               #raw=cursor.fetchall()
               

               import pandas as pd
               GF=pd.read_sql_query(query,conn)
               #GF=cursor.fetchall()
               json_=GF
            
            
            
            
            
            
            
            
            
               #json_ = request.json 
               #CustomerName = request.form.get("CustomerName")
               #json_ = request.form.get("json_")
               #json_ = request.json
               #myjson = request.json
               #myjson2=pd.DataFrame(request.json)
               #CustomerName=myjson2.CustomerName
               
               #json_=myjson['Array']
               #json_=request.json
               CustomerName="Aramesh"
               
               
               #CustomerName=myjson['CustomerName']
               
               
               #CustomerName = request.args['CustomerName']
               
               #CustomerName = request.args.get('CustomerName')
               
               #json_ = json_.json
               
               import joblib
               x_train,y_train,x_test,y_test,x_train1, x_test1, y_train1, y_test1,to_drop,H,B=preprocess_data(json_,CustomerName)
               #CustomerName=input("please enter CustomerName")
               q=x_train1.shape
               w=x_train.shape
               
              
               from sklearn.linear_model import LogisticRegression
               LR = LogisticRegression()
               LR.fit(x_train, y_train)
               LR1 = LogisticRegression()
               LR1.fit(x_train1, y_train1)
              
 
               # Predicting the Test set results
               #y_pred = LR.predict(x_test)
               #y_pred1 = LR.predict(x_test1)
               from sklearn.metrics import accuracy_score
               LR_test_accuracy=LR.score(x_test,y_test)
               LR1_test_accuracy=LR1.score(x_test1,y_test1)
               #print('accuracy on test data:',confidence)
 
              # RF_train_accuracy=RF.score(x_train,y_train)
               from sklearn.ensemble import GradientBoostingClassifier
               
               XG = GradientBoostingClassifier(n_estimators=150, learning_rate=1.0,
                         max_depth=5, random_state=0)
#>>> clf.score(X_test, y_test)

               XG.fit(x_train, y_train)
               XG1 = GradientBoostingClassifier(n_estimators=150, learning_rate=1.0,
                         max_depth=5, random_state=0)
#>>> clf.score(X_test, y_test)

               XG1.fit(x_train1, y_train1)
 
# Predicting the Test set results
               #y_pred = XG.predict(x_test)
               XG_test_accuracy = XG.score(x_test, y_test)
               XG1_test_accuracy = XG1.score(x_test1, y_test1)
              
              
              
              
 


               if(LR_test_accuracy>XG_test_accuracy) :
    
                   joblib.dump(LR, 'D:/Dorosti/100.20python Service/GLOBAL PYTHON SERVICE/CheckModels/'+str(CustomerName)+'1model.pkl')
                   # Convert into ONNX format
                   from skl2onnx import convert_sklearn
                   from skl2onnx.common.data_types import FloatTensorType
                   initial_type = [('float_input', FloatTensorType([None, len(H)]))]
                   onx = convert_sklearn(LR, initial_types=initial_type)
                   with open('D:/Dorosti/100.20python Service/GLOBAL PYTHON SERVICE/CheckModels/'+str(CustomerName)+"1model.onnx", "wb") as f:
                        f.write(onx.SerializeToString())

                   #joblib.dump(LR, '1model.pkl')
                   
               else:
                   joblib.dump(XG,'D:/Dorosti/100.20python Service/GLOBAL PYTHON SERVICE/CheckModels/'+str(CustomerName) +'1model.pkl')
                   #joblib.dump(XG,'1model.pkl')
                   from skl2onnx import convert_sklearn
                   from skl2onnx.common.data_types import FloatTensorType
                   initial_type = [('float_input', FloatTensorType([None, len(H)]))]
                   onx = convert_sklearn(XG, initial_types=initial_type)
                   with open('D:/Dorosti/100.20python Service/GLOBAL PYTHON SERVICE/CheckModels/'+str(CustomerName)+"1model.onnx", "wb") as f:
                        f.write(onx.SerializeToString())
               
               if(LR1_test_accuracy>XG1_test_accuracy) :
    
                   joblib.dump(LR1,'D:/Dorosti/100.20python Service/GLOBAL PYTHON SERVICE/CheckModels/'+ str(CustomerName)+'2model.pkl')
                   from skl2onnx import convert_sklearn
                   from skl2onnx.common.data_types import FloatTensorType
                   initial_type = [('float_input', FloatTensorType([None, len(B)]))]
                   onx = convert_sklearn(LR1, initial_types=initial_type)
                   with open('D:/Dorosti/100.20python Service/GLOBAL PYTHON SERVICE/CheckModels/'+str(CustomerName)+"2model.onnx", "wb") as f:
                        f.write(onx.SerializeToString())
                   #joblib.dump(LR1, '2model.pkl')
                   
               else:
                   joblib.dump(XG1,'D:/Dorosti/100.20python Service/GLOBAL PYTHON SERVICE/CheckModels/'+ str(CustomerName)+'2model.pkl')
                   #joblib.dump(XG1, '2model.pkl')
                   joblib.dump(XG1,'D:/Dorosti/100.20python Service/GLOBAL PYTHON SERVICE/CheckModels/'+ str(CustomerName)+'2model.pkl')
                   from skl2onnx import convert_sklearn
                   from skl2onnx.common.data_types import FloatTensorType
                   initial_type = [('float_input', FloatTensorType([None, len(B)]))]
                   onx = convert_sklearn(LR1, initial_types=initial_type)
                   with open('D:/Dorosti/100.20python Service/GLOBAL PYTHON SERVICE/CheckModels/'+str(CustomerName)+"2model.onnx", "wb") as f:
                        f.write(onx.SerializeToString())
                
                
                   

               
            
               #return  '{} {} '.format(K,U)
               
               return  (str((LR1_test_accuracy,XG1_test_accuracy)))
               #return(str(GF))
               #return jsonify({'RESULT': str("Model dumped!")})
               #return(str(s))
               #return(str(df1))
               #return render_template('simple.html.html',  tables=[cluster_map.to_html(classes='data')], titles=df.columns.values)

               
               
               
               
               
              

               
               
        except:

               return jsonify({'trace': traceback.format_exc()})
    #else:
       # print ('Train the model first')
       # return ('No model here to use')

'''if __name__ == '__main__':
    try:
        port = int(sys.argv[1]) # This is for a command-line input
    except:
        port = 16944 # If you don't provide any port the port will be set to 12345

    
    #model = joblib.load('xgb.pkl')
    #model = joblib.load('xgb2 .pkl')
    

    app.run(port=port, debug=True)'''

               
               
               

