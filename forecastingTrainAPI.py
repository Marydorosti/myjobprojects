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
#from flask import Flask, request, jsonify
#import joblib
import traceback
import pandas as pd
import numpy as np
#import sys
#import json
#from flask import Flask, request, render_template, session, redirect

from flask import Blueprint
forecastingTrainAPI_bp = Blueprint('forecastingTrainAPI', __name__)
#app = Flask(__name__,template_folder='template')

def preprocess_data(json_,CustomerName,forecast_out):
      train1 = pd.DataFrame(json_)
      #import pandas as pd
      
      
      from sklearn.ensemble import IsolationForest

      #CustomerName = input("Enter customer name: ")
      #forecast_out=input("PleaseEnterDAYS:")
      #forecast_out=10






      #train1=pd.read_excel(str(CustomerName)+'.xlsx')
      df=train1[['PMonth','PSession','PDay','SalesAmount','InvoicesCount','GdsCount','SalesTotalPrice']]
      train=df
      def anomaly_detect(df):

         
# Assume that 13% of the entire data set are anomalies
 
         outliers_fraction = 0.005
         isolationforest =  IsolationForest(contamination=outliers_fraction)
    
         isolationforest.fit(df.values)
         df4=isolationforest.predict(df.values)
         df4=pd.DataFrame(df4)
         df['anomaly'] = pd.Series(isolationforest.predict(df.values))
  # visualization
         df['anomaly'] = pd.Series(df['anomaly'].values, index=df.index)
         a = df.loc[df['anomaly'] == -1] #anomaly
         
         return df4,df

    
    
    
    
    
    
    
       
      df4,df=anomaly_detect(train[['SalesTotalPrice']]) 

    
      ff=np.where(df4[0]==-1)

      row = train1.iloc[np.where(df4[0]==-1)] #
      F=print('number of anomalies is:',len(row),row)

      ff=list(ff)
      for i in ff:
          train=train.drop(index=i)



      print(len(train))

      from sklearn import preprocessing
      from sklearn.preprocessing import MinMaxScaler
      sc = MinMaxScaler(feature_range = (0, 1))
    
    
     # import numpy as np

      y = train.iloc[:,6]
 
      x = train.iloc[:,0:6]
    
      x['Prediction']= y.shift(-forecast_out)
      X = np.array(x.drop(['Prediction'],1)) #delete prediction column and convert to numpy array


   
      X = X[:-forecast_out] #all but the last n row




      Y = np.array(x['Prediction'])

      Y = Y[:-forecast_out]
      X_ = sc.fit_transform(X)
    
      Y=np.array(Y).reshape(-1,1)
      Y_=sc.fit_transform(Y)
      y = train.iloc[:,6]
      mino=y.min()
      print(mino)
      maxo=y.max()
      print(maxo)
      A=[mino,maxo]
      textfile = open('C:/Users/paya8/Desktop/GLOBAL PYTHON SERVICE/ForecastingModels/'+str(CustomerName)+"MIN&MAX.txt", "w")
      for element in A:
        textfile. write(str(element) + "\n")

#X_,Y_=data_preprocess(train,10)   
      from sklearn import preprocessing
      from sklearn.preprocessing import MinMaxScaler
      #from xgboost import XGBRegressor
      #import joblib
      #XG = XGBRegressor(n_estimators=300)
      from sklearn.model_selection import train_test_split
    
      x_train, x_test, y_train, y_test = train_test_split(X_, Y_, test_size=0.2,random_state=40) 
      return  x_train, x_test, y_train, y_test,F
    #tr=DecisionTreeRegressor(max_features= 'auto',random_state=40,max_depth=3,criterion='mae')

@forecastingTrainAPI_bp.route('/Train',methods=['POST'])
#@app.route('/train', methods=['POST'])
def Train():
    #if model:
        try :
               #json_ = request.json
               
               #CustomerName = request.args['CustomerName']
               #CustomerName = request.args.get('CustomerName')
               #forecast_out = request.args['forecast_out']
               #forecast_out = request.args.get('forecast_out',type=int,default=10)   
               myjson = request.json
               myjson2=pd.DataFrame(request.json)
               #CustomerName=myjson2.CustomerName
               
               json_=myjson['Array']
               CustomerName=myjson['CustomerName']
               forecast_out=10
               import joblib
               #import onnx
               #import onnxmltools
               #import onnxmltools.convert.common.data_types
               from xgboost import XGBRegressor
               #from xgboost import XGBClassifier
               x_train, x_test, y_train, y_test,F=preprocess_data(json_,CustomerName,forecast_out)
               XG = XGBRegressor(n_estimators=300)
               XG.fit(x_train,y_train)
               XG_test_accuracy=XG.score(x_test,y_test)
               XG_train_accuracy=XG.score(x_train,y_train)
               print("XG_train_confidence:",XG_train_accuracy)
               print("XG_test_confidence:", XG_test_accuracy)
               #from skl2onnx import convert_sklearn
               #from skl2onnx.common.data_types import FloatTensorType

    
               from sklearn.ensemble import RandomForestRegressor

               RF = RandomForestRegressor(n_estimators=50, random_state=20)
               RF.fit(x_train,y_train)
               RF_test_accuracy=RF.score(x_test,y_test)
               #joblib.dump(RF, 'RFmodel.pkl')
               #RF_test_accuracy=RF.score(x_test,y_test)
               #return  (str("Model dumped!"),str(F))
 


               if(RF_test_accuracy>XG_test_accuracy) :
    
                   joblib.dump(RF, 'C:/Users/paya8/Desktop/GLOBAL PYTHON SERVICE/ForecastingModels/'+str(CustomerName)+'ForecastModel.pkl')
                   #print("Model dumped!")
                   #num_features = 6
                  # initial_type = [('feature_input', FloatTensorType([None, num_features]))]

# Convert the trained model to an ONNX format.
                  # onx = convert_sklearn(RF, initial_types=initial_type)
                  # with open(str(CustomerName)+"RF.onnx", "wb") as f:
                     # f.write(onx.SerializeToString())
               else:
                   joblib.dump(XG,'C:/Users/paya8/Desktop/GLOBAL PYTHON SERVICE/ForecastingModels/'+ str(CustomerName)+'ForecastModel.pkl')
                  # print("Model dumped!") 
                  # num_features = 6
                  # initial_type = [('feature_input', FloatTensorType([1, num_features]))]

# Convert the trained model to an ONNX format.
                  # onx = onnxmltools.convert.convert_xgboost(XG, initial_types=initial_type)
    #onx = convert_sklearn(XG, initial_types=initial_type)
                  # with open(str(CustomerName)+"XG.onnx", "wb") as f:
                     # f.write(onx.SerializeToString())
    
               
               #z=[]
               #z.append(( XG_test_accuracy, RF_test_accuracy))
               #XG_test_accuracy
               
               #return  (str("Model dumped!"),str(Z))
               return jsonify({"RESULT": str("Model dumped!")})
               #return jsonify({"RESULT": str(z)})

               
               
               
               
               
              

               
               
        except:

               return jsonify({'trace': traceback.format_exc()})
    #else:
       # print ('Train the model first')
       # return ('No model here to use')

'''if __name__ == '__main__':
    try:
        port = int(sys.argv[1]) # This is for a command-line input
    except:
        port = 15944 # If you don't provide any port the port will be set to 12345

    
    #model = joblib.load('xgb.pkl')
    #model = joblib.load('xgb2 .pkl')
    

    app.run(port=port, debug=True)'''

               
               
               

