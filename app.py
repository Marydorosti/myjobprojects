# -*- coding: utf-8 -*-
"""
Created on Wed Oct  6 11:57:29 2021

@author: m.dorosti
"""
import numpy as np
from flask import Flask
from Apriory import Apriory_bp
from kmeansApp import kmeansApp_bp
from SimilarLettersTrainApi import SimilarLettersTrainApi_bp
from SimilarLettersPredictApi import SimilarLettersPredictApi_bp
from letter_recommendation_train import letter_recommendation_train_bp
from letter_recommendation_predict import letter_recommendation_predict_bp
from TrainChekAPI import TrainChekAPI_bp
from Predict1CheckApi import Predict1CheckApi_bp
from Predict2CheckApi import  Predict2CheckApi_bp
from main_words import main_words_bp
from forecastingTrainAPI import forecastingTrainAPI_bp
from forecastingPredict import forecastingPredict_bp
from Main_WordsPredict import Main_WordsPredict_bp
from SQClusteringService import SQClusteringService_bp
from SQPredictService import SQPredictService_bp
from similarletterClusteringPredict import similarletterClusteringPredict_bp
from similarlettersClustering import similarlettersClustering_bp
from KeyWordsYake import KeyWordsYake_bp
from CustomerClusteringPredict import CustomerClusteringPredict_bp
from PredictAssociation import Predict_Association_bp




app = Flask(__name__)

app.register_blueprint(Apriory_bp, url_prefix='/Apriory')
app.register_blueprint(Predict_Association_bp,url_prefix='/Predict_Association')
app.register_blueprint(SimilarLettersTrainApi_bp, url_prefix='/SimilarLettersTrainApi')
app.register_blueprint(SimilarLettersPredictApi_bp, url_prefix='/SimilarLettersPredictApi')
app.register_blueprint(kmeansApp_bp, url_prefix='/kmeansApp')
app.register_blueprint(letter_recommendation_train_bp, url_prefix='/letter_recommendation_train')
app.register_blueprint(letter_recommendation_predict_bp, url_prefix='/letter_recommendation_predict')
app.register_blueprint(TrainChekAPI_bp, url_prefix='/TrainChekAPI')
app.register_blueprint(Predict1CheckApi_bp, url_prefix='/Predict1CheckApi')
app.register_blueprint(Predict2CheckApi_bp, url_prefix='/Predict2CheckApi')
app.register_blueprint(main_words_bp, url_prefix='/main_words')
app.register_blueprint(forecastingTrainAPI_bp, url_prefix='/forecastingTrainAPI')
app.register_blueprint(forecastingPredict_bp, url_prefix='/forecastingPredict')
app.register_blueprint(Main_WordsPredict_bp, url_prefix='/Main_WordsPredict')
#app.register_blueprint(similarletterClusteringPredict_bp,url_prefix='/similarletterClusteringPredict')
app.register_blueprint(similarlettersClustering_bp,url_prefix='/similarlettersClustering')
app.register_blueprint(similarletterClusteringPredict_bp,url_prefix='/similarletterClusteringPredict')
app.register_blueprint(SQPredictService_bp,url_prefix='/SQPredictService')
app.register_blueprint(SQClusteringService_bp,url_prefix='/SQClusteringService')
app.register_blueprint(KeyWordsYake_bp,url_prefix='/KeyWordsYake')
app.register_blueprint(CustomerClusteringPredict_bp,url_prefix='/CustomerClusteringPredict')

 
port = 18888 # If you don't provide any port the port will be set to 18888
#from waitress import serve
#serve(app,host=,port=)
app.run(port=port, debug=True)



























