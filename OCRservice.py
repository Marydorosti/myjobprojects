# -*- coding: utf-8 -*-
"""
Created on Tue Aug 16 09:47:26 2022

@author: m.dorosti
"""

from flask import Flask, request, jsonify
import traceback
import pandas as pd
import numpy as np
import flask
import sys
import base64
#import cv2
from pytesseract import pytesseract
from PIL import Image
#app=Flask(__name__)
from flask import Blueprint
import json
import io
#from io import String


OCRService_bp = Blueprint('OCRService_bp', __name__)

with open('PythonPathConfig.txt', 'r', encoding="utf-8") as file:   

        path=file.read()






@OCRService_bp.route('/OCRService',methods=['POST'])
#@app.route('/ocr', methods=['POST','GET'])
def OCRService():
   try: 
      
      myjson=request.json
      pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'
      imagefile=myjson['Image']
      OCRLang=myjson['OCRLang']
      im=Image.open(io.BytesIO(base64.b64decode(imagefile)))
      if (OCRLang==1):
          text = pytesseract.image_to_string(im,lang="fas")
         
      elif(OCRLang==0):
          text = pytesseract.image_to_string(im,lang="fas+eng")
          
      #im=Image.open(io.BytesIO(base64.b64decode(imagefile)))
      #text = pytesseract.image_to_string(im,lang="fas")
      text=text.encode('utf-8')
      text=text.decode('UTF-8')

      #return jsonify({"RESULT":str(text)})
      return (str(text))
      #return jsonify({str(text)})
 
 
   except:
     
     return jsonify({'trace': traceback.format_exc()})
 

    
 
     
      