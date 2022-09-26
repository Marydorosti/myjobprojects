# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 16:06:09 2022

@author: m.dorosti
"""
import re
#A is input word
#B is one of gds list item

def SearchFunction(A,B):
   SCORE=0
    
   
   finds=re.findall(r'[\s]'+str(A)+'[\s]+',B)
   finds2=re.findall(r'[\w\.-]+'+str(A)+' [\w\.-]+',B)
   finds3=re.findall(r'[\w\.-]+'+str(A),B)
   finds4=re.findall(str(A)+r'[\w\.-]+',B)
   finds5=re.findall(r'[\w\.-]*'+str(A)+' [\w\.-]*',B)
   
   
   if finds:
      #out.append(i)
      SCORE=SCORE+3
   elif finds2:
      SCORE=SCORE+2
   elif finds3:
      SCORE=SCORE+1 
   elif finds4:
      SCORE=SCORE+1
   
   return SCORE    
B='وسری طرح بوته جقه طوسی  '
A='بوته'
SCORE=SearchFunction(A, B)
print(SCORE)  
      
