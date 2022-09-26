# -*- coding: utf-8 -*-
"""
Created on Sat Jul 23 12:32:39 2022

@author: m.dorosti
"""
import pyodbc
import pandas as pd
import re


conn=pyodbc.connect('Driver={Sql Server};'
                                   'Server=192.168.100.17\\SQL2019;'
                                   'Database=AsrJadidData;'
                                   'UID=sa;'
                                   'PWD=PAYA+master;'
                                   )
    
     
with open('D://Dorosti//PayaSoft.BI//PythonCodes//similarLettersModels//SearchEngineQuery.txt' ,'r')as file:
                   query=file.read()
                  
cursor = conn.cursor()  
cursor.close()      
GF=pd.read_sql_query(query,conn)
df=pd.DataFrame(GF)
df.dropna()
#print(df.head())
#f=list(df[['DscGds']])
#f=str(df[['DscGds']])
f=list(df['DscGds'])
#f=list(df)
#patterns=f
#text=[مانتو]*
#for pattern in patterns:
    #if re.findall(pattern, text):
        #print(pattern)
'''match=re.findall(text,patterns)
if match:
    print(match)'''

#print(patterns)
'''print(len(df["DscGds"].unique()))
#print(f)

finds1=re.findall(r'مانتو',f)
finds2=re.findall(r'[\w\.-]* بوت [\w\.-]*',f)

finds2=re.findall(r'*بوت*',f)
finds2=re.findall(r'[\w\.-]+آرشال[\w\.-]+',f)
#for word in finds2:
print(finds2)
'''
out=[]
out2=[]
for i in f:

   #finds=re.findall(r"بوت",i)
   #if finds:
      #out.append(i)
   finds2=re.findall(r'[\s]+بوت[\s]+',i)
   if finds2:
      out.append(i)
   regex= r' بوت'  
   finds=re.findall(r'[\s^]بوت',i)
   finds=re.search(regex,i)
   if finds:
      out2.append(i)
      #print(i)
   #if finds1:
f= out2+out
s=set(out2)-set(out)
s=list(s)
p=out+s

      
print(p)
#print(df)


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
      

   


#print(f)

