#!/usr/bin/env python
# coding: utf-8

# In[1]:


import face_recognition as fr
import cv2
import numpy as np
import pandas as pd 
import os
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from datetime import datetime
import pickle


# In[2]:


def markAttendance(name,top,right,left,bottom):            
    df = pd.read_csv('F:/Durham/Courses/AIDI/S2/2005 Capstone 2/Project/face-recognition-python-code/Attendance.csv',index_col=False)
    nameList = []
    nameList = df['name'].tolist()
    now = datetime.now()
    time = now.strftime('%I:%M:%S:%p')
    date = now.strftime('%d-%B-%Y')
    if name not in nameList:
        df2 = pd.DataFrame([[name, date, time, time,'(' + str(top) + '/' + str(bottom) + '/' + str(left) + '/' + str(right) + ')']], 
                           columns=['name','date','timein','timeout','location'])
        df = df.append(df2, ignore_index=False)
    else: 
        df.loc[df["name"] == name, "timeout"] = time    
    df.to_csv('F:/Durham/Courses/AIDI/S2/2005 Capstone 2/Project/face-recognition-python-code/Attendance.csv',index=False)


# In[3]:


def attendance (source, path): 
    known_names = []
    known_name_encodings = []
    images = os.listdir(path)
    for _ in images:
        image = fr.load_image_file(path + _)
        image_path = path + _
        encoding = fr.face_encodings(image)[0]
        known_name_encodings.append(encoding)
        known_names.append(os.path.splitext(os.path.basename(image_path))[0].capitalize()) 
    while (True):
        ret,img=source.read()
        if img is None:
            print("No image")
            break
        imgS = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        face_locations_test = fr.face_locations(imgS)
        face_encodings_test = fr.face_encodings(imgS, face_locations_test)
        for (top, right, bottom, left), face_encoding in zip(face_locations_test, face_encodings_test):
            matches = fr.compare_faces(known_name_encodings, face_encoding)
            name = ""
            face_distances = fr.face_distance(known_name_encodings, face_encoding)
            best_match = np.argmin(face_distances)
            if matches[best_match]:
                name = known_names[best_match]
            cv2.rectangle(imgS, (left, bottom), (right, top), (0, 0, 255), 2)
            cv2.rectangle(imgS, (left, bottom - 10), (right, bottom), (0, 255, 0), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(imgS, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)
            markAttendance(name,top,right,left,bottom)
        cv2.imshow('LIVE',imgS)
        key=cv2.waitKey(1)
        
        if(key==27):
            break
        
    cv2.destroyAllWindows()
    source.release()


# In[4]:


source=cv2.VideoCapture(0)
path = "F:/Durham/Courses/AIDI/S2/2005 Capstone 2/Project/face-recognition-python-code/train/"
ip = 'F:/Durham/Courses/AIDI/S2/2005 Capstone 2/Project/face-recognition-python-code/Attendance.csv'
op = 'F:/Durham/Courses/AIDI/S2/2005 Capstone 2/Project/face-recognition-python-code/Attendance.csv'


# In[5]:


file = 'model.pkl'
pickle.dump(attendance, open(file,'wb'))


# In[6]:


loaded_model = pickle.load(open(file, 'rb'))


# In[7]:


loaded_model (source, path)


# In[ ]:




