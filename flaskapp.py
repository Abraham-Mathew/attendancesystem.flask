import numpy as np
import pickle
import pandas as pd
from flask import Flask, request, jsonify, render_template
import cv2
from datetime import datetime

app=Flask(__name__)
pickle_in = open("model.pkl","rb")
fr_model = pickle.load(pickle_in)

def markAttendance(name,top,right,left,bottom):            
    df = pd.read_csv('Attendance.csv',index_col=False)
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
    df.to_csv('Attendance.csv',index=False)

source=cv2.VideoCapture(0)
path = "train/"

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict',methods=["POST"])
def predict():
    """
    For rendering results on HTML GUI
    """
    fr_model(source, path)

if __name__=='__main__':
    app.run()