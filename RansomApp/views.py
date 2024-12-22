from django.shortcuts import render
from django.template import RequestContext
from django.contrib import messages
import pymysql
from django.http import HttpResponse
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer #loading tfidf vector
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

global uname

dataset = pd.read_csv("Dataset/dga_data.csv",nrows=5000)
labels = np.unique(dataset['subclass'].values.ravel())
dataset = dataset.dropna()
dataset = dataset.values

def getLabel(name):
    index = -1
    for i in range(len(labels)):
        if labels[i] == name:
            index = i
            break
    return index    

X = []
Y = []

for i in range(len(dataset)):
    dga = dataset[i,1]
    label = getLabel(dataset[i,3])
    X.append(dga)
    Y.append(label)    

X = np.asarray(X)
Y = np.asarray(Y)

tfidf_vectorizer = TfidfVectorizer()
X = tfidf_vectorizer.fit_transform(X).toarray()
print(X.shape)
sc = StandardScaler()
X = sc.fit_transform(X)

indices = np.arange(X.shape[0])
np.random.shuffle(indices)
X = X[indices]
Y = Y[indices]

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2) #split dataset into train and test
X_train, X_test1, y_train, y_test1 = train_test_split(X, Y, test_size=0.1) #split dataset into train and test
    
lr_cls = LogisticRegression(max_iter=300) #create Logistic Regression object
lr_cls.fit(X_train, y_train)
predict = lr_cls.predict(X_test)
acc = accuracy_score(y_test,predict)*100

def RunDGAAction(request):
    if request.method == 'POST':
        global lr_cls, tfidf_vectorizer, sc, labels
        domain = request.POST.get('t1', False)
        vector = tfidf_vectorizer.transform([domain]).toarray()
        vector = sc.transform(vector)
        predict = lr_cls.predict(vector)[0]
        predict = int(predict)
        print(predict)
        predict = labels[predict]
        output = "Given Domain = "+domain+"<br/>"
        output += "DGA Predicted AS ====> "+predict
        context= {'data':output}
        return render(request, 'RunDGA.html', context)        

def UserLogin(request):
    if request.method == 'GET':
       return render(request, 'UserLogin.html', {})

def index(request):
    if request.method == 'GET':
       return render(request, 'index.html', {})

def Signup(request):
    if request.method == 'GET':
       return render(request, 'Signup.html', {})

def Aboutus(request):
    if request.method == 'GET':
       return render(request, 'Aboutus.html', {})

def SignupAction(request):
    if request.method == 'POST':
        username = request.POST.get('t1', False)
        password = request.POST.get('t2', False)
        contact = request.POST.get('t3', False)
        email = request.POST.get('t4', False)
        address = request.POST.get('t5', False)
        
        status = 'none'
        con = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = 'root', database = 'RansomApp',charset='utf8')
        with con:
            cur = con.cursor()
            cur.execute("select username from signup where username = '"+username+"'")
            rows = cur.fetchall()
            for row in rows:
                if row[0] == email:
                    status = 'Given Username already exists'
                    break
        if status == 'none':
            db_connection = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = 'root', database = 'RansomApp',charset='utf8')
            db_cursor = db_connection.cursor()
            student_sql_query = "INSERT INTO signup(username,password,contact_no,email_id,address) VALUES('"+username+"','"+password+"','"+contact+"','"+email+"','"+address+"')"
            db_cursor.execute(student_sql_query)
            db_connection.commit()
            print(db_cursor.rowcount, "Record Inserted")
            if db_cursor.rowcount == 1:
                status = 'Signup Process Completed'
        context= {'data':status}
        return render(request, 'Signup.html', context)

def UserLoginAction(request):
    if request.method == 'POST':
        global uname
        option = 0
        username = request.POST.get('username', False)
        password = request.POST.get('password', False)
        con = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = 'root', database = 'RansomApp',charset='utf8')
        with con:
            cur = con.cursor()
            cur.execute("select * FROM signup")
            rows = cur.fetchall()
            for row in rows:
                if row[0] == username and row[1] == password:
                    uname = username
                    option = 1
                    break
        if option == 1:
            context= {'data':'welcome '+username}
            return render(request, 'UserScreen.html', context)
        else:
            context= {'data':'Invalid login details'}
            return render(request, 'UserLogin.html', context)

def LoadDGA(request):
    if request.method == 'GET':
        global labels, acc
        output = "DGA can detect different Ransomware : "+str(labels)+"<br/>"
        output += "DGA Ransomware Attack Detection Model Loaded<br/>Detection Accuracy % = "+str(acc)
        context= {'data':output}
        return render(request, 'UserScreen.html', context)

def RunDGA(request):
    if request.method == 'GET':
        return render(request, 'RunDGA.html', {})     

