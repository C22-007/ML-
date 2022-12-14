# -*- coding: utf-8 -*-
"""Machine Learning Capstone

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/17vTz-GI1xJej3HQjRlOFu5cy3tawVkSW

# Import Library
"""

from zipfile import ZipFile
import numpy as np 
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from pathlib import Path
from keras import layers
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
import seaborn  as sns

"""# Import kaggle"""

pip install -q kaggle

import os
os.environ['KAGGLE_USERNAME'] = "alpiansyahrizqi"
os.environ['KAGGLE_KEY'] = "4ec2871806d4a125d51415bdc1e5a75a"

!kaggle datasets download -d alpiansyahrizqi/capstone-project

!unzip -q capstone-project.zip -d .

"""# Import dataset"""

df = pd.read_csv('/content/c22_007.csv')

df.head()

"""#  Exploratory Data Analysis"""

df.isnull().sum()

df.info()

df

df.describe()

plt.figure(figsize=(18,15))
sns.heatmap(df.corr(),annot=True)
plt.title('Heatmap of Variable Correlations')
plt.show()

plt.figure(figsize=(15,5))
sns.lineplot(x='Abdomen',y='BodyFat',data=df)
plt.title('Body Fat vs Abdomen size')
plt.xlabel('Abdomen size')
plt.ylabel('Body Fat')
plt.show()

plt.figure(figsize=(15,5))
sns.lineplot(x='Chest',y='BodyFat',data=df)
plt.title('Body Fat vs Chest size')
plt.xlabel('Chest size')
plt.ylabel('Body Fat')
plt.show()

plt.figure(figsize=(5,5))
sns.distplot(df['BodyFat'])
plt.title('Distribution of Body Fat')
plt.xlabel('Body Fat')
plt.show()

df['BodyFat'].describe()

sns.scatterplot(x=df['Weight'],y=df['BodyFat'])

df['Weight'].describe()

df.hist(figsize=(16,12))

df.columns

sns.pairplot(df,x_vars=['Density','Age','Weight','Height','Neck'],y_vars=['BodyFat'])

sns.pairplot(df,x_vars=['Chest','Abdomen','Hip','Thigh','Knee'],y_vars=['BodyFat'])

sns.pairplot(df,x_vars=['Ankle','Biceps','Forearm','Wrist','BMI'],y_vars=['BodyFat'])

sns.scatterplot(x=df['BMI'],y=df['BodyFat'])

len(df.columns)

px = 1
plt.figure(figsize=(20,20))
for i in ['Density', 'BodyFat', 'Age', 'Weight', 'Height', 'Neck', 'Chest',
       'Abdomen', 'Hip', 'Thigh', 'Knee', 'Ankle', 'Biceps', 'Forearm',
       'Wrist', 'BMI']:
    if px<17:
        plt.subplot(6,5,px)
        plt.boxplot(df[i])
        plt.title(i)
        px=px+1

def outlier():
    l = ['Density', 'BodyFat', 'Age', 'Weight', 'Height', 'Neck', 'Chest',
       'Abdomen', 'Hip', 'Thigh', 'Knee', 'Ankle', 'Biceps', 'Forearm',
       'Wrist', 'BMI']
    for i in l:
        x = np.quantile(df[i],[0.25,0.75])
        iqr = x[1]-x[0]   
        lof = x[0]-1.5*iqr   
        upf = x[1]+1.5*iqr   
        df[i] = np.where(df[i]>upf,upf,(np.where(df[i]<lof,lof,df[i])))
outlier()

px = 1
plt.figure(figsize=(20,20))
for i in ['Density', 'BodyFat', 'Age', 'Weight', 'Height', 'Neck', 'Chest',
       'Abdomen', 'Hip', 'Thigh', 'Knee', 'Ankle', 'Biceps', 'Forearm',
       'Wrist', 'BMI']:
    if px<17:
        plt.subplot(6,5,px)
        plt.boxplot(df[i])
        plt.title(i)
        px=px+1

df.describe()

"""#MOdel 1 predict density

# Data preperation
"""

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X = df[['Age','Weight','Height','Density','BMI']]
y = df['BodyFat']
X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=0.7,random_state=100)

X_test

y_test.head()

"""# Modeling dan Eval"""

from sklearn.linear_model import   ElasticNet , Lasso , Ridge
from sklearn.metrics import r2_score,accuracy_score
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Lasso

bay = BayesianRidge()
testing = bay.fit(X_train,y_train.values)
y_pred = testing.predict(X_test)

x_ax = range(len(y_test))
plt.figure(figsize=(18,10))
plt.scatter(x_ax, y_test, s=5, color="blue", label="original")
plt.plot(x_ax, y_pred, lw=0.8, color="red", label="predicted")
plt.legend()
plt.show()

import pickle

pickle.dump(testing, open('bodyfat_prediction.pkl', 'wb'))

pickled_model = pickle.load(open('bodyfat_prediction.pkl', 'rb'))
pickled_model.predict(X_test)

X_test

df.head()

pickled_model.predict(np.array([(23.0,69.4125,	172.085 ,1.059463,23.439679)]).reshape(1,-1))

y_pred[0]