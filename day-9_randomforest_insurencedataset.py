import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data=pd.read_csv("C:\\Users\\elcot\\Desktop\\machine learning tutorial\\classification\\SVM & Decision tree\\insurance.csv")

from sklearn.preprocessing import LabelEncoder
label=LabelEncoder()
data['Smoke']=label.fit_transform(data['smoker'])
data['location']=label.fit_transform(data['region'])


x=data.iloc[:,[0,2,3,7,8]].values
y=data.iloc[:,[6]].values


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)

from sklearn.ensemble import RandomForestRegressor
classifier=RandomForestRegressor()
classifier.fit(x_train,y_train)
y_pred=classifier.predict(x_test)
classifier.score(x_train,y_train)

from sklearn.metrics import r2_score
r2_score(y_test,y_pred)

