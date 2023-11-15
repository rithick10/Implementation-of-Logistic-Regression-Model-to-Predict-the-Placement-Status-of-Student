# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:

To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:

Hardware – PCs
Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm:

1.Import the standard libraries.

2.Upload the dataset and check for any null or duplicated values using .isnull() and .duplicated() function respectively.

3.Import LabelEncoder and encode the dataset.

4.Import LogisticRegression from sklearn and apply the model on the dataset.

5.Predict the values of array.

6.Calculate the accuracy, confusion and classification report by importing the required modules from sklearn.
Apply new unknown values


## Program:
~~~
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: Udayanithi N
RegisterNumber:  212221220056
*/

import pandas as pd
df=pd.read_csv("Placement_Data(1).csv")
df.head()

df1=df.copy()
df1=df1.drop(["sl_no","salary"],axis=1)
df1.head()

df1.isnull().sum()

df1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df1["gender"]=le.fit_transform(df1["gender"])
df1["ssc_b"]=le.fit_transform(df1["ssc_b"])
df1["hsc_b"]=le.fit_transform(df1["hsc_b"])
df1["hsc_s"]=le.fit_transform(df1["hsc_s"])
df1["degree_t"]=le.fit_transform(df1["degree_t"])
df1["workex"]=le.fit_transform(df1["workex"])
df1["specialisation"]=le.fit_transform(df1["specialisation"])
df1["status"]=le.fit_transform(df1["status"])
df1

x=df1.iloc[:,:-1]
x

y=df1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
classification_report1

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
~~~

## Output:

Original Data:

<img width="655" alt="41" src="https://user-images.githubusercontent.com/100425381/202348514-cc2e9123-cd77-434f-bbd7-d1e527af08d2.png">


After removing:

<img width="655" alt="42" src="https://user-images.githubusercontent.com/100425381/202348569-2f7ae0b0-fcd5-46d9-ac12-c521e85f68aa.png">


Null Data:

<img width="347" alt="43" src="https://user-images.githubusercontent.com/100425381/202349240-7f9a50e3-67b2-4016-afc8-32b425e7c342.png">



Label Encoder:

<img width="648" alt="44" src="https://user-images.githubusercontent.com/100425381/202349046-bfc3890e-f8bb-445f-9585-a82826df1e5a.png">


X:

<img width="648" alt="45" src="https://user-images.githubusercontent.com/100425381/202349073-5f1b61ef-7317-423a-9ac9-b0ce082cfa77.png">


Y:

<img width="648" alt="46" src="https://user-images.githubusercontent.com/100425381/202349080-500448be-7175-4736-b4bb-1c408d61e1f6.png">


Y_prediction:

<img width="648" alt="47" src="https://user-images.githubusercontent.com/100425381/202349085-bda9c376-85c0-457b-a1f4-6cb2cb9d685b.png">


Accuracy:

<img width="648" alt="48" src="https://user-images.githubusercontent.com/100425381/202349090-8ee19e4c-3817-4f3c-ba76-4a301c17b5d5.png">


Cofusion:

<img width="460" alt="49" src="https://user-images.githubusercontent.com/100425381/202349095-12bf85eb-f950-4854-aeb4-2e28f5b1d81a.png">


Classification:

<img width="648" alt="410" src="https://user-images.githubusercontent.com/100425381/202349135-3cd9bc9d-368e-4c7e-a742-234a8eab44a0.png">


## Result:

Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
