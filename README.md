# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

Developed by: Mohana K.V.S.L
RegisterNumber:24900659

*/import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn import metrics


data = pd.read_csv("C:\\Users\\sudharshan\\Downloads\\salary.csv")


print(data.head())
print(data.info())
print(data.isnull().sum())


le = LabelEncoder()
data["Position"] = le.fit_transform(data["Position"])


x = data[["Position", "Level"]]
y = data["Salary"]              


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)


dt = DecisionTreeRegressor()
dt.fit(x_train, y_train)


y_pred = dt.predict(x_test)


r2 = metrics.r2_score(y_test, y_pred)
print(f"R-squared: {r2}")


print("Predicted salaries:", y_pred)
```

## Output:
![Screenshot (41)](https://github.com/user-attachments/assets/cc1fbe54-2929-4cdb-8909-a7e532f53f10)



## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
