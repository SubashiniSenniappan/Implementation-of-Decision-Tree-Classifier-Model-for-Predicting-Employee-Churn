# Ex-06 Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn
## DATE:

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
Import the required libraries.
Upload and read the dataset.
Check for any null values using the isnull() function.
From sklearn.tree import DecisionTreeClassifier and use criterion as entropy.
Find the accuracy of the model and predict the required values by importing the required module from sklearn
## Program:
```
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: Subashini.S
RegisterNumber:  212222240106

```
```
import pandas as pd
data = pd.read_csv("Employee.csv")
data.head()
data.info()
data.isnull().sum()
data["left"].value_counts()
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data["salary"] = le.fit_transform(data["salary"])
data.head()
x = data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()
y = data["left"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=100)
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(criterion = "entropy")
dt.fit(x_train,y_train)
y_pred = dt.predict(x_test)
from sklearn import metrics
accuracy = metrics.accuracy_score(y_test,y_pred)
accuracy
dt.predict([[0.5,0.8,9,260,6,0,1,2]])
```

## Output:
## Data head:

![276489296-a75903c7-f185-42e2-853a-276435f4e6f3](https://github.com/SubashiniSenniappan/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119404951/afafaf39-cf5a-4883-863f-fa311bd61cb9)

## Information:
![276489431-74603144-2086-4add-8001-5f0db6c08a4b](https://github.com/SubashiniSenniappan/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119404951/bcd87aa2-a45a-4d8f-bb59-e7ec770d1735)

## Null set

![276489527-7b35f090-016e-4aa8-9ab2-d4a1b477f420](https://github.com/SubashiniSenniappan/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119404951/0f5412b8-1586-4278-b9f1-def2a1e7bdb2)

## Value:

![276489622-f94ef29d-fdee-4b48-a40d-4e17e51cdf2e](https://github.com/SubashiniSenniappan/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119404951/039eb4af-3ead-4b2e-a6f7-f0cbc24180dc)

## Data head:
![276489691-56767095-20e1-4b9f-b44a-6885b53a7d33](https://github.com/SubashiniSenniappan/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119404951/6eff2d20-c921-472a-91e7-e807c7e9e587)

## x.head():
![276489792-bcdeb18e-03be-4f84-82a8-91cb909e29dc](https://github.com/SubashiniSenniappan/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119404951/3cabfd5e-4850-44b2-a0f7-b8545451d944)

## Data Prediction:
![276489903-f6950012-686f-4982-9b56-b66bbf238449](https://github.com/SubashiniSenniappan/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119404951/d9ae68f6-9df0-4863-bae6-eec382134886)


## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
