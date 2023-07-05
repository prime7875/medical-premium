import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures


db = pd.read_csv("D:/all_datasets/insurance.csv")



db.head()

db.shape

db.info()


db.region.value_counts()


db.isnull().sum()

db.describe()

## distribution of age value

sns.set()
plt.figure(figsize=(6,6))
sns.distplot(db["age"])
plt.title("age distribution")
plt.show()

## count the number of categories in the sex column and graph it

plt.figure(figsize=(6,6))
sns.countplot(x = 'sex',data = db)
plt.title("sex distribution")
plt.show()

db.sex.value_counts()

## bmi distribution

plt.figure(figsize=(6,6))
sns.distplot(db.bmi)
plt.title("bmi distribution")
plt.show()

## children column

plt.figure(figsize=(6,6))
sns.countplot(x = 'children',data = db)
plt.title("children distribution")
plt.show()

db.children.value_counts()

db.head()

plt.figure(figsize=(6,6))
sns.countplot(x = "smoker",data = db)
plt.title("smoker distribution")
plt.show()

db.smoker.value_counts()

db.columns

# Encoding 'region' column
db_encoded = pd.get_dummies(db, columns=['region'], drop_first=True)

# Verify the encoded DataFrame
print(db_encoded.head())


plt.figure(figsize=(6, 6))
sns.countplot(x='region_southwest', data=db_encoded)
plt.title("region distribution")
plt.show()


## distribution of the charges values
plt.figure(figsize=(6,6))
sns.distplot(db.expenses)
plt.title("charges distribution")
plt.show()

###encoding the string to int

db.replace({'sex':{'male':0,'female':1}},inplace = True)

db.replace({'smoker':{'yes':0,'no':1}},inplace = True)

db.replace({'region':{'southeast':0,'southwest':1,'northwest':2,'northeast':3}},inplace=True)

x = db.drop(columns = 'expenses',axis=1)
y = db.expenses

arr_x = x.to_numpy()

if arr_x.ndim > 2:
    arr_x = arr_x.reshape(x,(x.shape[0],-1))

arr_x


arr_y = y.to_numpy()


arr_y = y.to_numpy().reshape(-1, 1)


### spliting the data set into training and testing

x_train,x_test,y_train,y_test = train_test_split(arr_x,arr_y,test_size=0.2,random_state = 2)

## loading the regression model

regressor = LinearRegression()

regressor.fit(x_train,y_train)


### predicting the price

predict = regressor.predict(x_test)

score = metrics.r2_score(y_train,predict)
score
