import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing

import os

#Importing the dataset

train_KDD = pd.read_csv("./KDDTrain+.csv")
test_KDD = pd.read_csv("./KDDTest+.csv")
print(f"Train identity shape: {train_KDD.shape}")
print(f"Test identity shape: {test_KDD.shape}")

#feature selection
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder

for column in train_KDD.columns:
    if train_KDD[column].dtype == type(object):
        le = LabelEncoder()
        train_KDD[column] = le.fit_transform(train_KDD[column])

X = train_KDD.iloc[:, 0:41]
y = train_KDD.iloc[:, 42]
print(y.shape)
res = mutual_info_classif(X, y)
# print(res)
print("\n",res.shape)
res_d = dict()
res_d[0] = -1
for ind, val in enumerate(res):
    res_d[ind+1] = val

for v in (sorted(res_d.items(), key=lambda x: x[1], reverse=True)):
    print(v[0], "-->", v[1])
# print(res_d)