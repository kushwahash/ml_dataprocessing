#for math operation
import numpy as np
# for plotting
import matplotlib.pyplot as plt
#to import and manage dataaset
import pandas as pd

#importing dataset
dataset = pd.read_csv('Data.csv')

#reading dataset and creating Matrix X for independent variable and Y for dependent variable

X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:,3].values

#take care of missing data
from sklearn.preprocessing import Imputer
imputer_obj = Imputer(missing_values=np.nan,strategy="mean",axis=0)
#fit the Matrix X and the needed column which needs cleaning
imputer_obj = imputer_obj.fit(X[:, 1:3])

#now get the transform data to X
X[:,1:3] = imputer_obj.transform(X[:, 1:3])

#Encoding categorical Data (like country and purchased)
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
encode_X = LabelEncoder()
#fit the encoder with first column of data
X[:,0] = encode_X.fit_transform(X[:,0])
'''
but this create problem as it will assign numbers which will
make one country important then other which is not true
USING ONEHOTENCODER
'''
onehotencoder = OneHotEncoder(categorical_features=[0])
X = onehotencoder.fit_transform(X).toarray()

#using labler encoder for dependent variable
encode_Y = LabelEncoder()
Y = encode_Y.fit_transform(Y)