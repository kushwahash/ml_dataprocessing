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
USING ONEHOTENCODER which will create different column for each country and assign 1 or 0 for
whether the country is present for the row
'''
onehotencoder = OneHotEncoder(categorical_features=[0])
X = onehotencoder.fit_transform(X).toarray()

#using labler encoder for dependent variable
encode_Y = LabelEncoder()
Y = encode_Y.fit_transform(Y)

#splitting the dataset for Training and Test
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size = 0.2,random_state=0)

'''
feature scaling to avoid dominance of one feature. For example, some data mining techniques use the 
Euclidean distance. Therefore, all parameters should have the same scale for a fair comparison between them.
Example, salary here is big range number which will dominate age. 
So, scale it We can use standarisation or normalisation
'''

#feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
