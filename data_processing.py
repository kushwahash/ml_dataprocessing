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
Y = dataset.iloc[:,:4].values

#take care of missing data
from sklearn.preprocessing import Imputer
imputer_obj = Imputer(missing_values=np.nan,strategy="mean",axis=0)
#fit the Matrix X and the needed column which needs cleaning
imputer_obj = imputer_obj.fit(X[:, 1:3])
#now get the transform data to X
X[:,1:3] = imputer_obj.transform(X[:, 1:3])