# A template for data preprocessing to apply ML

# Import the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas  as pd

# Import the data set
dataset = pd.read_csv("Data.csv")
#divide it into independent(X) and dependent (Y) section
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,3].values

#splitting the dataset into train and test
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)


'''
#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
'''