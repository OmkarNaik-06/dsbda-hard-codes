import numpy as np
import pandas as pd 
from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
#loading the dataset direclty from sklearn
boston = datasets.load_boston()
print(boston)

bos.describe()

#sklearn returns Dictionary-like object, the interesting attributes are: ‘data’, the data to learn, ‘target’, the 
regression targets, ‘DESCR’, the full description of the dataset, and ‘filename’, the physical location of boston 
csv dataset.
print(type(boston))
41
print('\n')
print(boston.keys())
print('\n')
print(boston.data.shape)
print('\n')
print(boston.feature_names)

#The details about the features and more information about the dataset can be seen by using boston.DESCR
print(boston.DESCR)

#Before applying any model we have to convert this to a pandas dataframe, 
#which we can do by calling the dataframe on boston.data. We also adds the target variable to the dataframe
from boston.target
bos = pd.DataFrame(boston.data, columns = boston.feature_names)
bos['PRICE']=pd.DataFrame(boston.target)
print(bos.head())

#Get some statistics from dataset
print(bos.describe())

#initialize linear regression model
reg=LinearRegression()
#split into training-80% & testing data-20%
X_train, X_test, Y_train, Y_test = train_test_split(bos, bos['PRICE'], test_size = 0.20,random_state=10)
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)

#train model with our training data
reg.fit(X_train,Y_train)
#print predictions on our test data
y=reg.predict(X_test)
print(y)

#actucal values
print(Y_test)
reg.score(X_test,Y_test)

from sklearn.metrics import mean_squared_error
y = reg.predict(X_test)
rmse = (np.sqrt(mean_squared_error(Y_test, y)))
r2 = round(reg.score(X_test, Y_test),2)
print("The model performance for training set")
42
print("--------------------------------------")
print("Root Mean Squared Error: {}".format(rmse))
print("R^2: {}".format(r2))
print("\n")
