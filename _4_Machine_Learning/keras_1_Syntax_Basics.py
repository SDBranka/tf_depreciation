# build a model that predicts what price an item should be 
# sold at based upon feature1 and feature2

import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split              # to split the data into sets
from sklearn.preprocessing import MinMaxScaler                    # to normalize and scale the data
from tensorflow.keras


# Step 1: Read in your data
df = pd.read_csv("Data/fake_reg.csv")

# print(df.head())
#         price     feature1     feature2
# 0  461.527929   999.787558   999.766096
# 1  548.130011   998.861615  1001.042403
# 2  410.297162  1000.070267   998.844015
# 3  540.382220   999.952251  1000.440940
# 4  546.024553  1000.446010  1000.338531


# Step 2: Explore your data
# see the features vs the price
# sns.pairplot(df)
# plt.show()


# Step 3: Split the data into training and testing sets
# grab the features that we want to use
# because of the way tf works we need to pass in numpy arrays instead
# of pandas dataframes, so we add .values to the end of a series or
# dataframe and it will return it back as a numpy array
# capital "X" is used to denote that it's a 2d array
X = df[["feature1", "feature2"]].values

# grab the label, what we intend to predict
y = df["price"].values

# print(X)
# [[ 999.78755752  999.7660962 ]
#  [ 998.86161491 1001.04240315]
#  [1000.07026691  998.84401463]
#  ...
#  [1001.45164617  998.84760554]
#  [1000.77102275  998.56285086]
#  [ 999.2322436  1001.45140713]]

# 30% of data will be broken off as a test set. Random_state controls
# how the random split of the data occurs, if you want the same split each
# time the program is run seed this random state with a specific number.
# The value itself it arbitrary, it essentially just seeds the random.
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.3, 
                                                    random_state=42
                                                    )

# check data
# print(X_train.shape)
# (700,2)

# print(X_test.shape)
# (300,2)


# Step 4 Normalize or Scale the Data
# Because we are working with weights and biases if we have very large
# numbers in our feature set that could cause errors with the weights
# One way to try and avoid this is to normalize and scale your feature data
# It is not neccessary to scale the label because it will only be used for 
# comparison and will not be passed through the network

# get information about MinMaxScaler - prints to console
# help(MinMaxScaler)
# see /Resources/MinMaxScaler_help_output.txt for console output

# create an instance of the scaler
scaler = MinMaxScaler()

# fit the scaler onto the training data
# The fit call calculates the parameters it needs to perform the scaling
# later on
# The reason we only run fit on the training set is because we want to 
# prevent what's called data leakage from the test set. We don't want
# to assume that we have prior information of the test set. So we only fit
# our scaler to the training set to not try to "cheat" and look into the 
# test set
scaler.fit(X_train)

# transform the data for both the training and test sets of data
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# taking a look at the data now we can see it has been scaled
# print(X_train)
# [[0.74046017 0.32583248]
#  [0.43166001 0.2555088 ]
#  [0.18468554 0.70500664]
#  ...
#  [0.54913363 0.79933822]
#  [0.2834197  0.38818708]
#  [0.56282703 0.42371827]]

# checking now all the data falls between 0.0 and 1.0
# print(X_train.max())
# 1.0

# print(X_train.max())
# 0.0


# Create the model/neural network

































