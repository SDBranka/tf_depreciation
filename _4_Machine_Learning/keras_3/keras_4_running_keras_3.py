import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pickle import load                                           # to load scaler
from sklearn.model_selection import train_test_split              # to split the data into sets
from sklearn.preprocessing import MinMaxScaler                    # to normalize and scale the data
from sklearn.metrics import mean_absolute_error, mean_squared_error, explained_variance_score   # for grabbing further metrics about the model's performance
from tensorflow.keras.models import Sequential                    # used to build the model
from tensorflow.keras.layers import Dense                         # used to build the model
from tensorflow.keras.models import load_model                    # to save and load trained models


# load the model
model = load_model("_4_Machine_Learning/keras_3/kc_house_model.h5")
# load the scaler
scaler = load(open("_4_Machine_Learning/keras_3/kc_house_scaler.pkl", "rb"))
# load the data sets
df = load(open("_4_Machine_Learning/keras_3/kc_house_df.pkl","rb"))
x_train = load(open("_4_Machine_Learning/keras_3/kc_house_x_train.pkl","rb"))
x_test = load(open("_4_Machine_Learning/keras_3/kc_house_x_test.pkl","rb"))
y_train = load(open("_4_Machine_Learning/keras_3/kc_house_y_train.pkl","rb"))
y_test = load(open("_4_Machine_Learning/keras_3/kc_house_y_test.pkl","rb"))


# some evaluation on the test data
# predict on the test set
predictions = model.predict(x_test)
# print(predictions)
# [[502279.4 ]
#  [638654.9 ]
#  [514293.47]
#  ...
#  [422930.28]
#  [217008.4 ]
#  [987994.6 ]]


# compare predictions to actual values
mse_pred_vs_true = mean_squared_error(y_true=y_test,y_pred=predictions)
# print(mse_pred_vs_true)
# 27735988882.64759

# as you can see this number is very large, that is because the price of a
# house is large and with mse a value is squared so instead let's look at
# the root mean squared error
rmse_pred_vs_true = np.sqrt(mean_squared_error(y_true=y_test,y_pred=predictions))
# print(f"rmse_pred_vs_true: {rmse_pred_vs_true}")
# rmse_pred_vs_true: 166541.25279535874

# or the mean absolute error
mae_pred_vs_true = mean_absolute_error(y_true=y_test,y_pred=predictions)
# print(f"mae_pred_vs_true: {mae_pred_vs_true}")
# mae_pred_vs_true: 103726.33512901951


# are these values good or bad, let's look back to the data itself
# print(df["price"].describe())
# count    2.161300e+04
# mean     5.400881e+05
# std      3.671272e+05
# min      7.500000e+04
# 25%      3.219500e+05
# 50%      4.500000e+05
# 75%      6.450000e+05
# max      7.700000e+06
# Name: price, dtype: float64

# as you can see the average price of a house is 540,088.1. Compared to 
# the mae that's off by about 20% which is not that great. What we can also 
# do is use an explained variance score to try and get a deeper understanding
# of our evaluation metrics. Best possible score is 1, lower values are worse. 
# What this tells us is how much variance is explained by our model.

# print(explained_variance_score(y_true = y_test,y_pred = predictions))
# 0.8022880135999582
# this score is just okay, it really depends on the context. Do we have a 
# previous model that actually performs better than this
# We could also continue training the model and try to lower the loss (since
# we haven't yet reached the point of overfitting)


# chart1 
# what we can also compare is our predictions and plot them out against a 
# perfect fit
# plt.scatter(y_test,predictions)
# plt.title("Chart 1")
# plt.show()
# In a perfect world, this would be a straight line

# chart2
# show with a comparison line of y_test
# plt.scatter(y_test,predictions)
# plt.title("Chart 2")
# plt.plot(y_test,y_test,color="r")
# plt.show()
# you can see, we're really getting punished here by the expensive outliers
# when it comes to the expensive houses, we are far off; but when it comes
# to the cheaper houses the model performs pretty well. Essentially this is
# what the explained variance score is trying to report back to us
# it may be worth it to retrain the model only using the lowest 99% of houses


# use the model to predict on a brand new house
# print(df.head())
#       price  bedrooms  bathrooms  sqft_living  sqft_lot  ...     long  sqft_living15  sqft_lot15  year  mon
# th
# 0  221900.0         3       1.00         1180      5650  ... -122.257           1340        5650  2014
# 10
# 1  538000.0         3       2.25         2570      7242  ... -122.319           1690        7639  2014
# 12
# 2  180000.0         2       1.00          770     10000  ... -122.233           2720        8062  2015
#  2
# 3  604000.0         4       3.00         1960      5000  ... -122.393           1360        5000  2014
# 12
# 4  510000.0         3       2.00         1680      8080  ... -122.045           1800        7503  2015
#  2

# [5 rows x 20 columns]

# we'll use the first house of our dataframe as a control
# we know from the dataframe is 221900.0  
single_house = df.drop("price", axis=1).iloc[0]
# print(single_house)
# bedrooms            3.0000
# bathrooms           1.0000
# sqft_living      1180.0000
# sqft_lot         5650.0000
# floors              1.0000
# waterfront          0.0000
# view                0.0000
# condition           3.0000
# grade               7.0000
# sqft_above       1180.0000
# sqft_basement       0.0000
# yr_built         1955.0000
# yr_renovated        0.0000
# lat                47.5112
# long             -122.2570
# sqft_living15    1340.0000
# sqft_lot15       5650.0000
# year             2014.0000
# month              10.0000
# Name: 0, dtype: float64

# convert the data values of the house from a pd.Series to a np.array and 
# reshape it to fit this model (1 col of 19 rows)
single_house = single_house.values.reshape(-1, 19)
# scale the data to match how the model was trained
single_house = scaler.transform(single_house)
# print(single_house)
# [[0.27272727 0.125      0.06716981 0.00310751 0.         0.
#   0.         0.5        0.5        0.09758772 0.         0.47826087
#   0.         0.57149751 0.21760797 0.16193426 0.00573322 0.
#   0.81818182]]

# predict the price against the known value (21900.0 )
print(model.predict(single_house))
# [[289104.28]]

# given how far off this is it would probably be best to retrain the model
# or to drop the highest 1-2% of prices from the data and then to retrain the
# model and see if it becomes more accurate

# This is good enough for this project. Just to show the basics of 
# data engineering


