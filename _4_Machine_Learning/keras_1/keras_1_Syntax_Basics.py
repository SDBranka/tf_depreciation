# build a model that predicts what price an item should be 
# sold at based upon feature1 and feature2


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pickle import dump                                           # to save the scaler           
from sklearn.model_selection import train_test_split              # to split the data into sets
from sklearn.preprocessing import MinMaxScaler                    # to normalize and scale the data
from sklearn.metrics import mean_absolute_error, mean_squared_error   # for grabbing further metrics about the model's performance
from tensorflow.keras.models import Sequential                    # used to build the model
from tensorflow.keras.layers import Dense                         # used to build the model
from tensorflow.keras.models import load_model                    # to save and load trained models


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
# see /Resources/Help_Outputs/MinMaxScaler_help_output.txt for console output

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


# help(Sequential)
# see /Resources/Help_Outputs/Sequential_help.txt
# help(Dense)
# see /Resources/Help_Outputs/Dense_help.txt

# Create the model/neural network

# # one way to do this
# # call sequential and pass in a list of the layers you want
# # a Dense layer is a normal feed forward network where every neuron is
# # connected to every other neuron in the next layer
# # units is the number of neurons the layer will have, activation is the 
# # activation function that will be used
# # so 4 input neurons, 2 hidden layer neurons, 1 output neuron
# model = Sequential([Dense(units=4, activation="relu"), 
#                     Dense(units=2, activation="relu"),
#                     Dense(units=1)
#                     ])

# our preferred way to do this
# build model with 3 layers of 4 neurons each with an output layer of 
# one neuron. Output layer is built this way because we only want to produce
# a single predicted sales price
model = Sequential()
model.add(Dense(4,activation="relu"))
model.add(Dense(4,activation="relu"))
model.add(Dense(4,activation="relu"))

model.add(Dense(1))

# compile the model
# this is a linear regression problem so we will use mean squared error
# as the loss function
model.compile(optimizer="rmsprop", loss="mse")


# train/fit the model to the training data
# x is the features we are training the model on
# y is the label we intend to predict from the training set
# epochs is the number of times we will pass the data through the network
# verbose is used to determine if the training process will be shown in the
# console output, if you include this parameter and set it to 0, the 
# training data will not output to the console
model.fit(x=X_train, y=y_train, epochs=250)


# take a look at the training history
# shows the loss of each epoch as a dictionary of a list of all the 
# loss values
# print(model.history.history)
# {'loss': [256630.625, 256538.3125, 256447.109375, 256350.359375, 256247.640625, 256139.359375, 256025.4843
# 75, 255903.15625, 255772.4375, 255633.28125, 255484.234375, 255326.421875, 255157.375, 254977.171875, 2547
# 87.15625, 254584.453125, 254369.484375, 254141.09375, 253899.046875, 253641.78125, 253372.234375, 253086.9
# 21875, 252785.625, 252468.140625, 252134.609375, 251782.875, 251414.15625, 251024.75, 250618.1875, 250190.
# 515625, 249740.890625, 249272.703125, 248782.40625, 248268.34375, 247733.46875, 247172.390625, 246587.3593
# 75, 245979.796875, 245345.078125, 244681.890625, 243990.78125, 243277.046875, 242530.5625, 241756.609375,
# 240948.640625, 240117.546875, 239253.6875, 238356.828125, 237422.53125, 236457.21875, 235465.390625, 23443
# 4.65625, 233367.078125, 232262.90625, 231117.421875, 229944.34375, 228732.3125, 227474.53125, 226188.29687
# 5, 224852.609375, 223480.28125, 222070.375, 220614.5625, 219122.3125, 217579.25, 216000.4375, 214377.84375
# , 212707.21875, 210991.453125, 209246.109375, 207440.109375, 205597.8125, 203714.828125, 201775.0, 199789.
# 8125, 197756.484375, 195680.359375, 193561.171875, 191397.171875, 189182.0625, 186922.390625, 184605.42187
# 5, 182252.65625, 179847.46875, 177387.046875, 174907.046875, 172361.578125, 169776.90625, 167154.375, 1644
# 78.640625, 161758.0625, 158991.53125, 156191.484375, 153346.140625, 150463.03125, 147533.890625, 144560.03
# 125, 141540.1875, 138502.5625, 135429.796875, 132303.84375, 129151.2578125, 125982.8671875, 122783.234375,
#  119546.921875, 116306.265625, 113026.171875, 109710.078125, 106386.3671875, 103033.09375, 99664.4765625,
# 96309.2109375, 92935.8203125, 89557.296875, 86178.1953125, 82780.5859375, 79394.875, 76025.4453125, 72663.
# 6953125, 69308.9765625, 65964.328125, 62659.25, 59389.71484375, 56142.83984375, 52933.3671875, 49762.01562
# 5, 46628.3984375, 43560.15234375, 40541.98828125, 37591.71875, 34699.328125, 31893.869140625, 29191.697265
# 625, 26536.0078125, 23991.685546875, 21546.076171875, 19196.943359375, 16985.75390625, 14881.681640625, 12
# 927.5380859375, 11103.20703125, 9420.4833984375, 7894.24560546875, 6523.81005859375, 5326.33984375, 4307.2
# 998046875, 3469.801513671875, 2824.153076171875, 2347.34619140625, 2049.598876953125, 1908.7803955078125,
# 1857.5028076171875, 1835.057861328125, 1818.4478759765625, 1799.7109375, 1783.423095703125, 1769.533691406
# 25, 1752.6314697265625, 1737.2330322265625, 1720.856201171875, 1705.016845703125, 1687.255859375, 1670.777
# 7099609375, 1654.938720703125, 1638.8707275390625, 1621.099609375, 1603.39013671875, 1586.8873291015625, 1
# 571.850341796875, 1555.4425048828125, 1536.839111328125, 1518.900390625, 1502.4090576171875, 1486.41503906
# 25, 1471.78466796875, 1455.6522216796875, 1437.6514892578125, 1419.6844482421875, 1402.901611328125, 1388.
# 1171875, 1371.81201171875, 1357.525390625, 1342.21240234375, 1326.4078369140625, 1312.2811279296875, 1294.
# 5091552734375, 1277.98828125, 1263.5076904296875, 1246.3568115234375, 1229.251953125, 1212.315673828125, 1
# 196.6519775390625, 1180.7537841796875, 1168.8951416015625, 1155.569091796875, 1140.6552734375, 1124.960449
# 21875, 1109.5386962890625, 1095.1634521484375, 1080.0111083984375, 1066.467041015625, 1051.597900390625, 1
# 036.1190185546875, 1022.0908203125, 1005.7154541015625, 993.2252807617188, 977.6715087890625, 962.37152099
# 60938, 949.4417114257812, 933.5592041015625, 919.76953125, 907.3487548828125, 891.8981323242188, 876.91796
# 875, 865.7172241210938, 851.9208984375, 837.4083862304688, 822.3204345703125, 807.1279296875, 792.46704101
# 5625, 780.0341796875, 767.7009887695312, 754.1807250976562, 743.4107666015625, 730.1913452148438, 717.8844
# 604492188, 704.9631958007812, 691.4811401367188, 680.100341796875, 668.493408203125, 657.0263671875, 644.8
# 218994140625, 632.9769287109375, 620.1166381835938, 607.2181396484375, 594.8644409179688, 585.415161132812
# 5, 572.0447387695312, 559.69921875, 547.468994140625, 536.5372924804688, 525.968017578125, 514.80743408203
# 12, 504.1950988769531, 492.21588134765625, 482.5263977050781, 472.5870056152344, 462.22412109375, 450.6370
# 2392578125, 442.775390625]}

# so we can turn this into a dataframe
# loss_df = pd.DataFrame(model.history.history)
# print(loss_df.head())
#             loss
# 0  256647.718750
# 1  256515.312500
# 2  256368.578125
# 3  256199.062500
# 4  256008.390625

# chart1
# so we are able to plot this out on a graph
# loss_df.plot()
# plt.title("Chart 1")
# plt.show()


# evaluate the model's performance
# one way to do this is with .evaluate(). to do this we pass in our test
# set (verbose set to 0 to eliminate console output) this returns back the 
# model's metric "loss" for the test set of data, based on the loss function
# selected earlier (in this case "mse" or mean squared error)
print(model.evaluate(X_test, y_test, verbose=0))
# 25.034265518188477

# we can also test the loss on the training set
print(model.evaluate(X_train, y_train, verbose=0))
# 23.747806549072266

# get our actual true predictions
test_predictions = model.predict(X_test)
# so here is a list of prices based off our X_test set
# print(test_predictions)
# [[408.61255]
#  [621.2526 ]
#  [590.2641 ]
# ...
#  [607.52954]
#  [419.1815 ]
#  [414.70398]]

# test the predictions
# let's bring these together with the true values from the test set and 
# then plot them out and compare them to each other
# turn the test predictions into a pandas series and reshape to 300 bc 
# just so that it matches the dimensions that a pandas Series call expects
#  since the test data has 300 values (set up earlier in the app) now we 
# have the same predictions as a pandas series instead of a numpy array
test_predictions = pd.Series(test_predictions.reshape(300,))

# create a dataframe that is the true value of y
pred_df = pd.DataFrame(y_test, columns=["True Test Y"])
# print(pred_df)
#      True Test Y
# 0     402.296319
# 1     624.156198
# 2     582.455066
# 3     578.588606
# 4     371.224104
# ..           ...
# 295   525.704657
# 296   502.909473
# 297   612.727910
# 298   417.569725
# 299   410.538250

# [300 rows x 1 columns]

# concat the predicted value df with the actual value df
# axis set to 1 to add the df as a new column
pred_df = pd.concat([pred_df, test_predictions], axis=1)
# print(pred_df)
#      True Test Y           0
# 0     402.296319  405.191376
# 1     624.156198  623.482483
# 2     582.455066  592.065186
# 3     578.588606  572.167603
# 4     371.224104  366.526459
# ..           ...         ...
# 295   525.704657  528.976929
# 296   502.909473  515.279297
# 297   612.727910  609.590149
# 298   417.569725  416.851593
# 299   410.538250  410.784454

# [300 rows x 2 columns]

# correct the names of the columns
pred_df.columns = ["Test True Y", "Model Predictions"]
# print(pred_df)
#      Test True Y  Model Predictions
# 0     402.296319         406.739960
# 1     624.156198         625.411743
# 2     582.455066         593.913391
# 3     578.588606         574.066833
# 4     371.224104         368.121735
# ..           ...                ...
# 295   525.704657         530.792297
# 296   502.909473         517.205994
# 297   612.727910         611.482178
# 298   417.569725         418.366913
# 299   410.538250         412.386475

# [300 rows x 2 columns]

# chart2
# create a scatter plot to visually represent the predictions vs the actual
# values
# sns.scatterplot(x="Test True Y",y="Model Predictions",data=pred_df)
# plt.title("Chart 2")
# plt.show()
# that this shows such a tightly packed line we can tell the model is 
# performing well

# grab metrics to show this information quantatively through the various
# regression methods
# if we want the mean_absolute_error then we just need to compare 
# y_test (true values of y) to the predicted values for y
# since we already have these organized in a dataframe we can call it like
mae = mean_absolute_error(pred_df["Test True Y"], 
                        pred_df["Model Predictions"]
                        )
# print(mae)
# 3.997765644868322
# This means we're about $4 off with the prediction from the actual price 
# point. Looking at metrics from the original data
# print(df.describe())
#              price     feature1     feature2
# count  1000.000000  1000.000000  1000.000000
# mean    498.673029  1000.014171   999.979847
# std      93.785431     0.974018     0.948330
# min     223.346793   997.058347   996.995651
# 25%     433.025732   999.332068   999.316106
# 50%     502.382117  1000.009915  1000.002243
# 75%     564.921588  1000.637580  1000.645380
# max     774.407854  1003.207934  1002.666308
# bc the average price is about $498 the model is performing with a better 
# than 1% error

# in this case it will be the same as the loss bc we used mse as the loss
# function
mse = mean_squared_error(pred_df["Test True Y"],
                        pred_df["Model Predictions"]
                        )
# print(mse)
# 25.802020835580425

# root mean squared error
# just take the square root of mean squared error
rmse = mean_squared_error(pred_df["Test True Y"],
                        pred_df["Model Predictions"]
                        )**0.5
# print(rmse)
# 5.105797469961507


# predicting on brand new data
# we pick a new gem out of the ground and want to see what we should price 
# it at
new_gem = [[998,1000]]

# the model is trained on scaled features, so we must scale the new data
new_gem = scaler.transform(new_gem)
print(new_gem)
# [[0.14117652 0.53968792]]

# make the prediction
prediction = model.predict(new_gem)
print(prediction)
# [[418.62036]]
# The model predicts this gem to be priced at around $418.62036 based on
# it's features

# the way you evaluate your test set is essentially the same as you'd 
# evaluate new data

# saving the model
# save the model as a .h5 file
model.save('my_gem_model.h5')
# save the scaler
dump(scaler,open("scaler.pkl","wb"))

# # to load a model
# later_model = load_model("my_gem_model.h5")
















