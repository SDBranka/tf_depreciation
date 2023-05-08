import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pickle import load                                           # to load scaler
from sklearn.model_selection import train_test_split              # to split the data into sets
from sklearn.preprocessing import MinMaxScaler                    # to normalize and scale the data
from sklearn.metrics import mean_absolute_error, mean_squared_error   # for grabbing further metrics about the model's performance
from tensorflow.keras.models import Sequential                    # used to build the model
from tensorflow.keras.layers import Dense                         # used to build the model
from tensorflow.keras.models import load_model                    # to save and load trained models


# load the model
model = load_model("_4_Machine_Learning/keras_3/kc_house_model.h5")
# load the scaler
scaler = load(open("_4_Machine_Learning/keras_3/kc_house_scaler.pkl", "rb"))













# # predicting on brand new data from a saved model
# # we pick a new gem out of the ground and want to see what we should price 
# # it at
# new_gem = [[998,1000]]

# # the model is trained on scaled features, so we must scale the new data
# new_gem = scaler.transform(new_gem)
# print(new_gem)

# # make the prediction
# prediction = model.predict(new_gem)
# print(prediction)




































