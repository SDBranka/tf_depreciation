import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pickle import load                                           # to load scaler
# from sklearn.model_selection import train_test_split              # to split the data into sets
# from sklearn.preprocessing import MinMaxScaler                    # to normalize and scale the data
from sklearn.metrics import mean_absolute_error, mean_squared_error,\
                            explained_variance_score, classification_report,\
                            confusion_matrix                      # for grabbing further metrics about the model's performance
# from tensorflow.keras.models import Sequential                    # used to build the model
# from tensorflow.keras.layers import Dense,Dropout                         # used to build the model
from tensorflow.keras.models import load_model                    # to save and load trained models



### --------- Load Saved Model, Scaler, and Datasets --------- ###
# load the model
model = load_model("_4_Machine_Learning/keras_5_classification/Runs/model.h5")
# load the scaler
scaler = load(open("_4_Machine_Learning/keras_5_classification/Runs/scaler.pkl", "rb"))
# load the data sets
df = load(open("_4_Machine_Learning/keras_5_classification/Runs/df.pkl","rb"))
X_train = load(open("_4_Machine_Learning/keras_5_classification/Runs/x_train.pkl","rb"))
X_test = load(open("_4_Machine_Learning/keras_5_classification/Runs/x_test.pkl","rb"))
y_train = load(open("_4_Machine_Learning/keras_5_classification/Runs/y_train.pkl","rb"))
y_test = load(open("_4_Machine_Learning/keras_5_classification/Runs/y_test.pkl","rb"))


### ---------  --------- ###
predictions = (model.predict(X_test) > 0.5).astype("int32")
# print(f"predictions: {predictions}")

# print(f"y_test: {y_test}")


# compare the predictions to the actual values of the test data set
# print(classification_report(y_test, predictions))
#               precision    recall  f1-score   support

#            0       0.98      0.99      0.98        88
#            1       0.98      0.96      0.97        55

#     accuracy                           0.98       143
#    macro avg       0.98      0.98      0.98       143
# weighted avg       0.98      0.98      0.98       143

# print(confusion_matrix(y_test, predictions))
# [[87  1]
#  [ 2 53]]

# the confusion matrix showst the network only misclassified 1 datapoint 








































