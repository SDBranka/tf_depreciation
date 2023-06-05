# Early Stopping 
# Keras can automatically stop training based on a loss condition on the 
# validation data passed during the model.fit() call.

# Dropout Layers
# Dropout can be added to layers to "turn off" neurons during training to 
# prevent overfitting
# Each dropout layer will "drop" a user-defined percentage of neuron units in 
# the previous layer every batch

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pickle import dump                                           # to save the scaler           
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout
from tensorflow.keras.callbacks import EarlyStopping


# relax the display limits of columns and rows
# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)


### --------- Data --------- ###
df = pd.read_csv("Data/cancer_classification_diagnosis_numeric.csv")


# check for null values
# print(df.info())
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 569 entries, 0 to 568
# Data columns (total 33 columns):
#  #   Column                   Non-Null Count  Dtype
# ---  ------                   --------------  -----
#  0   id                       569 non-null    int64
#  1   diagnosis                569 non-null    object
#  2   radius_mean              569 non-null    float64
#  3   texture_mean             569 non-null    float64
#  4   perimeter_mean           569 non-null    float64
#  5   area_mean                569 non-null    float64
#  6   smoothness_mean          569 non-null    float64
#  7   compactness_mean         569 non-null    float64
#  8   concavity_mean           569 non-null    float64
#  9   concave points_mean      569 non-null    float64
#  10  symmetry_mean            569 non-null    float64
#  11  fractal_dimension_mean   569 non-null    float64
#  12  radius_se                569 non-null    float64
#  13  texture_se               569 non-null    float64
#  14  perimeter_se             569 non-null    float64
#  15  area_se                  569 non-null    float64
#  16  smoothness_se            569 non-null    float64
#  17  compactness_se           569 non-null    float64
#  18  concavity_se             569 non-null    float64
#  19  concave points_se        569 non-null    float64
#  20  symmetry_se              569 non-null    float64
#  21  fractal_dimension_se     569 non-null    float64
#  22  radius_worst             569 non-null    float64
#  23  texture_worst            569 non-null    float64
#  24  perimeter_worst          569 non-null    float64
#  25  area_worst               569 non-null    float64
#  26  smoothness_worst         569 non-null    float64
#  27  compactness_worst        569 non-null    float64
#  28  concavity_worst          569 non-null    float64
#  29  concave points_worst     569 non-null    float64
#  30  symmetry_worst           569 non-null    float64
#  31  fractal_dimension_worst  569 non-null    float64
#  32  Unnamed: 32              0 non-null      float64

# drop "Unnamed: 32" columns
df = df.drop(["Unnamed: 32"], axis=1)
# print(df.head())
#          id  diagnosis  radius_mean  texture_mean  ...  concavity_worst  concave points_worst  symmetry_worst  fractal_dimension_wo
# rst
# 0    842302          1        17.99         10.38  ...           0.7119                0.2654          0.4601                  0.11
# 890
# 1    842517          1        20.57         17.77  ...           0.2416                0.1860          0.2750                  0.08
# 902
# 2  84300903          1        19.69         21.25  ...           0.4504                0.2430          0.3613                  0.08
# 758
# 3  84348301          1        11.42         20.38  ...           0.6869                0.2575          0.6638                  0.17
# 300
# 4  84358402          1        20.29         14.34  ...           0.4000                0.1625          0.2364                  0.07
# 678

# [5 rows x 32 columns]


# print(df.describe())
#                  id   diagnosis  radius_mean  ...  concave points_worst  symmetry_worst  fractal_dimension_worst
# count  5.690000e+02  569.000000   569.000000  ...            569.000000      569.000000               569.000000
# mean   3.037183e+07    0.372583    14.127292  ...              0.114606        0.290076                 0.083946
# std    1.250206e+08    0.483918     3.524049  ...              0.065732        0.061867                 0.018061
# min    8.670000e+03    0.000000     6.981000  ...              0.000000        0.156500                 0.055040
# 25%    8.692180e+05    0.000000    11.700000  ...              0.064930        0.250400                 0.071460
# 50%    9.060240e+05    0.000000    13.370000  ...              0.099930        0.282200                 0.080040
# 75%    8.813129e+06    1.000000    15.780000  ...              0.161400        0.317900                 0.092080
# max    9.113205e+08    1.000000    28.110000  ...              0.291000        0.663800                 0.207500

# [8 rows x 32 columns]


# chart1
# create a countplot of the label to see if this is a well balanced problem or not
# sns.countplot(x="diagnosis", data=df)
# plt.title("Chart_1_Label_Countplot")
# plt.show()


# check the correlation between features
# print(df.corr())
# check the correlation of the features to the label, sort from most inversely 
# correlated to most positively correlated
# print(df.corr()["diagnosis"].sort_values())
# smoothness_se             -0.067016
# fractal_dimension_mean    -0.012838
# texture_se                -0.008303
# symmetry_se               -0.006522
# id                         0.039769
# fractal_dimension_se       0.077972
# concavity_se               0.253730
# compactness_se             0.292999
# fractal_dimension_worst    0.323872
# symmetry_mean              0.330499
# smoothness_mean            0.358560
# concave points_se          0.408042
# texture_mean               0.415185
# symmetry_worst             0.416294
# smoothness_worst           0.421465
# texture_worst              0.456903
# area_se                    0.548236
# perimeter_se               0.556141
# radius_se                  0.567134
# compactness_worst          0.590998
# compactness_mean           0.596534
# concavity_worst            0.659610
# concavity_mean             0.696360
# area_mean                  0.708984
# radius_mean                0.730029
# area_worst                 0.733825
# perimeter_mean             0.742636
# radius_worst               0.776454
# concave points_mean        0.776614
# perimeter_worst            0.782914
# concave points_worst       0.793566
# diagnosis                  1.000000
# Name: diagnosis, dtype: float64

# chart2
# plot the correlation between the label and features as a bar graph 
# df.corr()["diagnosis"].sort_values().plot(kind="bar")
# plt.title("Chart_2_Label_Correlation")
# plt.show()


# chart3
# sns heatmap to display correlation
# sns.heatmap(df.corr())
# plt.title("Chart_3_Heatmap")
# plt.show()



### --------- Train/Test Split --------- ###
X = df.drop("diagnosis", axis=1).values
y = df["diagnosis"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, 
                                                    random_state=101
                                                    )

# scale the data
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
# remember to not fit the testing data to prevent data leakage
X_test = scaler.transform(X_test)


print(X_train.shape)
# (426, 31)


# ### --------- Build Model --------- ###
# model = Sequential()
# model.add(Dense(31, activation="relu"))
# model.add(Dense(16, activation="relu"))

# # sigmoid because this is a binary classification problem
# model.add(Dense(1, activation="sigmoid"))

# # model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
# model.compile(optimizer="adam", loss="binary_crossentropy")


# ### --------- Train Model --------- ###
# model.fit(x=X_train, y=y_train, epochs=600, validation_data=(X_test, y_test))



### --------- Evaluate Model --------- ###
# losses = pd.DataFrame(model.history.history)
# print(f"losses:\n{losses}")
# losses:
#          loss  val_loss
# 0    0.688833  0.683526
# 1    0.668803  0.664188
# 2    0.648154  0.640450
# 3    0.624063  0.612542
# 4    0.593172  0.577298
# ..        ...       ...
# 595  0.011780  0.366273
# 596  0.011264  0.365174
# 597  0.011702  0.381255
# 598  0.012689  0.367213
# 599  0.010860  0.383299

# [600 rows x 2 columns]

# chart4
# plot losses
# losses.plot()
# plt.title("Chart_4_Losses")
# plt.show()
# see Chart_4_Losses-OverFit.png to see classic evidence of having trained the 
# model with too many epochs


# ### --------- Build Model 2 --------- ###
# model = Sequential()
# model.add(Dense(31, activation="relu"))
# model.add(Dense(16, activation="relu"))

# # sigmoid because this is a binary classification problem
# model.add(Dense(1, activation="sigmoid"))

# # model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
# model.compile(optimizer="adam", loss="binary_crossentropy")


# ### --------- Train Model 2 --------- ###
# # read about EarlyStopping
# # help(EarlyStopping)
# # early_stop = EarlyStopping(monitor="val_loss", patience=10, verbose=1)
# early_stop = EarlyStopping(monitor="val_loss", mode=min, patience=25, verbose=1)


# model.fit(x=X_train, y=y_train, epochs=600, 
#             validation_data=(X_test, y_test),
#             callbacks=[early_stop]
#             )

# ### --------- Evaluate Model 2 --------- ###
# losses = pd.DataFrame(model.history.history)
# # print(f"losses:\n{losses}")


# # chart5
# # plot losses
# # losses.plot()
# # plt.title("Chart_5_Losses_After_EarlyStopping")
# # plt.show()


### --------- Dropout Layers --------- ###

### --------- Build Model 2 --------- ###
model = Sequential()

model.add(Dense(31, activation="relu"))
model.add(Dropout(rate=0.5))
model.add(Dense(16, activation="relu"))
model.add(Dropout(rate=0.5))

# sigmoid because this is a binary classification problem
model.add(Dense(1, activation="sigmoid"))

# model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.compile(optimizer="adam", loss="binary_crossentropy")


### --------- Train Model 2 --------- ###
early_stop = EarlyStopping(monitor="val_loss", mode=min, patience=25, verbose=1)

model.fit(x=X_train, y=y_train, epochs=600, 
            validation_data=(X_test, y_test),
            callbacks=[early_stop]
            )

### --------- Evaluate Model 2 --------- ###
losses = pd.DataFrame(model.history.history)
# print(f"losses:\n{losses}")


# chart6
# plot losses
# losses.plot()
# plt.title("Chart_6_Losses_After_EarlyStopping_and_Dropout")
# plt.show()




### --------- Save Model ------- ###
# save the model as a .h5 file
model.save("_4_Machine_Learning/keras_5_classification/Runs/model.h5")

# save the scaler
dump(scaler,open("_4_Machine_Learning/keras_5_classification/Runs/scaler.pkl","wb"))

# save the dataframe and data sets
# x_train, x_test, y_train, y_test
dump(df,open("_4_Machine_Learning/keras_5_classification/Runs/df.pkl","wb"))
dump(X_train,open("_4_Machine_Learning/keras_5_classification/Runs/X_train.pkl","wb"))
dump(X_test,open("_4_Machine_Learning/keras_5_classification/Runs/X_test.pkl","wb"))
dump(y_train,open("_4_Machine_Learning/keras_5_classification/Runs/y_train.pkl","wb"))
dump(y_test,open("_4_Machine_Learning/keras_5_classification/Runs/y_test.pkl","wb"))















