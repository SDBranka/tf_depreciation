import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split               # to split the data into sets 


# read in data
df = pd.read_csv("Data/Orig/build_volume_data_month.csv")
# print(df.head())
#    Engine Build Year (yyyy)  Engine Build Month (MM)  Build Volume
# 0                      2000                        1          3945
# 1                      2000                        2          3631
# 2                      2000                        3          4248
# 3                      2000                        4          3144
# 4                      2000                        5          3360

# shuffle data
df = df.reindex(np.random.permutation(df.index))
# print(df.head())
#      Engine Build Year (yyyy)  Engine Build Month (MM)  Build Volume
# 119                      2009                       12          9980
# 238                      2019                       11          7133
# 140                      2011                        9          9812
# 167                      2013                       12          5753
# 177                      2014                       10         11576

df = df.reset_index()
# print(df.head())
df = df.drop(labels="index",axis=1)
# print(df.head())

# split the set 
test_df = df.iloc[:28]
train_df = df.iloc[28:]

# save modified data to separate csv for futher use
test_df.to_csv("Data/month_test.csv", index=False)
train_df.to_csv("Data/month_train.csv", index=False)





