import numpy as np
import pandas as pd
from numpy.random import randint

labels = ["a", "b", "c"]

mylist = [10, 20, 30]

arr = np.array([10, 20, 30])

d = {"a": 10, "b": 20, "c": 30}

ser1 = pd.Series(data=mylist)
# print(f"ser1: \n{ser1}")
# ser1: 
# 0    10
# 1    20
# 2    30
# dtype: int64

ser1 = pd.Series(data=mylist, index=labels)
# print(f"ser1: \n{ser1}")
# ser1: 
# a    10
# b    20
# c    30
# dtype: int64

# convert dictionary to pandas series
ser1 = pd.Series(d)
# print(f"ser1: \n{ser1}")
# ser1:
# a    10
# b    20
# c    30
# dtype: int64


salesQ1 = pd.Series(data=[250, 450, 200, 150], index=["USA", "China", "India", "Brazil"])

salesQ2 = pd.Series(data=[260, 500, 210, 100], index=["USA", "China", "India", "Japan"])

# print(salesQ2["China"])
# 500

# print(salesQ2[0])
# 260

# print(salesQ1 + salesQ2)
# Brazil      NaN
# China     950.0
# India     410.0
# Japan       NaN
# USA       510.0
# dtype: float64




# seeding random to produce the repeatable values for tutorial purposes
np.random.seed(42)


columns = ["W", "X", "Y", "Z"]
index = ["A", "B", "C", "D", "E"]

# random numbers between -100 and 100 that is 5x4
data = randint(low=-100, high=100, size=(5, 4))
# print(f"data: \n{data}")
# data:
# [[  2  79  -8 -86]
#  [  6 -29  88 -80]
#  [  2  21 -26 -13]
#  [ 16  -1   3  51]
#  [ 30  49 -48 -99]]

df = pd.DataFrame(data=data, index=index, columns=columns)
# print(f"df: \n{df}")
# df:
#     W   X   Y   Z
# A   2  79  -8 -86
# B   6 -29  88 -80
# C   2  21 -26 -13
# D  16  -1   3  51
# E  30  49 -48 -99


# select a column
# print(f"df['W']: \n{df['W']}")
# df['W']:
# A     2
# B     6
# C     2
# D    16
# E    30
# Name: W, dtype: int32
# print(f"type(df['W']): \n{type(df['W'])}")
# type(df['W']):
# <class 'pandas.core.series.Series'>


# select a list of columns
df_list = df[["W", "Z"]]
# print(f"df_list: \n{df_list}")
# df_list:
#     W   Z
# A   2 -86
# B   6 -80
# C   2 -13
# D  16  51
# E  30 -99


# create a new column based off of old columns
df["new"] = df["W"] + df["Y"]
# print(f"df: \n{df}")
# df:
#     W   X   Y   Z  new
# A   2  79  -8 -86   -6
# B   6 -29  88 -80   94
# C   2  21 -26 -13  -24
# D  16  -1   3  51   19
# E  30  49 -48 -99  -18


# remove a column (axis= 0 is the rows, axis=1 is the columns)
df = df.drop(labels="new",axis=1)
# print(f"df: \n{df}")
# df:
#     W   X   Y   Z  
# A   2  79  -8 -86  
# B   6 -29  88 -80 
# C   2  21 -26 -13 
# D  16  -1   3  51 
# E  30  49 -48 -99 



# select a row
row_A = df.loc["A"]
# print(f"row_A: \n{row_A}")
# row_A:
# W     2
# X    79
# Y    -8
# Z   -86
# Name: A, dtype: int32

# select rows
row_AE = df.loc[["A", "E"]]
# print(f"row_AE: \n{row_AE}")
# row_AE:
#     W   X   Y   Z
# A   2  79  -8 -86
# E  30  49 -48 -99


# index location
row0 = df.iloc[0]
# print(f"row0: \n{row0}")
# row0:
# W     2
# X    79
# Y    -8
# Z   -86
# Name: A, dtype: int32


# delete a row
df_minus_C = df.drop("C")
# print(f"df_minus_C: \n{df_minus_C}")
# df_minus_C:
#     W   X   Y   Z
# A   2  79  -8 -86
# B   6 -29  88 -80
# D  16  -1   3  51
# E  30  49 -48 -99


# grab a cell value
aw = df.loc["A", "W"]
# print(f"aw: \n{aw}")
# aw:
# 2

aw = df.loc[["A", "C"], "W"]
# print(f"aw: \n{aw}")
# aw:
# A    2
# C    2
# Name: W, dtype: int32

aw = df.loc[["A", "C"], ["Y","W"]]
# print(f"aw: \n{aw}")
# aw:
#     Y  W
# A  -8  2
# C -26  2












