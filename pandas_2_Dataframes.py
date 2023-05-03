import numpy as np 
import pandas as pd
from numpy.random import randint


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


### ---------   pt2   --------- ###

# check df
# print(f"df: \n{df}")
# df:
#     W   X   Y   Z
# A   2  79  -8 -86
# B   6 -29  88 -80
# C   2  21 -26 -13
# D  16  -1   3  51
# E  30  49 -48 -99


# conditional selection

# find where values are greater than 0
more_than_zero = df > 0
# print(f"more_than_zero: \n{more_than_zero}")
# more_than_zero:
#       W      X      Y      Z
# A  True   True  False  False
# B  True  False   True  False
# C  True   True  False  False
# D  True  False   True   True
# E  True   True  False  False

# bringing back values greater than 0
more_than_zero = df[df > 0]
# print(f"more_than_zero: \n{more_than_zero}")
# more_than_zero:
#     W     X     Y     Z
# A   2  79.0   NaN   NaN
# B   6   NaN  88.0   NaN
# C   2  21.0   NaN   NaN
# D  16   NaN   3.0  51.0
# E  30  49.0   NaN   NaN

# filter by condition in column "X"
colX = df["X"] > 0
# print(f"colX: \n{colX}")
# colX:
# A     True
# B    False
# C     True
# D    False
# E     True
# Name: X, dtype: bool

# retrieve rows of col X > 0
colX = df[df["X"] > 0]
# print(f"colX: \n{colX}")
# colX:
#     W   X   Y   Z
# A   2  79  -8 -86
# C   2  21 -26 -13
# E  30  49 -48 -99

# pulling further values from the above df
w = df[df["X"] > 0]["W"]
# print(f"w: \n{w}")
# w:
# A     2
# C     2
# E    30
# Name: W, dtype: int32
# or 
w = colX["W"]
# print(f"w: \n{w}")
# w:
# A     2
# C     2
# E    30
# Name: W, dtype: int32
# grab the first row
first_row = df[df["X"] > 0].iloc[0]
# print(f"first_row: \n{first_row}")
# first_row:
# W     2
# X    79
# Y    -8
# Z   -86
# Name: A, dtype: int32


# use two conditions
two_cond = df[(df["W"] > 0) & (df["Y"] > 1)]
# print(f"two_cond: \n{two_cond}")
# two_cond:
#     W   X   Y   Z
# B   6 -29  88 -80
# D  16  -1   3  51


# modify the index
# reset the row labels to numbers and create a new row that contains
# the labels for the rows
res_ind = df.reset_index()
# print(f"res_ind: \n{res_ind}")
# res_ind:
#   index   W   X   Y   Z
# 0     A   2  79  -8 -86
# 1     B   6 -29  88 -80
# 2     C   2  21 -26 -13
# 3     D  16  -1   3  51
# 4     E  30  49 -48 -99

# convert a column to our new index
# must have as many values as there are rows in the df
new_ind = ["CA", "NY", "WY", "OR", "CO"]
df["States"] = new_ind
new_df_ind = df.set_index("States")
# print(f"new_df_ind: \n{new_df_ind}")
# new_df_ind:
#          W   X   Y   Z
# States
# CA       2  79  -8 -86
# NY       6 -29  88 -80
# WY       2  21 -26 -13
# OR      16  -1   3  51
# CO      30  49 -48 -99

# return df to orig form
df = df.drop("States", axis=1)
# print(f"df: \n{df}")
# df:
#     W   X   Y   Z
# A   2  79  -8 -86
# B   6 -29  88 -80
# C   2  21 -26 -13
# D  16  -1   3  51
# E  30  49 -48 -99


# confirm column names
# print(f"column names: \n {new_df_ind.columns}")
# column names:
#  Index(['W', 'X', 'Y', 'Z'], dtype='object')


# get statistics on the df
# print(df.describe())
#               W          X          Y          Z
# count   5.00000   5.000000   5.000000   5.000000
# mean   11.20000  23.800000   1.800000 -45.400000
# std    11.96662  42.109381  51.915316  63.366395
# min     2.00000 -29.000000 -48.000000 -99.000000
# 25%     2.00000  -1.000000 -26.000000 -86.000000
# 50%     6.00000  21.000000  -8.000000 -80.000000
# 75%    16.00000  49.000000   3.000000 -13.000000
# max    30.00000  79.000000  88.000000  51.000000


