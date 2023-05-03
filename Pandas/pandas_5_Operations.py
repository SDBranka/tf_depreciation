import numpy as np 
import pandas as pd


df_one = pd.DataFrame({"k1": ["A","A","B","B","C","C"],
                    "col1": [100,200,300,300,400,500],
                    "col2": ["NY","CA","WA","WA","AK","NV"]
                    })
# print(df_one)
#   k1  col1 col2
# 0  A   100   NY
# 1  A   200   CA
# 2  B   300   WA
# 3  B   300   WA
# 4  C   400   AK
# 5  C   500   NV


# find unique values in a column
col1_uniq = df_one["col1"].unique()
# print(col1_uniq)
# [100 200 300 400 500]


# find number of unique values in a column
num_uniq_k1 = df_one["k1"].nunique()
# print(num_uniq_k1)
# 3


# get a count of how many times a unique value appears in a column
col2_uniq_count = df_one["col2"].value_counts()
# print(col2_uniq_count)
# col2
# WA    2
# NY    1
# CA    1
# AK    1
# NV    1
# Name: count, dtype: int64


# drop duplicates
# once the program finds a unique row of values it drops all the rest
drop_dup = df_one.drop_duplicates()
# print(drop_dup)
#   k1  col1 col2
# 0  A   100   NY
# 1  A   200   CA
# 2  B   300   WA
# 4  C   400   AK
# 5  C   500   NV


# grab the first letter of the values of a column and turn it into a
# new column

# define a function
def grab_first_letter(state):
    return state[0]

# print(grab_first_letter("NY"))
# N

# call the function for use on a column
first_lets = df_one["col2"].apply(grab_first_letter)
# print(first_lets)
# 0    N
# 1    C
# 2    W
# 3    W
# 4    A
# 5    N
# Name: col2, dtype: object

# create new column of first letters
df_one["First Letter"] = df_one["col2"].apply(grab_first_letter)
# print(df_one)
#   k1  col1 col2 First Letter
# 0  A   100   NY            N
# 1  A   200   CA            C
# 2  B   300   WA            W
# 3  B   300   WA            W
# 4  C   400   AK            A
# 5  C   500   NV            N



# ex2
#
def complex_letter(state):
    if state[0] == "W":
        return "Washington"
    else:
        return "Error"


# print(df_one["col2"].apply(complex_letter))
# 0         Error
# 1         Error
# 2    Washington
# 3    Washington
# 4         Error
# 5         Error
# Name: col2, dtype: object



# mapping
# print(df_one["k1"])
# 0    A
# 1    A
# 2    B
# 3    B
# 4    C
# 5    C
# Name: k1, dtype: object

# convert the str values to integers
my_map = {"A":1,
        "B":2,
        "C":3
        }

df_one['num'] = df_one["k1"].map(my_map)
# print(df_one)
#   k1  col1 col2 First Letter  num
# 0  A   100   NY            N    1
# 1  A   200   CA            C    1
# 2  B   300   WA            W    2
# 3  B   300   WA            W    2
# 4  C   400   AK            A    3
# 5  C   500   NV            N    3



# find the index of a max value in a column
# print(df_one["col1"].idxmax())
# 5


# permanently change the labels of the columns
df_one.columns = ["c1","c2","c3","c4","c5"]
# print(df_one)
#   c1   c2  c3 c4  c5
# 0  A  100  NY  N   1
# 1  A  200  CA  C   1
# 2  B  300  WA  W   2
# 3  B  300  WA  W   2
# 4  C  400  AK  A   3
# 5  C  500  NV  N   3


# sort df
df_one = df_one.sort_values("c3")
# print(df_one)
#   c1   c2  c3 c4  c5
# 4  C  400  AK  A   3
# 1  A  200  CA  C   1
# 5  C  500  NV  N   3
# 0  A  100  NY  N   1
# 2  B  300  WA  W   2
# 3  B  300  WA  W   2


# concantenating df's
features= pd.DataFrame({"A":[100,200,300,400,500],
                        "B":[12,13,14,15,16]
                        })
predictions = pd.DataFrame({"pred":[0,1,1,0,1]})

# adds as new rows
# cond_df = pd.concat([features, predictions])
# print(cond_df)
#        A     B  pred
# 0  100.0  12.0   NaN
# 1  200.0  13.0   NaN
# 2  300.0  14.0   NaN
# 3  400.0  15.0   NaN
# 4  500.0  16.0   NaN
# 0    NaN   NaN   0.0
# 1    NaN   NaN   1.0
# 2    NaN   NaN   1.0
# 3    NaN   NaN   0.0
# 4    NaN   NaN   1.0


# adds as new columns
cond_df = pd.concat([features, predictions], axis=1)
# print(cond_df)
#      A   B  pred
# 0  100  12     0
# 1  200  13     1
# 2  300  14     1
# 3  400  15     0
# 4  500  16     1


# create dummy variables
df_one = df_one.sort_index()

# print(df_one)
#   c1   c2  c3 c4  c5
# 0  A  100  NY  N   1
# 1  A  200  CA  C   1
# 2  B  300  WA  W   2
# 3  B  300  WA  W   2
# 4  C  400  AK  A   3
# 5  C  500  NV  N   3


# print(df_one["c1"])
# 0    A
# 1    A
# 2    B
# 3    B
# 4    C
# 5    C
# Name: c1, dtype: object


# onehot encoding
print(pd.get_dummies(df_one["c1"]))
# 0   True  False  False
# 1   True  False  False
# 2  False   True  False
# 3  False   True  False
# 4  False  False   True
# 5  False  False   True


