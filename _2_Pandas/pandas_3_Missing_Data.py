import numpy as np 
import pandas as pd


# Three ways to deal with missing data
# - leave it as missing
# - remove the missing data
# - fill in the missing data


df = pd.DataFrame({"A": [1, 2, np.nan, 4],
                    "B": [5, np.nan, np.nan, 8],
                    "C": [10, 20, 30, 40]
                    })
# print(f"df: \n{df}")
# df:
#      A    B   C
# 0  1.0  5.0  10
# 1  2.0  NaN  20
# 2  NaN  NaN  30
# 3  4.0  8.0  40

### ---------   remove nan values   --------- ###
# drop rows with nan values
no_nan_rows = df.dropna()
# print(f"no_nan_rows: \n{no_nan_rows}")
# no_nan_rows:
#      A    B   C
# 0  1.0  5.0  10
# 3  4.0  8.0  40


# drop cols with nan values
no_nan_cols = df.dropna(axis=1)
# print(f"no_nan_cols: \n{no_nan_cols}")
# no_nan_cols:
#     C
# 0  10
# 1  20
# 2  30
# 3  40

# using a threshold value
# set threshold to 25% of the values / col must not be nan
th = 0.75 * len(df)
th_df = df.dropna(axis=1,thresh=th)
# print(f"th_df: \n{th_df}")
# th_df:
#      A   C
# 0  1.0  10
# 1  2.0  20
# 2  NaN  30
# 3  4.0  40


### ---------   fill in nan values   --------- ###

# fill nan's with a string
str_df = df.fillna(value="FILL VALUE")
# print(f"str_df: \n{str_df}")
# str_df:
#             A           B   C
# 0         1.0         5.0  10
# 1         2.0  FILL VALUE  20
# 2  FILL VALUE  FILL VALUE  30
# 3         4.0         8.0  40

# fill nan's with a value
num_fill_df = df.fillna(value=0)
# print(f"num_fill_df: \n{num_fill_df}")
# num_fill_df:
#      A    B   C
# 0  1.0  5.0  10
# 1  2.0  0.0  20
# 2  0.0  0.0  30
# 3  4.0  8.0  40

# fill nan's with a value in a column
fill_df = df["A"].fillna(value=0)
# print(f"fill_df: \n{fill_df}")
# fill_df:
# 0    1.0
# 1    2.0
# 2    0.0
# 3    4.0
# Name: A, dtype: float64

# permanently change the value in the df
# df["A"] = df["A"].fillna(value=0)
# print(f"df: \n{df}")
# df:
#      A    B   C
# 0  1.0  5.0  10
# 1  2.0  NaN  20
# 2  0.0  NaN  30
# 3  4.0  8.0  40

# fill nan values in a column with the mean of the column
# print(df["B"].fillna(value=df["B"].mean()))
# 0    5.0
# 1    6.5
# 2    6.5
# 3    8.0
# Name: B, dtype: float64

# fill all nan values with mean values by column
# print(df.fillna(df.mean()))
#           A    B   C
# 0  1.000000  5.0  10
# 1  2.000000  6.5  20
# 2  2.333333  6.5  30
# 3  4.000000  8.0  40


