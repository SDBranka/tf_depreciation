import numpy as np
import pandas as pd


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


