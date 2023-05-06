import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


Month = ["January","February","March","April","May","June"]
Card = [4,7,11,32,96,704]

month = np.array(Month)
card = np.array(Card)

month = pd.Series(month)
card = pd.Series(card)

test_df = pd.concat([month,card], axis=1)
test_df.columns = ["Month", "Card"]
# print(test_df)
#       Month  Card
# 0   January     4
# 1  February     7
# 2     March    11
# 3     April    32
# 4       May    96
# 5      June   704



