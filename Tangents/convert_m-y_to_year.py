import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# read data
df = pd.read_csv("Data/Orig/build_volume_data_month.csv")



# print(df.groupby("Engine Build Year (yyyy)").sum("Build Volume").sort_index(ascending=False))

df = df.groupby("Engine Build Year (yyyy)").sum("Build Volume")
# print(df)

df = df.reset_index()
print(df)

df = df.drop(labels="Engine Build Month (MM)", axis=1)
# print(df)



# save modified data to separate csv for futher use
df.to_csv("Data/year_build_volume_data.csv", index=False)










