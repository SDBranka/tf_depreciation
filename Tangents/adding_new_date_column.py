import numpy as np
import pandas as pd
from datetime import date


# read in data
df = pd.read_csv("Data/Original/build_volume_data_month.csv")

# build date array base on year and month
DATE = []
for y, m in zip(df["Engine Build Year (yyyy)"], df["Engine Build Month (MM)"]):
    DATE.append(date(y, m, 28))

# Create the new column
df['Month'] = DATE
# print(df.head())

# convert the elements of the column to a datetime object
df["Month"] = pd.to_datetime(df["Month"])
# df["Month"] = pd.to_numeric(df["Month"])

# rename Build Volume
df["Build_Volume"] = df["Build Volume"]
# print(df.head())

# get rid of orig year, month. and BV columns
df = df.drop(labels="Engine Build Year (yyyy)",axis=1)
df = df.drop(labels="Engine Build Month (MM)",axis=1)
df = df.drop(labels="Build Volume",axis=1)

# reorder the df to original order
# column_names = ["Month","Build_Volume"]

# df = df.reindex(columns=column_names)


print(df.head())
print(df.shape)
# save modified data to separate csv for futher use
df.to_csv("Data/monthly_bvs_dtobjects.csv", index=False)




















