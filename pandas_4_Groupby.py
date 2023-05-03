# I don't have access to the csv so some of this will be off

import numpy as np 
import pandas as pd


# seeding random to produce the repeatable values for tutorial purposes
np.random.seed(42)


df = pd.read_csv("Data/universities.csv")
# print(df.head())
#                                 Sector  ... Geography
# 0            Private for profit 2 year  ...    Nevada
# 1  Private for profit less than 2 year  ...    Nevada
# 2  Private for profit less than 2 year  ...    Nevada
# 3  Private for profit less than 2 year  ...    Nevada
# 4               Public 4 year or above  ...    Nevada

# [5 rows x 5 columns]

# see the total completions per year across all universities
# print(df.groupby("Year").sum("Completions"))
#       Completions
# Year
# 2016         2227

# see the mean completions per year across all universities
# print(df.groupby("Year").mean("Completions"))
#       Completions
# Year
# 2016        445.4

# see the total completions per year across all universities and present in 
# descending order
# print(df.groupby("Year").sum("Completions").sort_index(ascending=False))
#       Completions
# Year
# 2016         2227

# group by the year and then group by the sector and then sum completions
# print(df.groupby(["Year", "Sector"]).sum("Completions"))
#                                           Completions
# Year Sector
# 2016 Private for profit 2 year                    591
#      Private for profit less than 2 year          676
#      Public 4 year or above                       960
# 2017 Private for profit 2 year                    501
#      Private for profit less than 2 year          529
#      Public 4 year or above                       910


# get stats on a category
# print(df.groupby("Year").describe())
#      Completions
#            count   mean         std   min    25%    50%    75%    max
# Year
# 2016         5.0  445.4  354.902522  28.0  240.0  408.0  591.0  960.0
# 2017         5.0  388.0  339.737399  21.0  200.0  308.0  501.0  910.0
# or view another way
# print(df.groupby("Year").describe().transpose())
# Year                     2016        2017
# Completions count    5.000000    5.000000
#             mean   445.400000  388.000000
#             std    354.902522  339.737399
#             min     28.000000   21.000000
#             25%    240.000000  200.000000
#             50%    408.000000  308.000000
#             75%    591.000000  501.000000
#             max    960.000000  910.000000










