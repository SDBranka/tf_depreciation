import numpy as np 
import pandas as pd


### ---------   CSV IO   --------- ###
df = pd.read_csv("Data/example.csv")

# print(df)
#     a   b   c   d
# 0   0   1   2   3
# 1   4   5   6   7
# 2   8   9  10  11
# 3  12  13  14  15

# out the df as a csv without the index numbers
df.to_csv("Data/output.csv", index=False)


### ---------   HTML IO   --------- ###
tables = pd.read_html('https://www.fdic.gov/resources/resolutions/bank-failures/failed-bank-list/')

# print(type(tables))
# <class 'list'>

print("\n")

# print(tables[0])
#                          Bank NameBank  ... FundFund
# 0                  First Republic Bank  ...    10543
# 1                       Signature Bank  ...    10540
# 2                  Silicon Valley Bank  ...    10539
# 3                    Almena State Bank  ...    10538
# 4           First City Bank of Florida  ...    10537
# ..                                 ...  ...      ...
# 561                 Superior Bank, FSB  ...     6004
# 562                Malta National Bank  ...     4648
# 563    First Alliance Bank & Trust Co.  ...     4647
# 564  National State Bank of Metropolis  ...     4646
# 565                   Bank of Honolulu  ...     4645

# [566 rows x 7 columns]

tables[0].to_csv("Data/bank_failures.csv", index=False)


