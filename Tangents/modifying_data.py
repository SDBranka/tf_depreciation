import numpy as np 
import pandas as pd


df = pd.read_csv("Data/kc_house_data1.csv")


# grab the first letter of the values of a column and turn it into a
# new column

# define a function
def format_date(date):
    year =  date[0:4]
    month = date[4:6]
    day = date[6:8]
    formatted_date = f"{month}/{day}/{year}"
    return formatted_date

# print(format_date("20141013T000000"))
# 10/13/2014


# # call the function for use on a column
new_dates = df["date"].apply(format_date)
# print(new_dates)
# 0        10/13/2014
# 1        12/09/2014
# 2        02/25/2015
# 3        12/09/2014
# 4        02/18/2015
#             ...
# 21608    05/21/2014
# 21609    02/23/2015
# 21610    06/23/2014
# 21611    01/16/2015
# 21612    10/15/2014
# Name: date, Length: 21613, dtype: object

# drop the old date column
df = df.drop(labels="date",axis=1)

# create new column of formatted dates
df["date"] = new_dates
# print(df)
#                id     price  bedrooms  bathrooms  ...     long  sqft_living15  sqft_lot15        date
# 0      7129300520  221900.0         3       1.00  ... -122.257           1340        5650  10/13/2014
# 1      6414100192  538000.0         3       2.25  ... -122.319           1690        7639  12/09/2014
# 2      5631500400  180000.0         2       1.00  ... -122.233           2720        8062  02/25/2015
# 3      2487200875  604000.0         4       3.00  ... -122.393           1360        5000  12/09/2014
# 4      1954400510  510000.0         3       2.00  ... -122.045           1800        7503  02/18/2015
# ...           ...       ...       ...        ...  ...      ...            ...         ...         ...
# 21608   263000018  360000.0         3       2.50  ... -122.346           1530        1509  05/21/2014
# 21609  6600060120  400000.0         4       2.50  ... -122.362           1830        7200  02/23/2015
# 21610  1523300141  402101.0         2       0.75  ... -122.299           1020        2007  06/23/2014
# 21611   291310100  400000.0         3       2.50  ... -122.069           1410        1287  01/16/2015
# 21612  1523300157  325000.0         2       0.75  ... -122.299           1020        1357  10/15/2014

# [21613 rows x 21 columns]

# reorder the df to original order
column_names = ["id","date","price","bedrooms","bathrooms",
                "sqft_living","sqft_lot","floors","waterfront",
                "view","condition","grade","sqft_above","sqft_basement",
                "yr_built","yr_renovated","zipcode","lat","long",
                "sqft_living15","sqft_lot15"
                ]

df = df.reindex(columns=column_names)
# print(df)
#                id        date     price  bedrooms  ...      lat     long  sqft_living15  sqft_lot15
# 0      7129300520  10/13/2014  221900.0         3  ...  47.5112 -122.257           1340        5650
# 1      6414100192  12/09/2014  538000.0         3  ...  47.7210 -122.319           1690        7639
# 2      5631500400  02/25/2015  180000.0         2  ...  47.7379 -122.233           2720        8062
# 3      2487200875  12/09/2014  604000.0         4  ...  47.5208 -122.393           1360        5000
# 4      1954400510  02/18/2015  510000.0         3  ...  47.6168 -122.045           1800        7503
# ...           ...         ...       ...       ...  ...      ...      ...            ...         ...
# 21608   263000018  05/21/2014  360000.0         3  ...  47.6993 -122.346           1530        1509
# 21609  6600060120  02/23/2015  400000.0         4  ...  47.5107 -122.362           1830        7200
# 21610  1523300141  06/23/2014  402101.0         2  ...  47.5944 -122.299           1020        2007
# 21611   291310100  01/16/2015  400000.0         3  ...  47.5345 -122.069           1410        1287
# 21612  1523300157  10/15/2014  325000.0         2  ...  47.5941 -122.299           1020        1357

# [21613 rows x 21 columns]


# output as a new csv with no index
df.to_csv("Data/kc_house_data1.csv", index=False)
