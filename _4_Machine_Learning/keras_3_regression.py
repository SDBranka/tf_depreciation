# regression based project that predicts the price a house should sell 
# based upon the features of the house


# info about the data
# historical housing data from King County, Seatle, WA, USA
# id - unique ID for each home sold 
# date - Date of the home sale
# price - Price of each home sold
# bedrooms - Number of bedrooms
# bathrooms - Number of bathrooms (.5 indicates a room with a toilet by no shower)
# sqft_living - Square footage of the apartments interior living space
# sqft_lot - Square footage of the land space
# floors - Number of floors
# waterfront - A dummy variable for whether the apartment was overlooking the waterfront or not
# view - An index from 0 to 4 of how good the view of the property was
# condition - An index from 1 to 5 on the condition of the apartment
# grade - An index from 1 to 13, where 1-3 falls short of building construction and design, 7 is average, and 11-13 have a high quality level of construction and design
# sqft_above - The square footage of the interior housing space that is above ground level
# sqft_basement - The square footage of the interior housing space that is below ground level
# yr_built - The year the house was initially built
# yr_renovated - The year of the house's last renovation
# zipcode - What zipcode area the house is in
# lat - Latitude
# long - Longitude
# sqft_living15 - The square footage of the interior housing living space for the nearest 15 neighbors
# sqft_lot15 - The square footage of the land lots of the nearest 15 neighbors


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv("Data/kc_house_data.csv")


# explore the data
# view the first five entries
df_first_five = df.head()
# print(df_first_five)
#            id             date     price  bedrooms  ...      lat     long  sqft_living15  sqft_lot15
# 0  7129300520  20141013T000000  221900.0         3  ...  47.5112 -122.257           1340        5650
# 1  6414100192  20141209T000000  538000.0         3  ...  47.7210 -122.319           1690        7639
# 2  5631500400  20150225T000000  180000.0         2  ...  47.7379 -122.233           2720        8062
# 3  2487200875  20141209T000000  604000.0         4  ...  47.5208 -122.393           1360        5000
# 4  1954400510  20150218T000000  510000.0         3  ...  47.6168 -122.045           1800        7503


# see if there is any missing data
missing_data = df.isnull().sum()
# print(missing_data)
# id               0
# date             0
# price            0
# bedrooms         0
# bathrooms        0
# sqft_living      0
# sqft_lot         0
# floors           0
# waterfront       0
# view             0
# condition        0
# grade            0
# sqft_above       0
# sqft_basement    0
# yr_built         0
# yr_renovated     0
# zipcode          0
# lat              0
# long             0
# sqft_living15    0
# sqft_lot15       0
# dtype: int64

# this provides a sum of all of the missing data points per each column, 
# since all values are 0 we are missing no data

# get the statistical information on the data set
# transpose lists the stats as the columns
stats_df = df.describe().transpose()
# print(stats_df)
#                  count          mean           std  ...           50%           75%           max
# id             21613.0  4.580302e+09  2.876566e+09  ...  3.904930e+09  7.308900e+09  9.900000e+09
# price          21613.0  5.400881e+05  3.671272e+05  ...  4.500000e+05  6.450000e+05  7.700000e+06
# bedrooms       21613.0  3.370842e+00  9.300618e-01  ...  3.000000e+00  4.000000e+00  3.300000e+01
# bathrooms      21613.0  2.114757e+00  7.701632e-01  ...  2.250000e+00  2.500000e+00  8.000000e+00
# sqft_living    21613.0  2.079900e+03  9.184409e+02  ...  1.910000e+03  2.550000e+03  1.354000e+04
# sqft_lot       21613.0  1.510697e+04  4.142051e+04  ...  7.618000e+03  1.068800e+04  1.651359e+06
# floors         21613.0  1.494309e+00  5.399889e-01  ...  1.500000e+00  2.000000e+00  3.500000e+00
# waterfront     21613.0  7.541757e-03  8.651720e-02  ...  0.000000e+00  0.000000e+00  1.000000e+00
# view           21613.0  2.343034e-01  7.663176e-01  ...  0.000000e+00  0.000000e+00  4.000000e+00
# condition      21613.0  3.409430e+00  6.507430e-01  ...  3.000000e+00  4.000000e+00  5.000000e+00
# grade          21613.0  7.656873e+00  1.175459e+00  ...  7.000000e+00  8.000000e+00  1.300000e+01
# sqft_above     21613.0  1.788391e+03  8.280910e+02  ...  1.560000e+03  2.210000e+03  9.410000e+03
# sqft_basement  21613.0  2.915090e+02  4.425750e+02  ...  0.000000e+00  5.600000e+02  4.820000e+03
# yr_built       21613.0  1.971005e+03  2.937341e+01  ...  1.975000e+03  1.997000e+03  2.015000e+03
# yr_renovated   21613.0  8.440226e+01  4.016792e+02  ...  0.000000e+00  0.000000e+00  2.015000e+03
# zipcode        21613.0  9.807794e+04  5.350503e+01  ...  9.806500e+04  9.811800e+04  9.819900e+04
# lat            21613.0  4.756005e+01  1.385637e-01  ...  4.757180e+01  4.767800e+01  4.777760e+01
# long           21613.0 -1.222139e+02  1.408283e-01  ... -1.222300e+02 -1.221250e+02 -1.213150e+02
# sqft_living15  21613.0  1.986552e+03  6.853913e+02  ...  1.840000e+03  2.360000e+03  6.210000e+03
# sqft_lot15     21613.0  1.276846e+04  2.730418e+04  ...  7.620000e+03  1.008300e+04  8.712000e+05

# chart1
# show distribution plot of the prices across all the data
# sns.displot(df["price"])
# plt.title("Chart 1")
# plt.show()

# chart2
# print(df["bedrooms"])
# number of bedrooms
# sns.countplot(data=df, x="bedrooms")
# plt.title("Chart 2")
# plt.show()

# comparing your label to other points that you think may have a high
# correlation
print(df.corr())






























