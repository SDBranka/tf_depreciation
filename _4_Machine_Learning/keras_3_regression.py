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
# df = pd.read_csv("Data/kc_house_data1.csv")


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
# 1 means perfect correlation, -1 means perfectly inversely correlated, the
# closer to 0 the value the lower the correlation
# print(df.corr())
# ValueError: could not convert string to float: '20141013T000000'
# I have found a work around for this, but in the interest of this course
# I will not apply it here

# sort the columns correlation to price from -1 to 1
# print(df.corr()["price"].sort_values())
# ValueError: could not convert string to float: '20141013T000000'
# from course see output below:
# zipcode          -0.053402
# id               -0.016772
# long              0.022036
# condition         0.036056
# yr_built          0.053953
# sqft_lot15        0.082845
# sqft_lot          0.089876
# yr_renovated      0.126424
# floors            0.256804
# waterfront        0.266398
# lat               0.306692
# bedrooms          0.308787
# sqft_basement     0.323799
# view              0.397370
# bathrooms         0.525906
# sqft_living15     0.585241
# sqft_above        0.605368
# grade             0.667951
# sqft_living       0.701917
# price             1.000000
# Name: price, dtype: float64

# note that price, unsurprisingly has a 1:1 correlation, the next most 
# directly correlated is sqft_living


# explore features highly correlated to the label with a scatterplot
# chart3
# sns.scatterplot(x="price", y="sqft_living", data=df)
# plt.title("Chart 3")
# plt.show()

# chart4
# the distribution of prices per bedrooms
# sns.boxplot(x="bedrooms", y="price", data=df)
# plt.title("Chart 4")
# plt.show()

# distribution of prices per latitude vs longitude
# chart5
# first let's just look at the relationship to longitude
# sns.scatterplot(x="price", y="long", data=df)
# plt.title("Chart 5")
# plt.show()
# looks like longitude -122.2 is an expensive area

# chart6
# next with latitude
# sns.scatterplot(x="price", y="lat", data=df)
# plt.title("Chart 6")
# plt.show()
# here latitude 47.65-ish is the most expensive area

# it's not to difficult then to assume there is some sort of pricing
# hot spot where these two meet
# chart7
# make the chart larger to better see data
# plt.figure(figsize=(12,8))
# sns.scatterplot(x="long", y="lat", data=df)
# plt.title("Chart 7")
# plt.show()
# it's interesting how closely this plot lines up with a geographic map
# of the county

# let's attempt to hone in on this expensive area
# # chart8
# plt.figure(figsize=(12,8))
# sns.scatterplot(x="long", y="lat", data=df, hue="price")
# plt.title("Chart 8")
# plt.show()


# let's clean up some of the price data outliers 
# let's look at the 20 most expensive houses
twenty_most_expensive = df.sort_values("price", ascending=False).head(20)
# print(twenty_most_expensive)
#                id             date      price  bedrooms  ...      lat     long  sqft_living15  sqft_lot15
# 7252   6762700020  20141013T000000  7700000.0         6  ...  47.6298 -122.323           3940        8800
# 3914   9808700762  20140611T000000  7062500.0         5  ...  47.6500 -122.214           3930       25449
# 9254   9208900037  20140919T000000  6885000.0         6  ...  47.6305 -122.240           4540       42730
# 4411   2470100110  20140804T000000  5570000.0         5  ...  47.6289 -122.233           3560       24345
# 1448   8907500070  20150413T000000  5350000.0         5  ...  47.6232 -122.220           4600       21750
# 1315   7558700030  20150413T000000  5300000.0         6  ...  47.5631 -122.210           4320       24619
# 1164   1247600105  20141020T000000  5110800.0         5  ...  47.6767 -122.211           3430       26788
# 8092   1924059029  20140617T000000  4668000.0         5  ...  47.5570 -122.210           3270       10454
# 2626   7738500731  20140815T000000  4500000.0         5  ...  47.7493 -122.280           3030       23408
# 8638   3835500195  20140618T000000  4489000.0         4  ...  47.6208 -122.219           3720       14592
# 12370  6065300370  20150506T000000  4208000.0         5  ...  47.5692 -122.189           4740       19329
# 4149   6447300265  20141014T000000  4000000.0         4  ...  47.6151 -122.224           3140       15996
# 2085   8106100105  20141114T000000  3850000.0         4  ...  47.5850 -122.222           4620       22748
# 19017  2303900100  20140911T000000  3800000.0         3  ...  47.7296 -122.370           3430       45302
# 7035    853200010  20140701T000000  3800000.0         5  ...  47.6229 -122.220           5070       20570
# 16302  7397300170  20140530T000000  3710000.0         4  ...  47.6395 -122.234           2980       19602
# 6508   4217402115  20150421T000000  3650000.0         6  ...  47.6515 -122.277           3510       15810
# 18482  4389201095  20150511T000000  3650000.0         5  ...  47.6146 -122.213           4190       11275
# 15255  2425049063  20140911T000000  3640900.0         4  ...  47.6409 -122.241           3820       25582
# 19148  3625049042  20141011T000000  3635000.0         5  ...  47.6165 -122.236           2910       17600

# [20 rows x 21 columns]

# the most expensive house is $ 7,700,000.00, taking a look at the price 
# distribution chart (chart1) and the above data, it looks like the market
# drops off dramatically around $3,000,000

# to drop the expensive outliers
# print(len(df))
# 21613
# so there are 21,613 houses in our data set
# if we wanted to drop the top 1%
one_percent =  len(df) * 0.01
# build a df missing the top 1%
non_top_1_perc_df = df.sort_values("price", ascending=False).iloc[int(one_percent):]

# let's recheck the scatterplot
# chart9
# plt.figure(figsize=(12,8))
# sns.scatterplot(x="long", y="lat", data=non_top_1_perc_df, hue="price")
# plt.title("Chart 9")
# plt.show()
# now the distribution is a bit more clear
# let's get rid of the white edgecolor
# chart10
# plt.figure(figsize=(12,8))
# sns.scatterplot(x="long", y="lat", data=non_top_1_perc_df, hue="price", edgecolor=None)
# plt.title("Chart 10")
# plt.show()
# because we have so many colors stack on top of each other, we'll adjust the alpha
# we'll also adjust the color gradient
# chart11
# plt.figure(figsize=(12,8))
# sns.scatterplot(x="long", y="lat", data=non_top_1_perc_df, 
#                 edgecolor=None, alpha=0.2,
#                 # red/yellow/green gradient
#                 palette="RdYlGn", hue="price"
#                 )
# plt.title("Chart 11")
# plt.show()
# now this is a much better plot of where the most expensive homes in 
# king county are


# use a boxplot to see if the home is on the waterfront (using orig df)
# chart12
sns.boxplot(x="waterfront", y="price", data=df)
plt.title("Chart 12")
plt.show()




