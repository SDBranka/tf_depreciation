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
from sklearn.model_selection import train_test_split               # to split the data into sets 
from sklearn.preprocessing import MinMaxScaler                     # to scale data
from tensorflow.keras.models import Sequential                     # to build the model
from tensorflow.keras.layers import Dense                          # to build the model


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
# sns.boxplot(x="waterfront", y="price", data=df)
# plt.title("Chart 12")
# plt.show()



# let's take another look at the dataframe
# print(df.head())
#            id             date     price  bedrooms  ...      lat     long  sqft_living15  sqft_lot15
# 0  7129300520  20141013T000000  221900.0         3  ...  47.5112 -122.257           1340        5650
# 1  6414100192  20141209T000000  538000.0         3  ...  47.7210 -122.319           1690        7639
# 2  5631500400  20150225T000000  180000.0         2  ...  47.7379 -122.233           2720        8062
# 3  2487200875  20141209T000000  604000.0         4  ...  47.5208 -122.393           1360        5000
# 4  1954400510  20150218T000000  510000.0         3  ...  47.6168 -122.045           1800        7503

# [5 rows x 21 columns]

# since the id has no relation to the price of the house, let's get rid 
# of the id column (axis=1 to remove a column as opposed to a row)
df = df.drop("id", axis=1)
# print(df.head())
#               date     price  bedrooms  bathrooms  ...      lat     long  sqft_living15  sqft_lot15
# 0  20141013T000000  221900.0         3       1.00  ...  47.5112 -122.257           1340        5650
# 1  20141209T000000  538000.0         3       2.25  ...  47.7210 -122.319           1690        7639
# 2  20150225T000000  180000.0         2       1.00  ...  47.7379 -122.233           2720        8062
# 3  20141209T000000  604000.0         4       3.00  ...  47.5208 -122.393           1360        5000
# 4  20150218T000000  510000.0         3       2.00  ...  47.6168 -122.045           1800        7503

# [5 rows x 20 columns]

# convert the date column into a form we can work with, a datetime object
# now we can perform feature engineering on it
df["date"] = pd.to_datetime(df["date"])
# print(df.head())
#         date     price  bedrooms  bathrooms  ...      lat     long  sqft_living15  sqft_lot15
# 0 2014-10-13  221900.0         3       1.00  ...  47.5112 -122.257           1340        5650
# 1 2014-12-09  538000.0         3       2.25  ...  47.7210 -122.319           1690        7639
# 2 2015-02-25  180000.0         2       1.00  ...  47.7379 -122.233           2720        8062
# 3 2014-12-09  604000.0         4       3.00  ...  47.5208 -122.393           1360        5000
# 4 2015-02-18  510000.0         3       2.00  ...  47.6168 -122.045           1800        7503

# [5 rows x 20 columns]

# to show it's a datetime object
# print(df["date"].head())
# 0   2014-10-13
# 1   2014-12-09
# 2   2015-02-25
# 3   2014-12-09
# 4   2015-02-18
# Name: date, dtype: datetime64[ns]

# let's grab the year from these objects and create a new column
df["year"] = df["date"].apply(lambda date: date.year)
# create a month column
df["month"] = df["date"].apply(lambda date: date.month)

# print(df.head())
#         date     price  bedrooms  bathrooms  sqft_living  ...     long  sqft_living15  sqft_lot15  year  mo
# nth
# 0 2014-10-13  221900.0         3       1.00         1180  ... -122.257           1340        5650  2014
#  10
# 1 2014-12-09  538000.0         3       2.25         2570  ... -122.319           1690        7639  2014
#  12
# 2 2015-02-25  180000.0         2       1.00          770  ... -122.233           2720        8062  2015
#   2
# 3 2014-12-09  604000.0         4       3.00         1960  ... -122.393           1360        5000  2014
#  12
# 4 2015-02-18  510000.0         3       2.00         1680  ... -122.045           1800        7503  2015
#   2

# [5 rows x 22 columns]


# now we can explore if/how the year and month effect the price of these houses
# look at the distribution by month
# chart13
# sns.boxplot(x="month", y="price", data=df)
# plt.title("Chart 13")
# plt.show()

# it's hard to see from this plot whether there are any significant 
# distribution differences between the months the homes are sold by
# so let's look at the numbers
# look at the average home sale price by month
avg_price_by_month = df.groupby("month").mean()["price"]
# print(avg_price_by_month)
# 1     525870.889571
# 2     507851.371200
# 3     543977.187200
# 4     561837.774989
# 5     550768.785833
# 6     558002.199541
# 7     544788.764360
# 8     536445.276804
# 9     529253.821871
# 10    539026.971778
# 11    521961.009213
# 12    524461.866757
# Name: price, dtype: float64

# we can look at this on a graph
# chart14
# avg_price_by_month.plot()
# plt.title("Chart 14")
# plt.show()

# explore by year
avg_price_by_year = df.groupby("year").mean()["price"]

# chart15
# avg_price_by_year.plot()
# plt.title("Chart 15")
# plt.show()
# this graph indicates that the home prices increase every year

# because we don't plan to use the day in this case and we have extracted the
# month and year to their own columns let's drop the original date column
df = df.drop("date", axis=1)
# look at the remaining columns
# print(df.columns)
# Index(['price', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
#        'waterfront', 'view', 'condition', 'grade', 'sqft_above',
#        'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode', 'lat', 'long',
#        'sqft_living15', 'sqft_lot15', 'year', 'month'],
#       dtype='object')


# let's take at zipcode
# because it is written as a number, the program will treat these as numerical 
# values rather than categorical ones (ie 98174 is greater than 90828) if
# we just feed them in as is

# to see if we need to adjust this let's explore the zipcode column
# see the distribution of sales across zipcodes
# print(df["zipcode"].value_counts())
# zipcode
# 98103    602
# 98038    590
# 98115    583
# 98052    574
# 98117    553
#         ...
# 98102    105
# 98010    100
# 98024     81
# 98148     57
# 98039     50
# Name: count, Length: 70, dtype: int64
# this shows us that there are 70 different zipcodes included in the dataset
# because there are so many, for this case we will just drop the zipcode
# column. In a more realistic situations other options would include 
# grouping these into smaller sets (a group of 10(otl) neighboring zipcodes;
# North, East, South, West areas of the county; etc). This is where domain 
# knowledge (understanding of the topic) comes into play
df = df.drop("zipcode", axis=1)

# another feature that may cause an issue in this case is yr_renovated so 
# so let's explore that
# print(df["yr_renovated"].value_counts())
# yr_renovated
# 0       20699
# 2014       91
# 2013       37
# 2003       36
# 2005       35
#         ...
# 1951        1
# 1959        1
# 1948        1
# 1954        1
# 1944        1
# Name: count, Length: 70, dtype: int64
# it appears that the more recently a house was renovated, the more likely 
# it was sold. 0 indicates that the home has not been renovated. It may
# be more useful to split this column into renovated vs not renovated 
# In this case we will leave it as is


# let's look at sqft_basement
# print(df["sqft_basement"].value_counts())
# sqft_basement
# 0      13126
# 600      221
# 700      218
# 500      214
# 800      206
#        ...
# 518        1
# 374        1
# 784        1
# 906        1
# 248        1
# Name: count, Length: 306, dtype: int64
# 0 most likely indicates that the home does not have a basement



# after we've finished all the feature engineering, the next step 
# is to separate the features from the label

# establish the label
# use .values so that returns/stores the series as a numpy array
# begin a df of just the features by dropping the label from the orig
x = df.drop("price", axis=1).values
y = df["price"].values

# split the data into training/testing sets
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3,
                                                    random_state=101
                                                    )

# once the data is split we scale the data (remember we only do this on the 
# training set to prevent data leakage)

# create an instance of the scaler
scaler = MinMaxScaler()
# redefine the training data as a scaled version of the training data
# in this example we will fit and transform in one step as opposed to 2
# as in keras_1
x_train = scaler.fit_transform(x_train)
# scale the test data (remember not to fit this set to prevent data leakage)
x_test = scaler.transform(x_test)


# create the model
model = Sequential()
# typically we try to base the number of neurons in our layers on the size
# of the actual feature data
# so let's take a look at the shape of the data
# print(x_train.shape)
# (15129, 19)
# this shows we have 19 incoming features, so that's a good indicator
# that we should have 19 neurons in our input layer
model.add(Dense(19, activation="relu"))
# adding multiple hidden layers to make this a deep learning network
model.add(Dense(19, activation="relu"))
model.add(Dense(19, activation="relu"))
model.add(Dense(19, activation="relu"))
# adding too many may cause overfitting, but we can explore if that is
# happening later by feeding in validation data along with our training
# create the final output layer (1 neuron bc we only want 1 output value)
model.add(Dense(1))
# compile the model using the adam optimizer, since this is a regression 
# problem we're chosing a continuous(not categorical) label(price) for the 
# loss metric select we will select mean squared error 
model.compile(optimizer="adam",loss="mse")
# train the model
# for this case we will also use a validation data set to ensure that we are 
# not overfitting the model. after each epoch of the training data we'll 
# quickly run the test data and run the loss on the test data. This keeps a
# tracking of how well the model performs on data that is not from the training
# set. keras will not update the model (weight/biases) based on the results 
# of the validation data
# because the data set is so large we will also pass it in in batches
# it is most common to do batch sizes that are an even number
# the smaller the batch size the longer the training is going to take, but the
# less likely you are to overfit your data
model.fit(x=x_train, y=y_train, 
            validation_data=(x_test, y_test),
            batch_size = 128,
            epochs = 400
            )




































