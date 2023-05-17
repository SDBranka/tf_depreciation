# to look at correlation between date and other features

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv("Data/kc_house_data.csv")
df1 = pd.read_csv("Data/kc_house_data1.csv")

# print(df.date)
df["date"] = pd.to_datetime(df["date"])
# print(df.date)
# 0       2014-10-13
# 1       2014-12-09
# 2       2015-02-25
# 3       2014-12-09
# 4       2015-02-18
#            ...
# 21608   2014-05-21
# 21609   2015-02-23
# 21610   2014-06-23
# 21611   2015-01-16
# 21612   2014-10-15
# Name: date, Length: 21613, dtype: datetime64[ns]


df1["date"] = pd.to_datetime(df1["date"])
# print(df1.date)
# 0       2014-10-13
# 1       2014-12-09
# 2       2015-02-25
# 3       2014-12-09
# 4       2015-02-18
#            ...
# 21608   2014-05-21
# 21609   2015-02-23
# 21610   2014-06-23
# 21611   2015-01-16
# 21612   2014-10-15
# Name: date, Length: 21613, dtype: datetime64[ns]

df["date"] = pd.to_numeric(df["date"])
# print(df.date)
# 0        1413158400000000000
# 1        1418083200000000000
# 2        1424822400000000000
# 3        1418083200000000000
# 4        1424217600000000000
#                 ...
# 21608    1400630400000000000
# 21609    1424649600000000000
# 21610    1403481600000000000
# 21611    1421366400000000000
# 21612    1413331200000000000
# Name: date, Length: 21613, dtype: int64


df1["date"] = pd.to_numeric(df1["date"])
# print(df1.date)
# 0        1413158400000000000
# 1        1418083200000000000
# 2        1424822400000000000
# 3        1418083200000000000
# 4        1424217600000000000
#                 ...
# 21608    1400630400000000000
# 21609    1424649600000000000
# 21610    1403481600000000000
# 21611    1421366400000000000
# 21612    1413331200000000000
# Name: date, Length: 21613, dtype: int64

# now we can correlate dates to other data
# print(df.corr())
#                      id      date     price  bedrooms  ...       lat      long  sqft_living15  sqft_lot15
# id             1.000000  0.005577 -0.016762  0.001286  ... -0.001891  0.020799      -0.002901   -0.138798
# date           0.005577  1.000000 -0.004357 -0.016800  ... -0.032856 -0.007020      -0.031515    0.002566
# price         -0.016762 -0.004357  1.000000  0.308350  ...  0.307003  0.021626       0.585379    0.082447
# bedrooms       0.001286 -0.016800  0.308350  1.000000  ... -0.008931  0.129473       0.391638    0.029244
# bathrooms      0.005160 -0.034410  0.525138  0.515884  ...  0.024573  0.223042       0.568634    0.087175
# sqft_living   -0.012258 -0.034559  0.702035  0.576671  ...  0.052529  0.240223       0.756420    0.183286
# sqft_lot      -0.132109  0.006313  0.089661  0.031703  ... -0.085683  0.229521       0.144608    0.718557
# floors         0.018525 -0.022491  0.256794  0.175429  ...  0.049614  0.125419       0.279885   -0.011269
# waterfront    -0.002721  0.001356  0.266369 -0.006582  ... -0.014274 -0.041910       0.086463    0.030703
# view           0.011592 -0.001800  0.397293  0.079532  ...  0.006157 -0.078400       0.280439    0.072575
# condition     -0.023783 -0.050769  0.036362  0.028472  ... -0.014941 -0.106500      -0.092824   -0.003406
# grade          0.008130 -0.039912  0.667434  0.356967  ...  0.114084  0.198372       0.713202    0.119248
# sqft_above    -0.010842 -0.027924  0.605567  0.477600  ... -0.000816  0.343803       0.731870    0.194050
# sqft_basement -0.005151 -0.019469  0.323816  0.303093  ...  0.110538 -0.144765       0.200355    0.017276
# yr_built       0.021380 -0.000355  0.054012  0.154178  ... -0.148122  0.409356       0.326229    0.070958
# yr_renovated  -0.016907 -0.024509  0.126434  0.018841  ...  0.029398 -0.068372      -0.002673    0.007854
# zipcode       -0.008224  0.001404 -0.053203 -0.152668  ...  0.267048 -0.564072      -0.279033   -0.147221
# lat           -0.001891 -0.032856  0.307003 -0.008931  ...  1.000000 -0.135512       0.048858   -0.086419
# long           0.020799 -0.007020  0.021626  0.129473  ... -0.135512  1.000000       0.334605    0.254451
# sqft_living15 -0.002901 -0.031515  0.585379  0.391638  ...  0.048858  0.334605       1.000000    0.183192
# sqft_lot15    -0.138798  0.002566  0.082447  0.029244  ... -0.086419  0.254451       0.183192    1.000000

# [21 rows x 21 columns]


# df1["date"] = pd.to_datetime(df1["date"])
# print(df1)
#                id       date     price  bedrooms  ...      lat     long  sqft_living15  sqft_lot15
# 0      7129300520 2014-10-13  221900.0         3  ...  47.5112 -122.257           1340        5650
# 1      6414100192 2014-12-09  538000.0         3  ...  47.7210 -122.319           1690        7639
# 2      5631500400 2015-02-25  180000.0         2  ...  47.7379 -122.233           2720        8062
# 3      2487200875 2014-12-09  604000.0         4  ...  47.5208 -122.393           1360        5000
# 4      1954400510 2015-02-18  510000.0         3  ...  47.6168 -122.045           1800        7503
# ...           ...        ...       ...       ...  ...      ...      ...            ...         ...
# 21608   263000018 2014-05-21  360000.0         3  ...  47.6993 -122.346           1530        1509
# 21609  6600060120 2015-02-23  400000.0         4  ...  47.5107 -122.362           1830        7200
# 21610  1523300141 2014-06-23  402101.0         2  ...  47.5944 -122.299           1020        2007
# 21611   291310100 2015-01-16  400000.0         3  ...  47.5345 -122.069           1410        1287
# 21612  1523300157 2014-10-15  325000.0         2  ...  47.5941 -122.299           1020        1357

# [21613 rows x 21 columns]