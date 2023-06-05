# www.kaggle.com/wordsforthewise/lending-club

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


### --------- Data --------- ###
df = pd.read_csv("Data/keras_7/lending_club.csv")

# print(df.head())
#    loan_amnt  int_rate  installment  annual_inc    dti  ...  revol_bal  revol_util  total_acc  mort_acc  pub_rec_bankruptcies
# 0     3600.0     13.99       123.03     55000.0   5.91  ...     2765.0        29.7       13.0       1.0                   0.0
# 1    24700.0     11.99       820.28     65000.0  16.06  ...    21470.0        19.2       38.0       4.0                   0.0
# 2    20000.0     10.78       432.66     63000.0  10.78  ...     7869.0        56.2       18.0       5.0                   0.0
# 3    35000.0     14.85       829.90    110000.0  17.06  ...     7802.0        11.6       17.0       1.0                   0.0
# 4    10400.0     22.45       289.91    104433.0  25.37  ...    21929.0        64.5       35.0       6.0                   0.0

# [5 rows x 12 columns]


# describe the data
# print(df.describe())
#           loan_amnt    int_rate  installment     annual_inc  ...  revol_util   total_acc    mort_acc  pub_rec_bankruptcies
# count    636.000000  636.000000   636.000000     636.000000  ...  636.000000  636.000000  636.000000            636.000000
# mean   15702.712264   12.135189   456.649167   79789.118129  ...   49.749528   26.661950    1.930818              0.138365
# std     8570.002016    4.317393   251.679901   40665.658244  ...   23.082624   12.286111    2.053979              0.354552
# min     1000.000000    5.320000    32.970000   13000.000000  ...    0.000000    4.000000    0.000000              0.000000
# 25%     9000.000000    8.490000   268.765000   51600.000000  ...   32.675000   19.000000    0.000000              0.000000
# 50%    15000.000000   11.990000   398.020000   72000.000000  ...   47.950000   24.000000    1.000000              0.000000
# 75%    20612.500000   14.850000   602.300000  100000.000000  ...   65.750000   33.000000    3.000000              0.000000
# max    35000.000000   27.490000  1252.560000  300000.000000  ...  102.400000   89.000000   12.000000              2.000000

# describe revol_util
# print(df["revol_util"].describe())
# count    636.000000
# mean      49.749528
# std       23.082624
# min        0.000000
# 25%       32.675000
# 50%       47.950000
# 75%       65.750000
# max      102.400000
# Name: revol_util, dtype: float64


def feat_info(col_name):
    print(df[col_name].describe())

# feat_info("mort_acc")

# TODO 1: Since we will be attemptng to predict loan_status, create a 
# countplot 
# chart1
# sns.countplot(x="loan_status", data=df)
# plt.title("Chart_1_Loan_Status_Countplot")
# plt.show()

# TODO 2: create a histogram of the loan_amnt column
# chart2
# sns.distplot(df["loan_amnt"])
# plt.title("Chart_2_Loan_Amount_Distribution")
# plt.show()


# TODO 3: explore correlation between the continuous feature variables.
# Caluculate the correlation between all continuous numeric variables
# using .corr()
# print(df.corr())
#                       loan_amnt  int_rate  installment  annual_inc  ...  revol_util  total_acc  mort_acc  pub_rec_bankruptcies
# loan_amnt              1.000000  0.154156     0.941140    0.366457  ...    0.168804   0.163887  0.197731             -0.122995
# int_rate               0.154156  1.000000     0.130952   -0.092336  ...    0.234689   0.030481 -0.104266              0.119826
# installment            0.941140  0.130952     1.000000    0.350334  ...    0.179446   0.132957  0.165280             -0.111959
# annual_inc             0.366457 -0.092336     0.350334    1.000000  ...    0.140506   0.184706  0.400526             -0.033204
# dti                   -0.011006  0.219091    -0.006935   -0.255264  ...    0.141858   0.220344 -0.059804             -0.010956
# open_acc               0.151039  0.057491     0.125702    0.111216  ...   -0.135242   0.703880  0.082607              0.015777
# pub_rec               -0.107238  0.106479    -0.094257    0.023543  ...   -0.064141   0.000983 -0.076934              0.671751
# revol_bal              0.376096 -0.028855     0.366815    0.341036  ...    0.333936   0.189501  0.290795             -0.178119
# revol_util             0.168804  0.234689     0.179446    0.140506  ...    1.000000  -0.118898  0.064292             -0.108654
# total_acc              0.163887  0.030481     0.132957    0.184706  ...   -0.118898   1.000000  0.281452              0.049799
# mort_acc               0.197731 -0.104266     0.165280    0.400526  ...    0.064292   0.281452  1.000000             -0.066846
# pub_rec_bankruptcies  -0.122995  0.119826    -0.111959   -0.033204  ...   -0.108654   0.049799 -0.066846              1.000000

# [12 rows x 12 columns]


# TODO 4: Visualize this using a heatmap.
# chart3
# sns heatmap to display correlation
# sns.heatmap(df.corr())
# plt.title("Chart_3_Heatmap")
# plt.show()

# chart4
# sns heatmap to display correlation
# sns.heatmap(df.corr(), annot=True, cmap="viridis")
# plt.title("Chart_4_Heatmap")
# plt.show()

# You should have noticed almost perfect correlation with the installment feature.

# TODO 5: Explore this feature further. 
# Print out their descriptions and perform a scatterplot between them
# print(feat_info("installment"))
# count     636.000000
# mean      456.649167
# std       251.679901
# min        32.970000
# 25%       268.765000
# 50%       398.020000
# 75%       602.300000
# max      1252.560000
# Name: installment, dtype: float64


# chart5
# sns.scatterplot(x="installment", y="loan_amnt", data=df)
# plt.title("Chart_5_Installment_vs_Loan_Amount")
# plt.show()

# TODO 6: Create a boxplot showing the relationship between the loan_status 
# and the loan ammount
# chart6
# sns.boxplot(x="loan_status", y="loan_amnt", data=df)
# plt.title("Chart_6_Loan_Status_vs_Loan_Amount")
# plt.show()


# TODO 7: Calculate the summary statistics for the loan amount, grouped by loan_status
# print(df.groupby("loan_status").describe())

# print(df.groupby("loan_status")["loan_amnt"].describe())
#              count          mean          std      min      25%      50%      75%      max
# loan_status
# 0             96.0  16363.541667  7182.583024   2400.0  11412.5  16000.0  20000.0  35000.0
# 1             71.0  20282.746479  7052.919344  10000.0  15350.0  19100.0  24000.0  35000.0
# 2            463.0  14834.611231  8827.218303   1000.0   8000.0  12600.0  20000.0  35000.0
# 3              1.0  16000.000000          NaN  16000.0  16000.0  16000.0  16000.0  16000.0
# 4              5.0  18305.000000  9238.513138   8725.0  10800.0  16000.0  28000.0  28000.0


# TODO 8: Let's explore the grade and sub_grade columns what are the unique possible
# grades and subgrades
# what unique possible grades and subgrades are there?
# uni_grades = df["grade"].unique()
# uni_sub_grades = df["sub_grade"].unique()
# I don't have grade and sub_grade columns due to categorical data conversion
# issues

# TODO 9: Create a countplot per grade. Set the hue to the loan_status label
# chart7
# sns.countplot(x="grade", hue="loan_status", data=df)
# plt.title("Chart_7_Countplot_Per_Grade")
# plt.show()


# TODO 10: Create a countplot per sub_grade. You may need to reorder the x-axis.
# Feel Free to edit the color palatte. Explore both all loans made per sub_grade
# as wel being separated based on the loan_status.
# chart8
# plt.figure(figsize=(12, 6))
# subgrade_order = sorted(df["sub_grade"].unique())
# sns.countplot(x="sub_grade", data=df, order=subgrade_order, palette="coolwarm")
# plt.title("Chart_8_Countplot_Per_Sub_Grade")
# plt.show()

# chart9
# plt.figure(figsize=(12, 6))
# subgrade_order = sorted(df["sub_grade"].unique())
# sns.countplot(x="sub_grade", data=df, order=subgrade_order, 
#                 palette="coolwarm", hue="loan_status")
# plt.title("Chart_9_Countplot_Per_Sub_Grade_loand_status_hue")
# plt.show()

# It looks like F and G subgrades don't get paid back that often. Isolate those
# and recreate the countplot just for those subgrades.
# chart10
# f_and_g = df[(df["grade"] == "F") | (df["grade"] == "G")]
# subgrade_order = sorted(df["sub_grade"].unique())
# sns.countplot(x="sub_grade", data=f_and_g, order=subgrade_order, 
#                 palette="coolwarm", hue="loan_status"
#             )
# plt.title("Chart_10_Countplot_F_and_G_grade_Loans")
# plt.show()

# TODO 11: Create a new column called "loan_repaid" which will contain a 1 if it
# was "Fully Paid" and a 0 if it was "Charged Off"

# df["loan_repaid"] = df["loan_status"].map({"Fully Paid":1,"Charged Off":0})
# print(df["loan_repaid"])
# print(df["loan_repaid","loan_status"])
# array(["Fully Paid", "Charged Off"],dtype=object)


# TODO 12: Create a barplot showing the correlation of the numeric features to the
# new loan_repaid column.
# chart11
# df.corr()["loan_status"].plot(kind="bar")
# plt.title("Chart_11_Correlation_of_Numeric_Features")
# plt.show()

# chart12
# df.corr()["loan_status"].sort_values().plot(kind="bar")
# plt.title("Chart_12_Correlation_of_Numeric_Features_Sorted")
# plt.show()

# chart13
# df.corr()["loan_status"].sort_values().drop("loan_status").plot(kind="bar")
# plt.title("Chart_13_Correlation_of_Numeric_Features_Sorted")
# plt.show()


### --------- Data PreProcessing --------- ###
# print(df.head())
#    loan_status  loan_amnt  int_rate  ...  total_acc  mort_acc  pub_rec_bankruptc
# ies
# 0            1     3600.0     13.99  ...       13.0       1.0
# 0.0
# 1            1    24700.0     11.99  ...       38.0       4.0
# 0.0
# 2            1    20000.0     10.78  ...       18.0       5.0
# 0.0
# 3            1    10400.0     22.45  ...       35.0       6.0
# 0.0
# 4            1    11950.0     13.44  ...        6.0       0.0
# 0.0

# [5 rows x 13 columns]

# print(df.shape)
# (559, 13)


# create a series that displays the total count of missing values per column
# null_vals = df.isnull().sum().sort_values(ascending=False)
# print(f"null_vals:\n{null_vals}")
# null_vals: 
# loan_status             0
# loan_amnt               0
# int_rate                0
# installment             0
# annual_inc              0
# dti                     0
# open_acc                0
# pub_rec                 0
# revol_bal               0
# revol_util              0
# total_acc               0
# mort_acc                0
# pub_rec_bankruptcies    0
# dtype: int64


# Convert this series to be in term of percentage of the total Dataframe
# perc_null_vals = (null_vals / df.shape[0]) * 100
# perc_null_vals = 100 * null_vals/len(df)
# print(f"perc_null_vals:\n{perc_null_vals}")

# print the number of unique emp_title values
# print(df["emp_title"].nunique())
# print(df["emp_title"].value_counts())
# drop emp_title column because there are too many values
# df.drop("emp_title", axis=1, inplace=True)


# TODO 13: Create a countplot of the emp_length feature column. Change the order
# of the values
# chart14
# sorted(df["emp_length"].dropna().unique())
# emp_length_order = ["< 1 year", "1 year", "2 years", "3 years", "4 years", 
#                     "5 years", "6 years", "7 years", "8 years", "9 years",
#                     "10+ years"
#                     ]

# sns.countplot(x="emp_length", data=df, order=emp_length_order, hue="loan_status")
# plt.title("Chart_14_Countplot_Emp_Length")
# plt.show()


# Fully Paid = 1, Charged Off = 0
# find the percentage of Charge Offs per category
# emp_co = df[df["loan_status"] == 0].groupby("emp_length").count()["loan_status"]

# emp_fp = df[df["loan_status"] == 1].groupby("emp_length").count()["loan_status"]

# per_co = (emp_co / df.shape[0]) * 100
# per_co = emp_co / (emp_co + emp_fp) * 100

# chart15
# per_co.plot(kind="bar")
# plt.title("Chart_15_Percentage_of_Charge_Offs_Per_Category")
# plt.show()

# because emp_length does not seem to effect the percentage of Charge Offs drop
# the column
# df.drop("emp_length", axis=1, inplace=True)


# recheck for null values
# print(df.isnull().sum().sort_values(ascending=False))


# TODO 14: Review the title column vs the purpose column. Is this repeated information?
# information is repeated, drop title column
# df.drop("title", axis=1, inplace=True)


# TODO 15: Create a value_counts of the mort_acc column
# print(df["mort_acc"].value_counts())
# mort_acc
# 0.0     196
# 2.0     100
# 1.0      98
# 3.0      51
# 4.0      40
# 5.0      37
# 6.0      18
# 7.0      10
# 8.0       6
# 12.0      1
# 9.0       1
# 10.0      1
# Name: count, dtype: int64


# review which columns are most highly correlated to mort_acc
# mort_acc_corr = df.corr()["mort_acc"].sort_values()
# print(f"mort_acc_corr:\n{mort_acc_corr}")
# mort_acc_corr:
# int_rate               -0.110604
# pub_rec                -0.059344
# dti                    -0.054832
# pub_rec_bankruptcies   -0.044564
# revol_util              0.074723
# open_acc                0.100686
# loan_status             0.125913
# installment             0.164838
# loan_amnt               0.187037
# total_acc               0.295323
# revol_bal               0.304012
# annual_inc              0.421603
# mort_acc                1.000000
# Name: mort_acc, dtype: float64



# find the mean of the mort_acc column per total_acc
# print(df.groupby("total_acc").mean()["mort_acc"])
# total_acc_mean = df.groupby("total_acc").mean()["mort_acc"]


# def fill_mort_acc(total_acc, mort_acc):
#     if np.isnan(mort_acc):
#         return total_acc_mean[total_acc]
#     else:
#         return mort_acc

# fill missing values in mort_acc column with the mean of the mort_acc and 
# total_acc column
# df["mort_acc"] = df.apply(lambda x: fill_mort_acc(x["total_acc"], x["mort_acc"]),axis=1)



# List all of the columns that are non_numeric
# df.select_dtypes(exclude="number").columns

# select only columns that have string values
non_num_columns = df.select_dtypes(["object"]).columns
# print(f"non_num_columns:\n{non_num_columns}")
# non_num_columns:
# Index([], dtype='object')

# select only columns that have numeric values
num_columns = df.select_dtypes(["number"]).columns
# print(f"num_columns:\n{num_columns}")
# num_columns:
# Index(['loan_status', 'loan_amnt', 'int_rate', 'installment', 'annual_inc',
#        'dti', 'open_acc', 'pub_rec', 'revol_bal', 'revol_util', 'total_acc',
#        'mort_acc', 'pub_rec_bankruptcies'],
#       dtype='object')


# term feature
# print(df["term"].value_counts())
# 36 months   434
# 60 months   125

# grab the first three characters of the term column to make this a numeric
# column
df["term"] = df["term"].apply(lambda term: int(term[1:3]))

# print(df["term"].value_counts())
# 36   434
# 60   125

# grade feature is already part of sub_grade, so drop the grade feature
# df.drop("grade", axis=1, inplace=True)

# convert the subgrade into dummy variables. The concantenate these new
# columns to the original dataframe. Remember to drop the original
# subgrade column and to add drop_first to your get_dummies call.

# create dummy variables for sub_grade
sub_grade_dummies = pd.get_dummies(df["sub_grade"], 
                                    drop_first=True
                                    )

# drop the original sub_grade column
df.drop("sub_grade", axis=1, inplace=True)

# concatenate the dummy variables to the original dataframe
df = pd.concat([df, sub_grade_dummies], axis=1)

# print(df.head())
#    loan_status  term  loan_amnt  int_rate  ...     F3     F5     G1     G2
# 0            1    36     3600.0     13.99  ...  False  False  False  False
# 1            1    36    24700.0     11.99  ...  False  False  False  False
# 2            1    60    20000.0     10.78  ...  False  False  False  False
# 3            1    60    10400.0     22.45  ...  False  False  False  False
# 4            1    36    11950.0     13.44  ...  False  False  False  False

# [5 rows x 44 columns]


# TODO 16: Convert verification_status, application_type, initial_list_status, and purpose
# into dummy variables and concantante them with the original dataframe. Remember
# to set drop_first = True and to drop the original columns.
df_dummies = pd.get_dummies(df[["verification_status","application_type",
                                "initial_list_status","purpose"]], 
                                drop_first=True
                                )
# app_type_dummies = pd.get_dummies(df["application_type"],
#                                     drop_first=True
#                                     )
# in_list_stat_dummies = pd.get_dummies(df["initial_list_status"],
#                                         drop_first=True
#                                         )
# purpose_dummies = pd.get_dummies(df["purpose"],
#                                 drop_first=True
#                                 )

df.drop(["verification_status","application_type","initial_list_status","purpose"],
        axis=1, inplace=True
        )

df = pd.concat([df, df_dummies], 
                axis=1
                )


# output all column names
# print(df.columns)
# Index(['loan_status', 'term', 'loan_amnt', 'int_rate', 'installment',
#        'annual_inc', 'dti', 'open_acc', 'pub_rec', 'revol_bal', 'revol_util',
#        'total_acc', 'mort_acc', 'pub_rec_bankruptcies', 'A2', 'A3', 'A4', 'A5',
#        'B1', 'B2', 'B3', 'B4', 'B5', 'C1', 'C2', 'C3', 'C4', 'C5', 'D1', 'D2',
#        'D3', 'D4', 'D5', 'E1', 'E2', 'E3', 'E4', 'E5', 'F1', 'F2', 'F3', 'F5',
#        'G1', 'G2', 'verification_status_Source Verified',
#        'verification_status_Verified', 'application_type_Joint App',
#        'initial_list_status_w', 'purpose_credit_card',
#        'purpose_debt_consolidation', 'purpose_home_improvement',
#        'purpose_house', 'purpose_major_purchase', 'purpose_medical',
#        'purpose_moving', 'purpose_other', 'purpose_small_business',
#        'purpose_vacation'],
#       dtype='object')


# TODO 17: Review the value_counts for the home_ownership column
# print(df["home_ownership"].value_counts())
# home_ownership
# MORTGAGE    305
# RENT        191
# OWN          63
# Name: count, dtype: int64


# if the home_ownership value is NONE or ANY change it to OTHER
df["home_ownership"] = df["home_ownership"].replace(["NONE", "ANY"], "OTHER")

# convert the home_ownership into dummy variables. The concantante these new values to 
# the original dataframe. 
dummies = pd.get_dummies(df["home_ownership"], drop_first=True)
df.drop("home_ownership", axis=1, inplace=True)
df = pd.concat([df, dummies], axis=1)

# grab the last five characters of the address column and create a zip_code column
# df["zip_code"] = df["address"].apply(lambda x: x[-5:])
# print(df["zip_code"].value_counts())


# TODO 18: convert the zip_code into dummy variables. The concantante these new values to 
# the original dataframe. 
# dummies = pd.get_dummies(df["zip_code"], drop_first=True)
# df.drop("address", axis=1, inplace=True)
# df = pd.concat([df, dummies], axis=1)


# TODO 19: Issue D: This would be data leakage since we wouldn't know beforehand
# whether of not a loan would be issued when using the model so we drop this column.
# df.drop("issue_d", axis=1, inplace=True)


# TODO 20: earliest_cr_line: This appears to be a historical time stamp feature. Extract
# the year using .apply, then convert it to a numeric feature. Create a new column 
# named earliest_cr_year from the year collected. Then drop the earliest_cr_line column

df["earliest_cr_year"] = df["earliest_cr_line"].apply(lambda x:x[-4:])
df.drop("earliest_cr_line", axis=1, inplace=True)

df.to_csv("_4_Machine_Learning/keras_7/Data/loan_data.csv")












































































































