# info about the data
# case - A number which denotes a specific country
# cc3 - A three letter country code
# country - The name of the country
# year - The year of the observation
# systemic_crisis - "0" means that no systemic crisis occurred in the year and "1" means that a systemic crisis occurred in the year.
# exch_usd - The exchange rate of the country vis-a-vis the USD
# domestic_debt_in_default - "0" means that no sovereign_external_debt_default domestic debt default occurred in the year and "1" means that a sovereign domestic debt default occurred in the year
# sovereign_external_debt_default - "0" > 0ans tdefault occurred in the year and "1" means that a sovereign external debt default occurred in the year
# gdp_weighted_default - The total debt inef
# ault vis-a-vis the Gprint(gdp_weighted_default)DP
# inflation_annual_cpi - The annual CPI Inflation rate
# independence - "0" means "no independence" and "1" means "independence"
# currency_crises - "0" means that no currency crisis occurred in the year and "1" means that a currency crisis occurred in the year
# inflation_crises - "0" means that no inflation crisis occurred in the year and "1" means that an inflation crisis occurred in the year
# banking_crisis - "no_crisis" means that no banking crisis occurred in the year and "crisis" means that a banking crisis occurred in the year



import numpy as np 
import pandas as pd


# TODO:1 read in the african_econ_crises.csv file 
df = pd.read_csv("Data/african_crises.csv")


# TODO:2 display the first five rows
print(df.head())
#    case  cc3  country  ...  currency_crises  inflation_crises  banking_crisis
# 0     1  DZA  Algeria  ...                0                 0          crisis
# 1     1  DZA  Algeria  ...                0                 0       no_crisis
# 2     1  DZA  Algeria  ...                0                 0       no_crisis
# 3     1  DZA  Algeria  ...                0                 0       no_crisis
# 4     1  DZA  Algeria  ...                0                 0       no_crisis

# [5 rows x 14 columns]
print("\n")


# TODO:3 How many countries are represented in this dataset
# df_one["k1"].nunique()
num_uniq_countries = df["country"].nunique()
# print(num_uniq_countries)
# 13


# TODO:4 What countries are represented in this dataset
uniq_countries = df["country"].unique()
# print(uniq_countries)
# ['Algeria' 'Angola' 'Central African Republic' 'Ivory Coast' 'Egypt'
#  'Kenya' 'Mauritius' 'Morocco' 'Nigeria' 'South Africa' 'Tunisia' 'Zambia'
#  'Zimbabwe']


# TODO:5 What country had the highest annual CPI inflation rate? What was 
# the inflation rate?
# max_annual_CPI = df["inflation_annual_cpi"].argmax()
# # print(max_annual_CPI)
# # 1053
# count_w_max_CPI = df["country"][max_annual_CPI]
# print(count_w_max_CPI)
# Zimbabwe
count_w_max_CPI = df["country"][df["inflation_annual_cpi"].argmax()]
# print(count_w_max_CPI)
# Zimbabwe

# max_cpi_zim = df[df["country"] == "Zimbabwe"]["inflation_annual_cpi"].max()
# print(max_cpi_zim)
# 21989695.22
max_cpi_zim = df["inflation_annual_cpi"].max()
print(max_cpi_zim)


# another way
# count_w_max_CPI = df.sort_values("inflation_annual_cpi", ascending=False).head(1)["country"]

# TODO:6 In what year did Kenya have its first System Crisis?
kenya_crisis = df[(df["country"] == "Kenya") & (df["systemic_crisis"] == 1)].sort_values(by="year",ascending=True)
# print(kenya_crisis)
kenya_first_sys_crisis_year = kenya_crisis["year"].iloc[0]
# print(kenya_first_sys_crisis_year)
# 1985


# TODO:7 How many yearly systemic crises have occurred per country?
crisis_count = df.groupby(["year", "country"]).sum("systemic_crisis")
crisis_count = crisis_count["systemic_crisis"]
# print(crisis_count)
# year  country
# 1860  Egypt        0
# 1861  Egypt        0
# 1862  Egypt        0
# 1863  Egypt        0
# 1864  Egypt        0
#                   ..
# 2014  Mauritius    0
#       Morocco      0
#       Nigeria      1
#       Tunisia      0
#       Zambia       0
# Name: systemic_crisis, Length: 1059, dtype: int64


# TODO:8 How many years did Zimbabwe have a sovereign external debt default occur?
zim_sedd = len(df[(df["country"] == "Zimbabwe") & (df["sovereign_external_debt_default"] > 0)])
# print(zim_sedd)
# 30

# TODO:9 In what year did Algeria have its highest exchange rate?
max_exch_alg = df[df["country"] == "Algeria"]["exch_usd"].max()
# print(max_exch_alg)
# 87.9706983


