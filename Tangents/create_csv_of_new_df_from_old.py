# www.kaggle.com/wordsforthewise/lending-club
# shortened the dataset from above address to 636 values 
# this app pulls the required columns from the original dataset
# and creates a new csv of the dataframe

import numpy as np
import pandas as pd

### --------- Data --------- ###
# df = pd.read_csv('lending_club.csv')
orig_df = pd.read_csv("Data/keras_7/aaa.csv")

# create empty dataframe
df = pd.DataFrame()

# df["address"] = orig_df["address"]
# df["emp_title"] = orig_df["emp_title"]#
# df["emp_length"] = orig_df["emp_length"]#
# df["title"] = orig_df["title"]#
# df["issue_d"] = orig_df["issue_d"]#


# categorical
# df["grade"] = orig_df["grade"]#
df["loan_status"] = orig_df["loan_status"]#
df["term"] = orig_df["term"]#
df["sub_grade"] = orig_df["sub_grade"]#
df["initial_list_status"] = orig_df["initial_list_status"]#
df["application_type"] = orig_df["application_type"]#
df["verification_status"] = orig_df["verification_status"]#
df["purpose"] = orig_df["purpose"]#
df["home_ownership"] = orig_df["home_ownership"]#
df["earliest_cr_line"] = orig_df["earliest_cr_line"]#


# numeric
df["loan_amnt"] = orig_df["loan_amnt"]#
df["int_rate"] = orig_df["int_rate"]#
df["installment"] = orig_df["installment"]#
df["annual_inc"] = orig_df["annual_inc"]#
df["dti"] = orig_df["dti"]#
df["open_acc"] = orig_df["open_acc"]#
df["pub_rec"] = orig_df["pub_rec"]#
df["revol_bal"] = orig_df["revol_bal"]#
df["revol_util"] = orig_df["revol_util"]#
df["total_acc"] = orig_df["total_acc"]#
df["mort_acc"] = orig_df["mort_acc"]
df["pub_rec_bankruptcies"] = orig_df["pub_rec_bankruptcies"]#

### --------- Create Dataframe of only Fully Paid or Charged Off--------- ###
df = df[(df["loan_status"] == "Fully Paid") | (df["loan_status"] == "Charged Off")]
# df = df[df["loan_status"] == "Charged Off"]



# ### --------- Change Categorical Column to Numeric --------- ###
# change the type of the column:
df.loan_status = pd.Categorical(df.loan_status)
# Now the data look similar but are stored categorically. To capture the 
# category codes:
df['loan_status'] = df.loan_status.cat.codes


# df.loan_status.astype('category').cat.codes
# # Or use the categorical column as an index:

# df2 = pd.DataFrame(df.temp)
# df2.index = pd.CategoricalIndex(df.loan_status)
# print(f"loan_status: {df.loan_status}")





df.to_csv("Data/keras_7/lending_club.csv", index=False)











