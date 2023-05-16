import pandas as pd

technologies = {
    'Courses':["Spark","PySpark","Python","pandas"],
    'Fee' :[20000,25000,22000,24000],
    'Duration':['30days','40days','35days','60days'],
    'Discount':[1000,2300,2500,2000]
    }
index_labels=['r1','r2','r3','r4']
df = pd.DataFrame(technologies,index=index_labels)
# print(df)
#     Courses    Fee Duration  Discount
# r1    Spark  20000   30days      1000
# r2  PySpark  25000   40days      2300
# r3   Python  22000   35days      2500
# r4   pandas  24000   60days      2000

# Number of rows to drop
n = 2
# By using DataFrame.iloc[] to drop first n rows
df2 = df.iloc[n:,:]
# print(df2)
#    Courses    Fee Duration  Discount
# r3  Python  22000   35days      2500
# r4  pandas  24000   60days      2000

# Using iloc[] to drop first n rows
df2 = df.iloc[2:]
# print(df2)
#    Courses    Fee Duration  Discount
# r3  Python  22000   35days      2500
# r4  pandas  24000   60days      2000

# Number of rows to drop
n = 2
# Using drop() function to delete first n rows
# df.drop(index=df.index[:n],axis=0, inplace=True)
# print(df)
#    Courses    Fee Duration  Discount
# r3  Python  22000   35days      2500
# r4  pandas  24000   60days      2000

# Number of rows to drop
n = 2
# Using DataFrame.tail() to Drop top two rows
df2 = df.tail(df.shape[0] -n)
# print(df2)
#    Courses    Fee Duration  Discount
# r3  Python  22000   35days      2500
# r4  pandas  24000   60days      2000

# Using DataFrame.tail() function to drop first n rows
df2 = df.tail(-2)
# print(df2)
#    Courses    Fee Duration  Discount
# r3  Python  22000   35days      2500
# r4  pandas  24000   60days      2000