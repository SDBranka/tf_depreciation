# I didn't have access to the csv, so some of this may be off

# data information
# age - age in years
# sex - 1=male, 0=female
# cp - chest pain type
# trestbps - resting bloodpressure in mmHg
# cholserum - cholesterol in mg/dl
# fbs - fasting blood sugar > 120 mg/dl, 1=true, 0=False
# restecg - resting ecg results
# thalach - max heart rate achieved
# exang - exercise induced angina, 1=yes,0=no
# oldpeak - ST depression induced by exercise relative to rest
# slope - the slope of the peak exercise ST segment
# ca - the number of major vessels (0-3) colored by flourosopy
# thal - 3=normal, 6=fixed defect, 7=reversable defect
# target - 1 or 0


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv("Data/heart.csv")

# print(df.head())
#    age  sex  cp  trestbps  ...  slope  ca  thal  target
# 0   63    1   3       145  ...      0   0     1       1
# 1   37    1   2       130  ...      0   0     2       1
# 2   41    0   1       130  ...      2   0     2       1
# 3   56    1   1       120  ...      2   0     2       1
# 4   57    0   0       120  ...      2   0     2       1


# chart1
# distribution plot
# sns.displot(df["age"])
# plt.show()


# chart2
# show kernel density estimation (estimates probability function)
# sns.displot(df["age"], kde=True)
# plt.show()


# chart3a  compare to 1a as these are pulled from full data
# change the number of bins
# sns.displot(df["age"], bins=2)
# plt.show()
# using too many bins may lead to gaps in your data representation 
# because the over-specificity means you may have no values in 
# those ranges


# resize the presentation
# before the sns call use figsize (in inches) first tuple value is the
# xstretch, the second is ystretch
# plt.figure(figsize=(3,8))
# sns.displot(df["age"])
# plt.show()


# chart 4a
# change the color
# plt.figure(figsize=(8,4))
# sns.displot(df["age"],kde=False,bins=50,color="red")
# plt.show()


# chart5a
# count plot
# sns.countplot(x="sex", data=df)
# plt.show()


# chart6a
# sns.countplot(x="cp", data=df)
# plt.show()


# chart7a
# distinguishing by secondary feature
# sns.countplot(x="cp", data=df, hue="sex")
# plt.show()


# chart8a
# change color mapping
# sns.countplot(x="cp", data=df, hue="sex", palette="terrain")
# plt.show()


# chart9a
# box plot - shows the distribution across different categorical features
# see the distribution of age across the sex category
# sns.boxplot(x="sex",y="age",data=df)
# plt.show()


# chart10a
# compare data of max heart rate achieved in people with heart disease vs
# those without and differentiate between sex
# sns.boxplot(x="target",y="thalach",data=df, hue="sex")
# plt.show()


# chart11a
# scatter plots
# sns.scatterplot(x="chol",y="trestbps",data=df)
# plt.show()


# chart12a
# sns.scatterplot(x="chol",y="trestbps",
#                 data=df,hue="sex",
#                 palette="Dark2",
#                 size="age"
#                 )
# plt.show()


# pair plots
iris = pd.read_csv("Data/iris.csv")

# chart13
# sns.pairplot(iris)
# plt.show()

# chart14
# sns.pairplot(iris, hue="Species")
# plt.show()


