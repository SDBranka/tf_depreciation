# information on data set
# price - price in US dollars (\$326 — \$18,823)
# carat - weight of the diamond (0.2–5.01)
# cut - quality of the cut (Fair, Good, Very Good, Premium, Ideal)
# color - diamond colour, from J (worst) to D (best)
# clarity - a measurement of how clear the diamond is (I1 (worst), SI2, SI1, VS2, VS1, VVS2, VVS1, IF (best))
# x - length in mm (0–10.74)
# y - width in mm (0–58.9)
# z - depth in mm (0–31.8)
# depth - total depth percentage = z / mean(x, y) = 2 * z / (x + y) (43–79)
# table - width of top of diamond relative to widest point (43–95)


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


diamonds = pd.read_csv("Data/diamonds.csv")
# print(diamonds.head())
#    carat      cut color clarity  ...  price     x     y     z
# 0   0.23    Ideal     E     SI2  ...    326  3.95  3.98  2.43
# 1   0.21  Premium     E     SI1  ...    326  3.89  3.84  2.31
# 2   0.23     Good     E     VS1  ...    327  4.05  4.07  2.31
# 3   0.29  Premium     I     VS2  ...    334  4.20  4.23  2.63
# 4   0.31     Good     J     SI2  ...    335  4.34  4.35  2.75


# TODO: 1 create a scatterplot of price vs carat
# sns.scatterplot(x="carat",y="price",data=diamonds)
# plt.title("ex1")
# plt.show()


# TODO: 2 Use alpha parameter and edgecolor parameter to deal with overlapping 
# issue and white edgemarker issue
# sns.scatterplot(x="carat",y="price",data=diamonds,alpha=0.1,edgecolor=None)
# plt.title("ex2")
# plt.show()

# TODO: 3 Make the previous plot larger
# plt.figure(figsize=(12,8))
# sns.scatterplot(x="carat",y="price",data=diamonds,alpha=0.1,edgecolor=None)
# plt.title("ex3")
# plt.show()


# TODO: 4 create histogram of the price column as shown (xlim at 18000, no kde)
# sns.distplot(diamonds["price"],kde=False)
# plt.xlim(left=0,right=18000)
# plt.title("ex4")
# plt.show()


# TODO: 5 create a count plot of the instances per cut type 
# sns.countplot(x="cut", data=diamonds)
# plt.title("ex5")
# plt.show()


# TODO: 6 create a large box plot figure showing the price distribution per cut
# type as shown
# sns.boxplot(x="cut",y="price",data=diamonds)
# plt.title("ex6")
# plt.show()


# TODO: 7 figure out how to change the ordering of the box plot ("fair","ideal","good cut","very good","premium")
sns.boxplot(x="cut",y="price",
            data=diamonds,
            order=["Fair","Ideal","Good","Very Good","Premium"],
            palette="cool"
            )
plt.title("ex7")
plt.show()


