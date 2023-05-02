import numpy as np


# seeding random to produce the repeatable values for tutorial purposes
np.random.seed(42)


arr = np.arange(0,10)
# print(f"arr: {arr}")
# # # arr: [0 1 2 3 4 5 6 7 8 9]


# math operations and arrays
# add 5 to every element in an array
arr1 = arr + 5
# print(f"arr1: {arr1}")
# # # arr1: [ 5  6  7  8  9 10 11 12 13 14]

arr2 = arr - 2
# print(f"arr2: {arr2}")
# # arr2: [-2 -1  0  1  2  3  4  5  6  7]

arr3 = arr + arr
# print(f"arr3: {arr3}")
# # arr3: [ 0  2  4  6  8 10 12 14 16 18]

# dividing by zero in an array (gives a warning)
# arr4 = arr / arr
# print(f"arr4: {arr4}")
# # arr4: [nan  1.  1.  1.  1.  1.  1.  1.  1.  1.]

# normal operations divided by numpy 0 (gives a warning)
# print(1 / arr)
# # [       inf 1.         0.5        0.33333333 0.25       0.2
# #  0.16666667 0.14285714 0.125      0.11111111]


# square root
arr5 = np.sqrt(arr)
# print(f"arr5: {arr5}")
# # arr5: [0.         1.         1.41421356 1.73205081 2.         2.23606798
# #  2.44948974 2.64575131 2.82842712 3.        ]

# sin 
arr6 = np.sin(arr)
# print(f"arr6: {arr6}")
# # arr6: [ 0.          0.84147098  0.90929743  0.14112001 -0.7568025  -0.95892427
# #  -0.2794155   0.6569866   0.98935825  0.41211849]

# log
# arr7 = np.log(arr)
# print(f"arr7: {arr7}")
# # RuntimeWarning: divide by zero encountered in log
# #   arr7 = np.log(arr)
# # arr7: [      -inf 0.         0.69314718 1.09861229 1.38629436 1.60943791
# #  1.79175947 1.94591015 2.07944154 2.19722458]


# sum all of the elements in the array
# print(f"sum of arr: {arr.sum()}")
# # sum of arr: 45

# avg of arr
# print(f"avg of arr: {arr.mean()}")
# avg of arr: 4.5

# max of arr
# print(f"max of arr: {arr.max()}")
# max of arr: 9

# standard deviation of arr
# print(f"standard deviation of arr: {arr.std()}")
# standard deviation of arr: 2.8722813232690143

arr_2d = np.arange(0,25).reshape(5,5)
# print(f"arr_2d: {arr_2d}")
# # arr_2d: [[ 0  1  2  3  4]
# #  [ 5  6  7  8  9]
# #  [10 11 12 13 14]
# #  [15 16 17 18 19]
# #  [20 21 22 23 24]]
# print(f"sum of arr: {arr_2d.sum()}")
# # sum of arr: 300

# sum across the rows (each column top to bottom)
row_sum = arr_2d.sum(axis=0)
# print(f"row_sum: {row_sum}")
# # row_sum: [50 55 60 65 70]

# sum across the columns (each row left to right)
column_sum = arr_2d.sum(axis=1)
# print(f"column_sum: {column_sum}")
# # column_sum: [ 10  35  60  85 110]









