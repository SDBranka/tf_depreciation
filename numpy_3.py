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
arr7 = np.log(arr)
# print(f"arr7: {arr7}")
# # RuntimeWarning: divide by zero encountered in log
# #   arr7 = np.log(arr)
# # arr7: [      -inf 0.         0.69314718 1.09861229 1.38629436 1.60943791
# #  1.79175947 1.94591015 2.07944154 2.19722458]














