import numpy as np


# 2 create an array of ten zeros
arr1 = np.zeros(10)
# print(f"arr1: {arr1}")
# arr1: [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]

# 3 create an array of ten ones
arr2 = np.ones(10)
# print(f"arr2: {arr2}")
# arr2: [1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]

# 4 create an array of ten fives
arr3 = np.full(shape=10, fill_value=5)
# print(f"arr3: {arr3}")
# arr3: [5 5 5 5 5 5 5 5 5 5]

# 5 create an array of the integers from 10 to 50
arr4 = np.arange(10,51)
# print(f"arr4: {arr4}")
# arr4: [10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33
#  34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50]

# 6 create an array of all the even integers from 10 to 50
arr5 = np.arange(10,51,2)
# print(f"arr5: {arr5}")
# arr5: [10 12 14 16 18 20 22 24 26 28 30 32 34 36 38 40 42 44 46 48 50]

# 7 create a 3x3 matrix with values ranging from 0 to 8
arr6 = np.arange(0,9).reshape(3,3)
# print(f"arr6: {arr6}")
# arr6: [[0 1 2]
#        [3 4 5]
#        [6 7 8]]

# 8 create a 3x3 identity matrix
arr7 = np.eye(3)
# print(f"arr7: {arr7}")
# arr7: [[1. 0. 0.]
#        [0. 1. 0.]
#        [0. 0. 1.]]

# 9 use numpy to generate a random number between 0 and 1
r_num = np.random.rand()
# print(f"r_num: {r_num}")
# r_num: 0.6602976247580878

# 10 use numpy to generate an array of 25 random numbers sampled from a 
# standard normal distribution
arr8 = np.random.randn(25)
# print(f"arr8: {arr8}")
# arr8: [-0.36044036 -1.32691053 -0.99816115 -1.76999539  1.08010734  0.21037869
#         0.81887131  0.22500024  0.16117642  0.60078728  0.65226438 -0.74337566
#        -0.99404101  2.15167765 -0.54239606  0.13223784 -1.02249671  0.23030049
#         0.06874845 -0.48908688  2.00107081 -1.39543846  0.18310341  0.19540262
#         0.65564117]

# 11 create a 10x10 matrix from 0.01 to 1. in steps of 0.01
arr9 = np.arange(start=0.01,stop=1.01,step=0.01).reshape(10,-1)
# print(f"arr9: {arr9}")
# arr9: [[0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1 ]
#        [0.11 0.12 0.13 0.14 0.15 0.16 0.17 0.18 0.19 0.2 ]
#        [0.21 0.22 0.23 0.24 0.25 0.26 0.27 0.28 0.29 0.3 ]
#        [0.31 0.32 0.33 0.34 0.35 0.36 0.37 0.38 0.39 0.4 ]
#        [0.41 0.42 0.43 0.44 0.45 0.46 0.47 0.48 0.49 0.5 ]
#        [0.51 0.52 0.53 0.54 0.55 0.56 0.57 0.58 0.59 0.6 ]
#        [0.61 0.62 0.63 0.64 0.65 0.66 0.67 0.68 0.69 0.7 ]
#        [0.71 0.72 0.73 0.74 0.75 0.76 0.77 0.78 0.79 0.8 ]
#        [0.81 0.82 0.83 0.84 0.85 0.86 0.87 0.88 0.89 0.9 ]
#        [0.91 0.92 0.93 0.94 0.95 0.96 0.97 0.98 0.99 1.  ]]

# 12 create an array of 20 linearly spaced points between 0 and 1
arr10 = np.linspace(start=0, stop=1, num=20)
# print(f"arr10: {arr10}")
# arr10: [0.         0.05263158 0.10526316 0.15789474 0.21052632 0.26315789
#         0.31578947 0.36842105 0.42105263 0.47368421 0.52631579 0.57894737
#         0.63157895 0.68421053 0.73684211 0.78947368 0.84210526 0.89473684
#         0.94736842 1.        ]



# starting matrix
mat = np.arange(1,26).reshape(5,5)
# print(f"mat: \n{mat}")
# # mat:
# # [[ 1  2  3  4  5]
# #  [ 6  7  8  9 10]
# #  [11 12 13 14 15]
# #  [16 17 18 19 20]
# #  [21 22 23 24 25]]


# slice row2-end/col1-end
slice1 = mat[2:,1:]
# print(f"slice1: {slice1}")
# slice1: [[12 13 14 15]
#          [17 18 19 20]
#          [22 23 24 25]]

# access row3,col4
acc = mat[3,4]
# print(f"acc: {acc}")
# acc: 20

# create a 2d array of r0c1,r1c1,r2c1
slice2 = mat[:3,1].reshape(3,-1)
# print(f"slice2: {slice2}")
# # slice2: [[ 2]
# #          [ 7]
# #          [12]]

# reproduce [21 22 23 24 25] from mat
slice3 = mat.flatten()[-5:]
# print(f"slice3: {slice3}")
# slice3: [21 22 23 24 25]

# reproduce [[16 17 18 19 20]
        #   [21 22 23 24 25]]
slice4 = mat.flatten()[-10:].reshape(2,-1)
# print(f"slice4: {slice4}")
# slice4: [[16 17 18 19 20]
#          [21 22 23 24 25]]

# get the sum of all the values in mat
mat_sum = mat.sum()
# print(f"mat_sum: {mat_sum}")
# mat_sum: 325

# get the standard deviation for the values in mat
mat_std = mat.std()
# print(f"mat_std: {mat_std}")
# mat_std: 7.211102550927978

# get the sum of all the columns in mat
column_sum = mat.sum(axis=0)
# print(f"column_sum: {column_sum}")
# column_sum: [55 60 65 70 75]



