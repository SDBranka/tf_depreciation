import numpy as np


mylist = [1, 2, 3]
# print(mylist)
# # [1, 2, 3]

nparr_mylist = np.array(mylist)
# print(nparr_mylist)
# # [1 2 3]


nested_list = [[1, 2], [3, 4], [5, 6]]
# print(nested_list)
# # [[1, 2], [3, 4], [5, 6]]

nparr_nested_list = np.array(nested_list)
# print(nparr_nested_list)
# # [[1 2]
# #  [3 4]
# #  [5 6]]


nparange = np.arange(start=0,stop=10)
# print(nparange)
# # [0 1 2 3 4 5 6 7 8 9]

nparange = np.arange(start=0,stop=11,step=2)
# print(nparange)
# # [ 0  2  4  6  8 10]


zero_arr = np.zeros(3)
# print(zero_arr)
# # [0. 0. 0.]

zero_arr = np.zeros(shape=3)
# print(zero_arr)
# # [0. 0. 0.]

zero_arr = np.zeros(shape=(4,4,4))
# print(zero_arr)
# # [[[0. 0. 0. 0.]
# #   [0. 0. 0. 0.]
# #   [0. 0. 0. 0.]
# #   [0. 0. 0. 0.]]

# #  [[0. 0. 0. 0.]
# #   [0. 0. 0. 0.]
# #   [0. 0. 0. 0.]
# #   [0. 0. 0. 0.]]

# #  [[0. 0. 0. 0.]
# #   [0. 0. 0. 0.]
# #   [0. 0. 0. 0.]
# #   [0. 0. 0. 0.]]

# #  [[0. 0. 0. 0.]
# #   [0. 0. 0. 0.]
# #   [0. 0. 0. 0.]
# #   [0. 0. 0. 0.]]]

ones_arr = np.ones(shape=4)
# print(ones_arr)
# # [1. 1. 1. 1.]


# start, stop, num evenly spaced samples, includes the stop point
lin_arr = np.linspace(start=0,stop=10,num=3)
# print(lin_arr)
# # [ 0.  5. 10.]

lin_arr = np.linspace(start=0,stop=10,num=21)
# print(lin_arr)
# # [ 0.   0.5  1.   1.5  2.   2.5  3.   3.5  4.   4.5  5.   5.5  6.   6.5
# #   7.   7.5  8.   8.5  9.   9.5 10. ]

# create a matrix of all zeros with ones on the diagonal
ident_matrix = np.eye(5)
# print(ident_matrix)
# # [[1. 0. 0. 0. 0.]
# #  [0. 1. 0. 0. 0.]
# #  [0. 0. 1. 0. 0.]
# #  [0. 0. 0. 1. 0.]
# #  [0. 0. 0. 0. 1.]]


# random functions
# generate random numbers between 0 and 1
a1 = np.random.rand(2)
# print(a1)
# # [0.26475847 0.60387204]

# generate random matrix of n rows by m columns all numbers between 0 and 1
a2 = np.random.rand(3, 4)
# print(a2)
# # [[0.67499692 0.5034988  0.10417178 0.23402672]
# #  [0.17600211 0.66580894 0.76814925 0.5030458 ]
# #  [0.25753894 0.43564687 0.90988153 0.60090858]]

# randn same as above but with normal distribution (mean is 0, variance is 1)
a3 = np.random.randn(5, 5)
# print(a3)
# # [[ 1.08518458  1.83310673  0.68741528 -0.31580653 -0.26447307]
# #  [-0.42779685  0.30183335 -0.09244797  1.03840852  1.53106563]
# #  [-0.23743519  0.81383883 -0.22293929  0.4930304   0.30066845]
# #  [ 0.79353169  1.41512458 -1.50838557  0.21076114 -0.78600383]
# #  [ 1.36834102  0.16893483 -0.21842767  0.26931579  1.43463004]]

# grab random integers (low(inclusive), high(exclusive), size)
# 10 random ages between 1 and 99
a4 = np.random.randint(low=1,high=100,size=10)
# print(a4)
# # [ 5 87 49 59 63 44 60 27 19 45]

a4 = np.random.randint(low=1,high=100,size=(2,3))
# print(a4)
# # [[64 96 32]
# #  [69 24 40]]

# # set the global random seed - generate reproducible random numbers 
# np.random.seed(42)
# a5 = np.random.rand(4)
# # print(a5)
# # # [0.37454012 0.95071431 0.73199394 0.59865848]


# random state without effecting the global seed
rng = np.random.RandomState(2021)
rang = rng.rand(4)
# print(rang)
# print(np.random.rand(4))
# # run1
# # [0.60597828 0.73336936 0.13894716 0.31267308]
# # [0.40452582 0.26109909 0.83622581 0.49842325]
# # run2
# # [0.60597828 0.73336936 0.13894716 0.31267308]
# # [0.38937282 0.44757928 0.53629255 0.30235438]


# reshape a vector
arr = np.arange(25)
# print(f"arr: {arr}")
# # arr: [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23
# # 24]
# print(f"arr shape: {arr.shape}")
# # arr shape: (25,)

# impermanent reshape
arr.reshape(5,5)
# print(f"arr reshape: {arr.reshape(5,5)}")
# # arr reshape: [[ 0  1  2  3  4]
# #  [ 5  6  7  8  9]
# #  [10 11 12 13 14]
# #  [15 16 17 18 19]
# #  [20 21 22 23 24]]
# print(f"arr shape: {arr.shape}")
# # arr shape: (25,)
# print(f"arr imperm reshape: {arr}")
# # arr imperm reshape: [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19
# #  20 21 22 23 24]


# permanent reshape
arr = arr.reshape(5,5)
# print(f"arr reshape: {arr.shape}")
# # arr reshape: (5, 5)

# print(f"arr perm reshape: {arr}")
# # arr perm reshape: [[ 0  1  2  3  4]
# #  [ 5  6  7  8  9]
# #  [10 11 12 13 14]
# #  [15 16 17 18 19]
# #  [20 21 22 23 24]]


ranarr = np.random.randint(0,50,10)
# print(f"ranarr: {ranarr}")
# # ranarr: [19 10 37 40 35 18 32 12 23 43]

# show max value of an array
max_ranarr = ranarr.max()
# print(f"max_ranarr: {max_ranarr}")
# # max_ranarr: 43

# show index of max of array
ind_max_ranarr = ranarr.argmax()
# print(f"index of max_ranarr: {ind_max_ranarr}")
# # index of max_ranarr: 9

# show min value of an array
min_ranarr = ranarr.min()
# print(f"min_ranarr: {min_ranarr}")
# # min_ranarr: 10

# show index of min of array
ind_min_ranarr = ranarr.argmin()
# print(f"index of min_ranarr: {ind_min_ranarr}")
# # index of min_ranarr: 1


# check datatype
dtype_ranarr = ranarr.dtype
# print(f"dtype_ranarr: {dtype_ranarr}")
# # dtype_ranarr: int32



















