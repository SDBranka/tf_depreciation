import numpy as np


arr = np.arange(start=0,stop=11)
# print(arr)
# # [ 0  1  2  3  4  5  6  7  8  9 10]


# indexing/list slicing
# print(f"arr at index 8: {arr[8]}")
# print(f"first four values of arr: {arr[0:4]}")
# print(f"third through sixth value of arr: {arr[2:5]}")
# print(f"sixth through last value of arr: {arr[5:]}")
# # arr at index 8: 8
# # first four values of arr: [0 1 2 3]
# # third through sixth value of arr: [2 3 4]
# # sixth through last value of arr: [ 5  6  7  8  9 10]


# broadcasting
# arr[0:5] = 100
# print(f"broadcasted arr: {arr}")
# # broadcasted arr: [100 100 100 100 100   5   6   7   8   9  10]

# arr[0:5:2] = 100
# print(f"broadcasted arr: {arr}")
# # broadcasted arr: [100   1 100   3 100   5   6   7   8   9  10]


# slices of lists are actually pointers and still broadcast
# print(arr)
# # [ 0  1  2  3  4  5  6  7  8  9 10]
slice_of_arr = arr[0:5]
# print(f"slice_of_arr: {slice_of_arr}")
# # [ 0  1  2  3  4]
# print(arr)
# # [ 0  1  2  3  4  5  6  7  8  9 10]
slice_of_arr[:] = 99
# print(f"slice_of_arr: {slice_of_arr}")
# # slice_of_arr: [99 99 99 99 99]
# print(arr)
# # [99 99 99 99 99  5  6  7  8  9 10]


arr = np.arange(start=0,stop=11)            #reset arr
# print(arr)
# # [ 0  1  2  3  4  5  6  7  8  9 10]


# modifiy array values of a slice w/o affecting orig array
arr_copy = arr.copy()
arr_copy[:] = 100
# print(f"arr_copy: {arr_copy}")
# print(f"arr: {arr}")
# # arr_copy: [100 100 100 100 100 100 100 100 100 100 100]
# # arr: [ 0  1  2  3  4  5  6  7  8  9 10]


# indexing on 2D arrays
arr_2d = np.array([[5, 10, 15], [20, 25, 30], [35, 40, 45]])
# print(f"arr_2d: {arr_2d}")
# # arr_2d: [[ 5 10 15]
# #  [20 25 30]
# #  [35 40 45]]

# print(f"arr_2d shape: {arr_2d.shape}")
# # (rows, columns)
# # (3,3)
# grab a single row
# print(f"first row of arr_2d: {arr_2d[0]}")
# # first row of arr_2d: [ 5 10 15]
# grab a single element
# print(f"row1,col1 of arr_2d: {arr_2d[1][1]}")
# print(f"or another way: {arr_2d[1, 1]}")
# # row1,col1 of arr_2d: 25
# # or another way: 25
# grab a slice of the matrix
# first two rows
# print(f"slice1 of arr_2d: {arr_2d[:2]}")
# print(f"slice2 of arr_2d: {arr_2d[:2,1:]}")
# # slice1 of arr_2d: [[ 5 10 15]
# #  [20 25 30]]
# # slice2 of arr_2d: [[10 15]
# #  [25 30]]


# conditional selection
arr = np.arange(1,11)       #reset arr
# print(f"arr: {arr}")
# # arr: [ 1  2  3  4  5  6  7  8  9 10]
# print(f"> 4?: {arr > 4}")
# # >4?: [False False False False  True  True  True  True  True  True]
# ex extract the values from arr that > 4 using index property
bool_arr = arr > 4
gre_than_four = arr[bool_arr]
# print(f"gre_than_four: {gre_than_four}")
# # gre_than_four: [ 5  6  7  8  9 10]
# written another way
# print(f"gre_than_four: {arr[arr > 4]}")
# # gre_than_four: [ 5  6  7  8  9 10]



