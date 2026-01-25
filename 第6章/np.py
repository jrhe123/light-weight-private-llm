# numpy的使用示例

import numpy as np

# 使用numpy创建一个一维数组
a = np.array([1, 2, 3, 4, 5])

# 使用numpy创建一个二维数组
b = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# 使用numpy创建一个三维数组
c = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])

# print(a)
# print(b)
# print(c)

# # 使用numpy创建一个全是0的数组
# d = np.zeros((3, 4))
# print(d)

# # 使用numpy创建一个全是1的数组
# e = np.ones((3, 4))
# print(e)

# # 使用numpy创建一个单位矩阵
# f = np.eye(3)
# print(f)

# 使用numpy创建进行加、减、乘、除运算
# g = np.array([10, 20, 30, 40])
# h = np.array([2, 3, 4, 5])
# print(g + h) 
# print(g - h)   

# print(g * h)
# print(np.multiply(g,h))

# print(g / h)

# # 使用numpy进行叉乘，@ 或者 np.dot()
# i = np.array([[1, 2], [3, 4]])
# j = np.array([[5, 6], [7, 8]])
# print(i @ j)
# print(np.dot(i, j))
# print(np.matmul(i, j))

# numpy的广播机制
k = np.array([[1, 2], [3, 4], [5, 6]])
l = np.array([10, 20])
print(k + l)
