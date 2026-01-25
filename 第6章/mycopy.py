# 可变变量与不可变变量的区别

# 可变变量
list1 = [1, 2, 3]
list2 = list1
list2[0] = 100
print(list2, list1)  # [100, 2, 3]

# 不可变变量
a = 1
b = a
b = 2
print(a, b)  # 1 2
