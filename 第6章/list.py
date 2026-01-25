# 四种复合类型示例

# # 列表
# list1 = [1, 2, 3, 4, 5]
# print(list1)

# # list 添加
# list1.append(6)
# print(list1)
# list1.append("hello")
# print(list1)

# list1.insert(1, 7)
# print(list1)

# # list 删除
# list1.pop()
# print(list1)

# list1.pop(1)
# print(list1)

# list1.remove(4)
# print(list1)
# list1.append([5, 6])
# print(list1)
# list1.remove([5, 6])
# print(list1)

# # list 修改
# list1[0] = 8
# print(list1)

# 元组
t = (1, 2, 3, 4, 5)
print(t)

# t[0] = 6
# print(t)

# 集合
s = {1, 2, 3, 4, 5, 5, 4, 3}
print(s)
s.add(7)
print(s)
s.remove(1)
print(s)
# s[0] = 99
# print(s)

# 字典
d = {'name': 'Tom', 'age': 18}
print(d)
d['name'] = 'Jerry'
print(d)
print(d.values())
print(d.keys())
d['shool'] = '清华大学'
print(d)
d.pop('age')
print(d)

del d['shool']
print(d)

list0 = [1, 2]
a, b = list0
print(a, b)

print(t)
print(*t)