# python 函数示例

# 定义一个函数
def func1():
    print("hello world")

def add(a, b):
    return a + b

def list_add(a, b):
    l = a + b
    return l, len(l)
    

# 调用函数
# func1()
# print(add(1, 2))
# print(add("hello", "world"))
list, len = list_add([1, 2, 3], [4, 5, 6])
print(list, len)