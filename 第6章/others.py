# 列表推导

list = [i**2 for i in range(10)]
print(list)

# 使用with语句对文件操作
with open("test.txt", "w") as f:
    f.write("hello world")

# 函数解包
def fun(a, b, c):
    print(a, b, c)

list = [1, 2, 3]
fun(*list)