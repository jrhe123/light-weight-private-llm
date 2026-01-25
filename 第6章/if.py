# 优先级判断

# 示例1
c = 10 + 2 * 10
print(c)

# 示例2
c = 16 * (1 >> 2)
print(c)

# 示例3
c = (4 | 1) << 3
print(c)

# 示例4
c = (4 | 2) & 1
print(c)

# 示例5
c = not 4 > 5 and 2 | 4  > 5
print(c)

if 4 > 5:
    print('4 > 5')
elif 4 < 5:
    print('4 < 5')
else:
    print('4 = 5')