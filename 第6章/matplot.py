# 使用matplotlib的示例

# 导入matplotlib库
import matplotlib.pyplot as plt
import numpy as np

# 创建一个画布
# plt.figure()
fig, ax = plt.subplots(1, 2, figsize=(8, 6))

# 通过numpy生成数据
x = np.linspace(0, 10, 100)
y = np.sin(x)

# 绘制图形, 散点图
# plt.scatter([1, 2, 3, 4], [10, 20, 25, 30])
ax[0].scatter(x, y)
# plt.show()

# 绘制饼状图
labels = ['A', 'B', 'C', 'D']
sizes = [15, 30, 45, 10]
ax[1].pie(sizes, labels=labels, autopct='%1.1f%%')

plt.show()



