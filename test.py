import matplotlib.pyplot as plt
import numpy as np

# 准备一些数据
# 例如，模拟三个不同组别的数据
data_group1 = np.random.normal(100, 10, 200) # 均值100，标准差10
data_group2 = np.random.normal(90, 20, 200)  # 均值90，标准差20
data_group3 = np.random.normal(110, 5, 200)  # 均值110，标准差5

data = [data_group1, data_group2, data_group3]

# 绘制箱线图
plt.boxplot(data, vert=True, patch_artist=True) # vert=True: 垂直箱线图

# 添加标题和标签
plt.title('Basic Box Plot with Matplotlib')
plt.xlabel('Data Groups')
plt.ylabel('Values')

# 设置x轴刻度标签
plt.xticks([1, 2, 3], ['Group 1', 'Group 2', 'Group 3'])

# 显示图表
plt.show()