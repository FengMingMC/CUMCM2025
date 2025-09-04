import pandas as pd
import numpy as np

# 创建一个包含非线性关系的示例数据集
data = {'X': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'Y': [1, 4, 9, 16, 25, 36, 49, 64, 81, 100]} # Y = X^2
df = pd.DataFrame(data)

print(df)

# 计算皮尔逊相关系数
pearson_corr = df.corr(method='pearson')
print("皮尔逊相关系数矩阵：\n", pearson_corr)

# 计算斯皮尔曼相关系数
spearman_corr = df.corr(method='spearman')
print("\n斯皮尔曼相关系数矩阵：\n", spearman_corr)