import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re

plt.rcParams['font.sans-serif'] = ['MiSans']
plt.rcParams['axes.unicode_minus'] = False

df_boy = pd.read_excel('data/附件.xlsx', sheet_name = '男胎检测数据' )

def calculate_week(row):
    pregnancy_week_str = row['检测孕周']
    match = re.match(r'(\d+)w\+(\d+)', pregnancy_week_str)
    if match:
        # 如果找到匹配，提取周数和天数，并转换为整数
        weeks = int(match.group(1))
        days = int(match.group(2))
        return weeks + days / 7
    else:
        return int(re.search(r'(\d+)w', pregnancy_week_str).group(1))


df_boy['孕周'] = df_boy.apply(calculate_week, axis=1)

df_temp = df_boy[df_boy['GC含量'] >= 0.4]
df_boy = df_temp


calculate_frame = pd.DataFrame([df_boy["孕妇BMI"], df_boy["Y染色体浓度"]]).T

print(calculate_frame)

pearson_corr = calculate_frame.corr(method='pearson')
print("皮尔逊相关系数矩阵：\n", pearson_corr)

spearman_corr = calculate_frame.corr(method='spearman')
print("\n斯皮尔曼相关系数矩阵：\n", spearman_corr)












