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









plt.scatter(df_boy.孕妇BMI, df_boy.Y染色体浓度, c='blue', alpha=0.3, s=10)


plt.title("BMI与Y染色体浓度相关散点图")
plt.xlabel("BMI")
plt.ylabel("Y染色体浓度")
plt.savefig('pic/BMI-Y染色体_散点图.svg')
plt.show()

plt.scatter(df_boy.孕周, df_boy.Y染色体浓度, c='blue', alpha=0.3, s=10)


plt.title("孕周与Y染色体浓度相关散点图")
plt.xlabel("孕周")
plt.ylabel("Y染色体浓度")
plt.savefig('pic/孕周染色体_散点图.svg')
plt.show()


print(df_boy)
plt.boxplot(df_boy.孕妇BMI, vert=True, patch_artist=True)

plt.title('Box Plot of BMI of Pregnant')
plt.xlabel('Data Groups')
plt.ylabel('BMI of Pregnant')

plt.xticks([1], [''])


plt.show()




groups = df_boy['孕妇代码'].unique()




# 2. 创建图表
plt.figure(figsize=(10, 6))

for i, group in enumerate(groups):
    df_group = df_boy[df_boy['孕妇代码'] == group]

    # 绘制散点图
    plt.scatter(df_group['孕妇BMI'], df_group['Y染色体浓度'],
                label=f'分组 {group}',
                color='blue',
                marker='o',
                s=10,
                alpha=0.20)  # s参数用于调整点的大小

    # 绘制连线
    # 按照X轴顺序对数据进行排序，确保连线是正确的
    df_group_sorted = df_group.sort_values(by='孕妇BMI')
    plt.plot(df_group_sorted['孕妇BMI'], df_group_sorted['Y染色体浓度'],linestyle='--')

# 4. 添加图表元素
plt.title('不同分组的散点图及连线', fontsize=16)
plt.xlabel('X轴', fontsize=12)
plt.ylabel('Y轴', fontsize=12)
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.show()



groups = df_boy['孕妇代码'].unique()




# 2. 创建图表
plt.figure(figsize=(10, 6))

for i, group in enumerate(groups):
    df_group = df_boy[df_boy['孕妇代码'] == group]

    # 绘制散点图
    plt.scatter(df_group['孕妇BMI'], df_group['Y染色体浓度'],
                label=f'分组 {group}',
                color='blue',
                marker='o',
                s=10,
                alpha=0.20)  # s参数用于调整点的大小

    # 绘制连线
    # 按照X轴顺序对数据进行排序，确保连线是正确的
    df_group_sorted = df_group.sort_values(by='孕妇BMI')
    plt.plot(df_group_sorted['孕妇BMI'], df_group_sorted['Y染色体浓度'],linestyle='--')

# 4. 添加图表元素
plt.title('不同分组的散点图及连线', fontsize=16)
plt.xlabel('X轴', fontsize=12)
plt.ylabel('Y轴', fontsize=12)
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.savefig('pic/BMI-y染色体浓度-散点图-连线')
plt.show()




plt.figure(figsize=(10, 6))

for i, group in enumerate(groups):
    df_group = df_boy[df_boy['孕妇代码'] == group]

    # 绘制散点图
    plt.scatter(df_group['孕周'], df_group['Y染色体浓度'],
                label=f'分组 {group}',
                color='blue',
                marker='o',
                s=10,
                alpha=0.20)  # s参数用于调整点的大小

    # 绘制连线
    # 按照X轴顺序对数据进行排序，确保连线是正确的
    df_group_sorted = df_group.sort_values(by='孕周')
    plt.plot(df_group_sorted['孕周'], df_group_sorted['Y染色体浓度'],linestyle='--')

# 4. 添加图表元素
plt.title('不同分组的散点图及连线', fontsize=16)
plt.xlabel('X轴', fontsize=12)
plt.ylabel('Y轴', fontsize=12)
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.savefig('pic/孕期-y染色体浓度-散点图-连线')
plt.show()







