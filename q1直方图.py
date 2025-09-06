# %% 1. 导入所需库
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



# %% 2. 设置绘图样式与中文字体
#    确保图表中的中文能够正确显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
# sns.set_style("whitegrid") # 设置seaborn的绘图风格

# %% 3. 读取并准备数据
#    请确保 "男胎检测数据.csv" 文件与此脚本在同一个文件夹下
file_path = '男胎检测数据.csv'
try:
    # 尝试使用UTF-8编码读取
    data = pd.read_csv(file_path, encoding='utf-8')
except UnicodeDecodeError:
    # 如果UTF-8失败，尝试使用GBK编码，这在处理中文Windows环境下生成的文件时很常见
    data = pd.read_csv(file_path, encoding='gbk')

# 为了编程方便，将关键的中文列名重命名为英文
data.rename(columns={
    "孕妇代码": "PatientID",
    "孕妇BMI": "BMI"
}, inplace=True)

print("数据加载完成，原始数据共有 {} 行。".format(len(data)))
# 打印数据前5行以作检查
# print("数据预览:\n", data.head())


# %% 4. 计算每个孕妇的BMI平均值
#    这是本任务的核心步骤
#    - 使用 .groupby('PatientID') 按孕妇对数据进行分组
#    - 选择 ['BMI'] 列
#    - 使用 .mean() 计算每个组（即每个孕妇）的BMI平均值
avg_bmi_per_patient = data.groupby('PatientID')['BMI'].mean()

# 打印出独立孕妇的数量和计算结果的预览
print(f"\n计算完成，共找到 {len(avg_bmi_per_patient)} 位独立孕妇。")
print("每位孕妇的平均BMI（部分数据预览）:\n", avg_bmi_per_patient.head())


# %% 5. 绘制BMI分布的柱状图（直方图）
print("\n正在绘制BMI分布直方图...")

# 创建一个图形窗口，设置尺寸
plt.figure(figsize=(12, 7))

# 使用seaborn的histplot函数来绘制直方图，它能自动计算频数并分组
# - data=avg_bmi_per_patient: 指定要绘制的数据
# - bins=25: 将BMI范围分成25个“桶”，可以调整这个数值来改变柱子的粗细
# - kde=True: 同时绘制一条核密度估计曲线，以平滑地展示分布趋势
sns.histplot(data=avg_bmi_per_patient, bins=25, kde=True)

# 添加图表标题和坐标轴标签
plt.title('每位男胎孕妇平均BMI的分布直方图', fontsize=16)
plt.xlabel('平均BMI (Average BMI)', fontsize=12)
plt.ylabel('孕妇人数 (Number of Patients)', fontsize=12)

# 显示网格线
plt.grid(True, linestyle='--', alpha=0.6)

# 显示图表
plt.show()

print("图表绘制完成。")