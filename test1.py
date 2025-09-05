# %% 1. 导入库并准备数据 (Import Libraries & Prepare Data)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pygam import LinearGAM, s, l, r

# 设置绘图样式，让图片更美观
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# --- 读取和预处理数据 ---
# 请确保 "男胎检测数据.csv" 文件在您的工作目录下
try:
    data = pd.read_csv("男胎检测数据.csv", encoding='utf-8')
except:
    data = pd.read_csv("男胎检测数据.csv", encoding='gbk')

# 重命名列以便于访问
data.rename(columns={
    "孕妇代码": "PatientID",
    "检测孕周": "GestWeekStr",
    "Y染色体浓度": "Y_Concentration",
    "孕妇BMI": "BMI",
    "GC含量": "GC_Content",
    "年龄": "Age"
}, inplace=True)

# 定义一个函数来转换孕周格式
def parse_gest_week(week_str):
    if isinstance(week_str, str):
        if '+' in week_str:
            parts = week_str.split('w+')
            return float(parts[0]) + float(parts[1]) / 7
        else:
            return float(week_str.replace('w', ''))
    return np.nan

data['GestWeek'] = data['GestWeekStr'].apply(parse_gest_week)

# 筛选建模所需的列，并移除任何包含缺失值的行
required_cols = ["Y_Concentration", "GestWeek", "BMI", "GC_Content", "Age", "PatientID"]
data_clean = data[required_cols].dropna()

# 将PatientID转换为可用于随机效应的整数编码
data_clean['PatientID_code'] = data_clean['PatientID'].astype('category').cat.codes

print(f"数据清洗后，剩余 {len(data_clean)} 行数据用于分析。")

# %% 2. 准备模型输入 (Prepare Model Inputs)

# 定义自变量X和因变量y
# 注意：pygam要求X是一个numpy数组
X = data_clean[['GestWeek', 'BMI', 'GC_Content', 'Age', 'PatientID_code']].values
y = data_clean['Y_Concentration'].values


# %% 3. 建立并求解GAMM模型 (Build and Fit the GAMM)

# 定义模型公式:
# s(0) -> 对第0列(GestWeek)使用平滑项
# s(1) -> 对第1列(BMI)使用平滑项
# l(2) -> 对第2列(GC_Content)使用线性项
# l(3) -> 对第3列(Age)使用线性项
# r(4) -> 对第4列(PatientID_code)使用随机效应(随机截距)
# n_splines参数可以控制曲线的复杂度/灵活性
gam_formula = s(0, n_splines=12) + s(1, n_splines=12) + l(2) + l(3) + r(4)

# 创建并拟合模型
# distribution='gamma'或'inv_gauss'可能更适合比例数据，但'normal'是最稳健的起点
gam = LinearGAM(gam_formula, distribution='normal', link='identity').fit(X, y)

# --- 4. 查看和解读模型结果 ---
print("\n--- GAMM模型结果摘要 ---")
gam.summary()


# %% 5. 可视化函数关系（关键产出）---
print("\n正在绘制各变量的函数关系图...")

# 获取变量名用于绘图
feature_names = ['检测孕周 (GestWeek)', '孕妇BMI (BMI)', 'GC含量 (GC_Content)', '年龄 (Age)']

fig, axes = plt.subplots(1, 4, figsize=(20, 5))

for i, ax in enumerate(axes):
    # 为每个项生成部分依赖图
    # gam.terms[i] 对应公式中的第i个项
    pdep, confi = gam.partial_dependence(term=i, X=X, width=0.95)
    
    ax.plot(X[:, i], pdep)
    ax.plot(X[:, i], confi, c='r', ls='--')
    ax.set_title(feature_names[i])
    ax.set_xlabel(feature_names[i])
    if i == 0:
        ax.set_ylabel("对Y染色体浓度的影响")
    ax.grid(True)

plt.suptitle("GAMM各变量的部分依赖图 (Partial Dependence Plots)", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()