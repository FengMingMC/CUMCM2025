import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
from scipy.stats import pearsonr, spearmanr
import statsmodels.api as sm
import seaborn as sns
from scipy.optimize import curve_fit
from q1数据清洗 import *

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


calculate_frame = {
    'Week': calculate_frame["孕周"],
    'BMI': calculate_frame["孕妇BMI"],
    "YContent": calculate_frame["Y染色体浓度"]}

data = {
    'Week': calculate_frame["Week"],
    'BMI': calculate_frame["BMI"],
    "YContent": calculate_frame["YContent"]}
df = pd.DataFrame(data)

df['WeekLog'] = np.log(df['Week'])
df['BMILog'] = np.log(df['BMI'])
df['YContentLog'] = np.log(df['YContent'])

A = np.vstack([df['WeekLog'], df['BMILog'], np.ones(len(df['BMILog']))]).T

coefficients, residuals, rank, s = np.linalg.lstsq(A, df['YContentLog'], rcond=None)

y_true = df['YContentLog']
y_mean = np.mean(y_true)
ss_tot = np.sum((y_true - y_mean)**2)

ss_res = residuals[0] # 取残差平方和的第一个值
r2 = 1 - (ss_res / ss_tot)
print(f"\n\nR2: {r2}")


b_fit = coefficients[0]
c_fit = coefficients[1]
log_a_fit = coefficients[2]
a_fit = np.exp(log_a_fit)

print(f"拟合双对数结果:")
print(f"a = {a_fit:.4f}")
print(f"b = {b_fit:.4f}")
print(f"c = {c_fit:.4f}")

from scipy import stats
before_treatment = calculate_frame["YContent"]
after_treatment = a_fit * calculate_frame["Week"]**b_fit * calculate_frame["BMI"]**c_fit

# 执行配对样本t检验
t_statistic, p_value = stats.ttest_rel(before_treatment, after_treatment)

'''绘制折线图
plt.figure(figsize=(10, 6))
plt.plot(np.array(df_boy["序号"]), np.array(before_treatment), marker='o', linestyle='-', color='b', label='数组 1')
plt.plot(np.array(df_boy["序号"]), np.array(after_treatment), marker='x', linestyle='--', color='r', label='数组 2')
plt.grid(True) # 添加网格线，可选
plt.show()'''
print(f"\n配对样本t检验:")
print(f"T统计量: {t_statistic:.4f}")
print(f"P值: {p_value:.4f}")


#二次函数 拟合

def func2(x_data, a, b, c, d, e, f):
    # x_data 现在是一个元组，包含 (x, y)
    x, y = x_data
    return a * x**2 + b * x * y + c * y**2 + d * x + e * y + f


from scipy.optimize import curve_fit

initial_guess = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
x_data_packed = (np.array(calculate_frame["Week"]), np.array(calculate_frame["BMI"]))
y_data = np.array(calculate_frame["YContent"])

try:
    # 进行拟合
    params, covariance = curve_fit(func2, x_data_packed, y_data, p0=initial_guess)

    # 提取拟合得到的参数
    a_fit, b_fit, c_fit, d_fit, e_fit, f_fit = params

    print("\n\n拟合二次函数得到的参数：")
    print(f"a = {a_fit}")
    print(f"b = {b_fit}")
    print(f"c = {c_fit}")
    print(f"d = {d_fit}")
    print(f"e = {e_fit}")
    print(f"f = {f_fit}")



except RuntimeError as e:
    print(f"拟合失败：{e}")
    print("请检查您的数据和初始猜测参数。")

after_treatment = a_fit * calculate_frame["Week"]**2 + b_fit * calculate_frame["Week"] * calculate_frame["BMI"] + c_fit * calculate_frame["BMI"]**2 + d_fit * calculate_frame["Week"] + e_fit * calculate_frame["BMI"] + f_fit

# 执行配对样本t检验
t_statistic, p_value = stats.ttest_rel(before_treatment, after_treatment)
# plt.figure(figsize=(10, 6))
# plt.plot(np.array(df_boy["序号"]), np.array(before_treatment), marker='o', linestyle='-', color='b', label='数组 1')
# plt.plot(np.array(df_boy["序号"]), np.array(after_treatment), marker='x', linestyle='--', color='r', label='数组 2')
# plt.grid(True) # 添加网格线，可选
# plt.show()
print(f"\n配对样本t检验:")
print(f"T统计量: {t_statistic:.4f}")
print(f"P值: {p_value:.4f}")

def func3 (x_data, a, b, c, d, e, f, g ,h ,i ,j):
    x, y = x_data
    return a*x**3 + b*x**2*y + c*x*y**2 + d*y**3 + e*x**2 + f*x*y + g*y**2 + h*x + i*y + j

initial_guess = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
x_data_packed = (np.array(calculate_frame["Week"]), np.array(calculate_frame["BMI"]))
y_data = np.array(calculate_frame["YContent"])

try:
    # 进行拟合
    params, covariance = curve_fit(func3, x_data_packed, y_data, p0=initial_guess)

    # 提取拟合得到的参数
    a_fit, b_fit, c_fit, d_fit, e_fit, f_fit, g_fit, h_fit, i_fit, j_fit = params

    print("\n\n拟合三次函数得到的参数：")
    print(f"a = {a_fit}")
    print(f"b = {b_fit}")
    print(f"c = {c_fit}")
    print(f"d = {d_fit}")
    print(f"e = {e_fit}")
    print(f"f = {f_fit}")
    print(f"g = {g_fit}")
    print(f"h = {h_fit}")
    print(f"i = {i_fit}")
    print(f"j = {j_fit}")



    after_treatment = a_fit*calculate_frame["Week"]**3 + b_fit*calculate_frame["Week"]**2*calculate_frame["BMI"] + c_fit*calculate_frame["Week"]*calculate_frame["BMI"]**2 + d_fit*calculate_frame["BMI"]**3 + e_fit*calculate_frame["Week"]**2 + f_fit*calculate_frame["Week"]*calculate_frame["BMI"] + g_fit*calculate_frame["BMI"]**2 + h_fit*calculate_frame["Week"] + i_fit*calculate_frame["BMI"] + j_fit


    # 执行配对样本t检验
    t_statistic, p_value = stats.ttest_rel(before_treatment, after_treatment)

    print(f"\n配对样本t检验:")
    print(f"T统计量: {t_statistic:.4f}")
    print(f"P值: {p_value:.4f}")
except RuntimeError as e:
    print(f"拟合失败：{e}")
    print("请检查您的数据和初始猜测参数。")



