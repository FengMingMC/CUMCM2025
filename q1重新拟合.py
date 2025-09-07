from q1数据清洗 import *

import numpy as np

calculate_frame = {
    'Week': calculate_frame["孕周"],
    'BMI': calculate_frame["孕妇BMI"],
    "YContent": calculate_frame["Y染色体浓度"]}

def funcTest(x_data,a,b,c,d):
    x,y = x_data
    return a + b*x + c*x**2 + d*np.log10(y+1)

from scipy.optimize import curve_fit

initial_guess = [0.1,
                 0.1,
                 0.1,
                 0.1]

x_data_packed = (np.array(calculate_frame["BMI"]), np.array(calculate_frame["Week"]))
y_data = np.array(calculate_frame["YContent"])

# try:
#     # 进行拟合
#     params, covariance = curve_fit(funcTest, x_data_packed, y_data, p0=initial_guess)
#
#     a_fit, b_fit, c_fit, d_fit, = params
#
#     print("\n\n拟合得到的参数：")
#     print(f"a = {a_fit}")
#     print(f"b = {b_fit}")
#     print(f"c = {c_fit}")
#     print(f"d = {d_fit}")
#
#     y_predicted = funcTest(x_data_packed, params[0], params[1], params[2], params[3])
#     ss_res = np.sum((y_data - y_predicted) ** 2)
#     ss_tot = np.sum((y_data - np.mean(y_data))**2) # 总平方和
#     r_squared = 1 - (ss_res / ss_tot)
#     print(f"R^2 = {r_squared}")
#
# except RuntimeError as e:
#     print(f"拟合失败：{e}")
#     print("请检查您的数据和初始猜测参数。")

from sklearn.metrics import mean_squared_error


import matplotlib.pyplot as plt
import scipy.stats as stats


try:
    # 进行拟合
    params, covariance = curve_fit(funcTest, x_data_packed, y_data, p0=initial_guess)

    a_fit, b_fit, c_fit, d_fit, = params

    print("\n\n拟合得到的参数：")
    print(f"a = {a_fit}")
    print(f"b = {b_fit}")
    print(f"c = {c_fit}")
    print(f"d = {d_fit}")

    y_predicted = funcTest(x_data_packed, params[0], params[1], params[2], params[3])
    ss_res = np.sum((y_data - y_predicted) ** 2)
    ss_tot = np.sum((y_data - np.mean(y_data))**2) # 总平方和
    r_squared = 1 - (ss_res / ss_tot)
    print(f"R^2 = {r_squared}")


    residuals = y_data - y_predicted


    plt.figure(figsize=(8, 6))
    stats.probplot(residuals, dist="norm", plot=plt)
    plt.title("残差的 Q-Q 图")
    plt.xlabel("理论分位数")
    plt.ylabel("样本分位数")
    plt.grid(True)
    plt.savefig("pic/残差的 Q-Q 图.pdf")
    plt.show()
    # --- End of Q-Q plot generation ---
    n = len(y_data)  # 样本数量
    k_full = len(params)  # 完整模型的参数数量 (a, b, c, d)

    # 假设零假设是所有系数 (b, c, d) 都为零，模型只有截距 a
    # 完整模型自由度 df_full = n - k_full
    # 受限模型自由度 df_restricted = n - 1 (只有截距 a)

    # 回归平方和 (SSR)
    ssr = ss_tot - ss_res

    # 均方回归 (MSR) - 对应完整模型
    # MSR = SSR / (df_full - df_restricted)  # 这是另一种计算方式
    # 或者更直接地：
    df_regression = k_full - 1  # 自变量的自由度 (b, c, d)
    msr = ssr / df_regression

    # 均方残差 (MSE) - 对应完整模型
    df_residual = n - k_full
    mse = ss_res / df_residual

    # 计算 F 统计量
    f_statistic = msr / mse

    # 计算 P 值
    # P 值是 F 统计量大于该值的概率
    p_value = stats.f.sf(f_statistic, df_regression, df_residual)

    print("\n\n--- F 检验结果 ---")
    print(f"F 统计量: {f_statistic:.4f}")
    print(f"P 值: {p_value:.4f}")

    if p_value < 0.05:
        print("结果显著: 拒绝原假设，表明模型中的至少一个自变量（BMI或Week的函数）对YContent有显著影响。")
    else:
        print("结果不显著: 未能拒绝原假设，模型可能不是显著的。")

except RuntimeError as e:
    print(f"拟合失败：{e}")
    print("请检查您的数据和初始猜测参数。")

mse = mean_squared_error(y_data, y_predicted)
rmse = np.sqrt(mse)
print(f"模型 RMSE: {rmse:.4f}")

# 计算 AIC 和 BIC
n = len(y_data) # 样本数量
k = len(params) # 参数数量
bic = n * np.log(ss_res / n) + k * np.log(n)
aic = n * np.log(ss_res / n) + 2 * k

print(f"模型的 AIC: {aic:.4f}")
print(f"模型的 BIC: {bic:.4f}")
