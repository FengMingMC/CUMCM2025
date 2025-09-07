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

except RuntimeError as e:
    print(f"拟合失败：{e}")
    print("请检查您的数据和初始猜测参数。")

