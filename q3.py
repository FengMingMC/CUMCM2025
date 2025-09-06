from q1数据清洗 import *
import matplotlib.pyplot as plt
import pandas as pd

data = {
    'Week': calculate_frame["孕周"],
    'BMI': calculate_frame["孕妇BMI"],
    'Age': calculate_frame["年龄"],
    'Height': calculate_frame["身高"],
    'Weight': calculate_frame["体重"],
    "YContent": calculate_frame["Y染色体浓度"]}

data['WeekLog'] = np.log(data['Week'])
data['BMILog'] = np.log(data['BMI'])
data['AgeLog'] = np.log(data['Age'])
data['HeightLog'] = np.log(data['Height'])
data['WeightLog'] = np.log(data['Weight'])
data['YContentLog'] = np.log(data['YContent'])

A = np.vstack([data['WeekLog'],
               data['BMILog'],
               data['HeightLog'],
               data['WeightLog'],
               data['AgeLog'],
               np.ones(len(data['YContentLog']))]).T

coefficients, residuals, rank, s = np.linalg.lstsq(A, data['YContentLog'])

b_fit = coefficients[0]
c_fit = coefficients[1]
d_fit = coefficients[2]
e_fit = coefficients[3]
f_fit = coefficients[4]
log_a_fit = coefficients[5]
a_fit = np.exp(log_a_fit)

print(f"拟合结果 (通过对数转换):")
print(f"lna = {log_a_fit:.4f}")
print(f"a = {a_fit:.4f}")
print(f"b = {b_fit:.4f}")
print(f"c = {c_fit:.4f}")
print(f"d = {d_fit:.4f}")
print(f"e = {e_fit:.4f}")
print(f"f = {f_fit:.4f}")






























