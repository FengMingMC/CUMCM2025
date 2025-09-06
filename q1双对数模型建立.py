import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
from scipy.stats import pearsonr, spearmanr
import statsmodels.api as sm
import seaborn as sns
from scipy.optimize import curve_fit

plt.rcParams['font.sans-serif'] = ['SimHei']
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


groups = df_boy['孕妇代码'].unique()


def pregnancyTimes(row):
    # 怀孕次数转化
    mapping = {
        '1': 1,
        '2': 2,
        '≥3': 3  # 或者其他你认为合理的数字
    }

    # 获取 '生产次数' 列的值
    pregnancytimes_str = str(row['生产次数'])

    # 根据映射获取数字，如果没有匹配则返回 None
    pregnancytimes_int = mapping.get(pregnancytimes_str, None)

    # 检查映射结果
    if pregnancytimes_int is not None and pregnancytimes_int == 1:
        return 0
    else:
        return 1


df_boy['PregnancyTimes'] = df_boy.apply(pregnancyTimes, axis=1)

def get_mode(x):
    """
    Calculates the mode of a Series. Returns the first mode if multiple exist,
    or NaN if the Series is empty.
    """
    if x.empty:
        return np.nan
    mode_values = x.mode()
    return mode_values[0] if not mode_values.empty else np.nan

# 确定分组的列
group_cols = ['孕妇代码', '孕周']

# 找出 DataFrame 中所有列的类型
all_cols = df_boy.columns.tolist()

# 分离出数值列和非数值列，排除分组列
numeric_cols = df_boy.select_dtypes(include=np.number).columns.tolist()
# 确保分组列不被重复处理
numeric_cols = [col for col in numeric_cols if col not in group_cols]

non_numeric_cols = df_boy.select_dtypes(exclude=np.number).columns.tolist()
# 确保分组列不被重复处理（虽然通常它们是数值型）
non_numeric_cols = [col for col in non_numeric_cols if col not in group_cols]

# 创建聚合字典
agg_dict = {}

# 对于数值列，应用中位数聚合
for col in numeric_cols:
    agg_dict[col] = 'median'

# 对于非数值列，应用众数聚合
for col in non_numeric_cols:
    agg_dict[col] = get_mode

df_temp = df_boy.groupby(group_cols).agg(agg_dict).reset_index()

df_boy = df_temp


df_temp = df_boy[df_boy['Y染色体浓度'] <= 0.2]
df_boy = df_temp
for i, group in enumerate(groups):
    df_group = df_boy[df_boy['孕妇代码'] == group]

print(df_boy)




calculate_frame = pd.DataFrame({
    "Week":df_boy["孕周"],
    "BMI": df_boy["孕妇BMI"],
    "BMILn": np.log(df_boy["孕妇BMI"]),
    "BMILog": np.log10(df_boy["孕妇BMI"]),
    "BMISqrt": np.sqrt(df_boy["孕妇BMI"]),
    "GC": df_boy["GC含量"],
    "原始读段数": df_boy["原始读段数"],
    "在参考基因组上比对的比例": df_boy["在参考基因组上比对的比例"],
    "重复读段的比例": df_boy["重复读段的比例"],
    "怀孕次数": df_boy['PregnancyTimes'],
    "生产次数": df_boy["生产次数"],
    "X染色体浓度": df_boy["X染色体浓度"],
    "YContent": df_boy["Y染色体浓度"],
    "ID": df_boy["孕妇代码"]
})







'''尝试归一化
column_to_normalize = 'BMI'
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
calculate_frame[column_to_normalize] = scaler.fit_transform(calculate_frame[[column_to_normalize]])


column_to_normalize = 'Week'
scaler = MinMaxScaler()
calculate_frame[column_to_normalize] = scaler.fit_transform(calculate_frame[[column_to_normalize]])

column_to_normalize = 'YContent'
scaler = MinMaxScaler()
calculate_frame[column_to_normalize] = scaler.fit_transform(calculate_frame[[column_to_normalize]])
'''


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

b_fit = coefficients[0]
c_fit = coefficients[1]
log_a_fit = coefficients[2]
a_fit = np.exp(log_a_fit)

print(f"拟合结果 (通过对数转换):")
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

# 进行三次多项式拟合


#
# results = model.fit()
#
# print(results.summary())

# def smallTest(calculate_frame: pd.DataFrame):
def func2(x_data, a, b, c, d, e, f):
    # x_data 现在是一个元组，包含 (x, y)
    x, y = x_data
    return a * x**2 + b * x * y + c * y**2 + d * x + e * y + f


from scipy.optimize import curve_fit

# 初始猜测参数值 (可以根据您的数据大致估计，这有助于提高拟合效率)
initial_guess = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
x_data_packed = (np.array(calculate_frame["Week"]), np.array(calculate_frame["BMI"]))
y_data = np.array(calculate_frame["YContent"])

try:
    # 进行拟合
    params, covariance = curve_fit(func2, x_data_packed, y_data, p0=initial_guess)

    # 提取拟合得到的参数
    a_fit, b_fit, c_fit, d_fit, e_fit, f_fit = params

    print("\n\n拟合得到的参数：")
    print(f"a = {a_fit}")
    print(f"b = {b_fit}")
    print(f"c = {c_fit}")
    print(f"d = {d_fit}")
    print(f"e = {e_fit}")
    print(f"f = {f_fit}")

    # 可以选择计算拟合优度 R^2 等指标来评估拟合效果
    # z_predicted = func(params, x_data, y_data)
    # ss_res = np.sum((z_data - z_predicted)**2) # 残差平方和
    # ss_tot = np.sum((z_data - np.mean(z_data))**2) # 总平方和
    # r_squared = 1 - (ss_res / ss_tot)
    # print(f"R^2 = {r_squared}")

except RuntimeError as e:
    print(f"拟合失败：{e}")
    print("请检查您的数据和初始猜测参数。")

after_treatment = a_fit * calculate_frame["Week"]**2 + b_fit * calculate_frame["Week"] * calculate_frame["BMI"] + c_fit * calculate_frame["BMI"]**2 + d_fit * calculate_frame["Week"] + e_fit * calculate_frame["BMI"] + f_fit

# 执行配对样本t检验
t_statistic, p_value = stats.ttest_rel(before_treatment, after_treatment)
plt.figure(figsize=(10, 6))
plt.plot(np.array(df_boy["序号"]), np.array(before_treatment), marker='o', linestyle='-', color='b', label='数组 1')
plt.plot(np.array(df_boy["序号"]), np.array(after_treatment), marker='x', linestyle='--', color='r', label='数组 2')
plt.grid(True) # 添加网格线，可选
plt.show()
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

    print("\n\n拟合得到的参数：")
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

    # 可以选择计算拟合优度 R^2 等指标来评估拟合效果
    # z_predicted = func(params, x_data, y_data)
    # ss_res = np.sum((z_data - z_predicted)**2) # 残差平方和
    # ss_tot = np.sum((z_data - np.mean(z_data))**2) # 总平方和
    # r_squared = 1 - (ss_res / ss_tot)
    # print(f"R^2 = {r_squared}")



    after_treatment = a_fit*calculate_frame["Week"]**3 + b_fit*calculate_frame["Week"]**2*calculate_frame["BMI"] + c_fit*calculate_frame["Week"]*calculate_frame["BMI"]**2 + d_fit*calculate_frame["BMI"]**3 + e_fit*calculate_frame["Week"]**2 + f_fit*calculate_frame["Week"]*calculate_frame["BMI"] + g_fit*calculate_frame["BMI"]**2 + h_fit*calculate_frame["Week"] + i_fit*calculate_frame["BMI"] + j_fit


    # 执行配对样本t检验
    t_statistic, p_value = stats.ttest_rel(before_treatment, after_treatment)
    plt.figure(figsize=(10, 6))
    plt.plot(np.array(df_boy["序号"]), np.array(before_treatment), marker='o', linestyle='-', color='b', label='数组 1')
    plt.plot(np.array(df_boy["序号"]), np.array(after_treatment), marker='x', linestyle='--', color='r', label='数组 2')
    plt.grid(True)  # 添加网格线，可选
    plt.show()
    print(f"\n配对样本t检验:")
    print(f"T统计量: {t_statistic:.4f}")
    print(f"P值: {p_value:.4f}")
except RuntimeError as e:
    print(f"拟合失败：{e}")
    print("请检查您的数据和初始猜测参数。")



