import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
from scipy.stats import pearsonr, spearmanr
import statsmodels.api as sm
import seaborn as sns
from scipy.optimize import curve_fit

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
    "孕周":df_boy["孕周"],
    "孕妇BMI": df_boy["孕妇BMI"],
    "BMI取自然对数": np.log(df_boy["孕妇BMI"]),
    "BMI取常用对数": np.log10(df_boy["孕妇BMI"]),
    "BMI开方": np.sqrt(df_boy["孕妇BMI"]),
    "GC含量": df_boy["GC含量"],
    "原始读段数": df_boy["原始读段数"],
    "在参考基因组上比对的比例": df_boy["在参考基因组上比对的比例"],
    "重复读段的比例": df_boy["重复读段的比例"],
    "怀孕次数": df_boy['PregnancyTimes'],
    "生产次数": df_boy["生产次数"],
    "X染色体浓度": df_boy["X染色体浓度"],
    "Y染色体浓度": df_boy["Y染色体浓度"]
})








#
#
#
#
#
# a_true, b_true, c_true, d_true = 5, 1.5, 0.8, 0 #假设参数值
# noise = np.random.normal(0, 20, calculate_frame.shape[0]) # 添加一些噪声
#
# yContent = a_true * (calculate_frame["孕周"]**b_true) * (calculate_frame["孕妇BMI"]**c_true) + noise
#
#


data = {
    '孕周': calculate_frame["孕周"],
    '孕妇BMI': calculate_frame["孕妇BMI"],
    "Y染色体浓度": calculate_frame["Y染色体浓度"]}
df = pd.DataFrame(data)
#
df['孕周Log'] = np.log(df['孕周'])
df['孕妇BMILog'] = np.log(df['孕妇BMI'])
df['y染色体含量Log'] = np.log(df['Y染色体浓度'])

A = np.vstack([df['孕周Log'], df['孕妇BMILog'], np.ones(len(df['孕妇BMILog']))]).T

coefficients, residuals, rank, s = np.linalg.lstsq(A, df['y染色体含量Log'], rcond=None)

b_fit = coefficients[0]
c_fit = coefficients[1]
log_a_fit = coefficients[2]
a_fit = np.exp(log_a_fit)

print(f"拟合结果 (通过对数转换):")
print(f"a = {a_fit:.4f}")
print(f"b = {b_fit:.4f}")
print(f"c = {c_fit:.4f}")
# print("--- 示例数据前5行 ---")
# print(df.head())
# print("\n")
#
# df['孕周Log'] = np.log(df['孕周'])
# df['孕妇BMILog'] = np.log(df['孕妇BMI'])
# df['y染色体含量Log'] = np.log(df['Y染色体浓度'])
#
#
# print("--- 对数转换后的数据前5行 ---")
# print(df[['孕周Log', '孕妇BMILog', 'y染色体含量Log']].head())
# print("\n")
#
# Y = df['y染色体含量Log']
# X = df[['孕周Log', '孕妇BMILog']]
# X = sm.add_constant(X)
#
# model = sm.OLS(Y, X)
# results = model.fit()
#
# r_squared = results.rsquared
#
# print(f"模型的R² (决定系数) 为: {r_squared:.4f}")
#
# beta_0 = results.params['const']     # log(a)
# beta_1 = results.params['孕周Log'] # b
# beta_2 = results.params['孕妇BMILog'] # c
#
# print("--- 线性回归结果 ---")
# print(results.summary())
# print("\n")
#
# a = np.exp(beta_0)
# b = beta_1
# c = beta_2
#
# print(f"拟合得到的模型参数 (d=0):")
# print(f"  a = {a:.4f}")
# print(f"  b = {b:.4f}")
# print(f"  c = {c:.4f}")
# print(f"  d = 0 (假设)")
# print("\n")
#
#
# y_pred = results.predict(X)
# residuals = Y - y_pred
#
# plt.figure(figsize=(12, 6))
#
# # 残差与拟合值的散点图
# plt.subplot(1, 2, 1)
# sns.scatterplot(x=y_pred, y=residuals)
# plt.axhline(0, color='red', linestyle='--')
# plt.title('残差与拟合值散点图')
# plt.xlabel('拟合值 (ln(Y染色体浓度))')
# plt.ylabel('残差')
#
# # 残差的正态性Q-Q图
# plt.subplot(1, 2, 2)
# sm.qqplot(residuals, line='s', ax=plt.gca())
# plt.title('残差正态性 Q-Q 图')
#
# plt.tight_layout()
# plt.show()
#
from scipy import stats
# import numpy as np

# 假设有配对数据 (例如，处理前后的测量值)
before_treatment = calculate_frame["Y染色体浓度"]
after_treatment = a_fit * calculate_frame["孕周"]**b_fit * calculate_frame["孕妇BMI"]**c_fit

# 执行配对样本t检验
t_statistic, p_value = stats.ttest_rel(before_treatment, after_treatment)

print(f"\n配对样本t检验:")
print(f"T统计量: {t_statistic:.4f}")
print(f"P值: {p_value:.4f}")


