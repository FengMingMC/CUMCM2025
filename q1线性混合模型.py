import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
import statsmodels.api as sm

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

# groups = df_boy['孕妇代码'].unique()
#

import statsmodels.formula.api as smf

# 假设您有一个名为 'my_data' 的 DataFrame
# 并且它包含以下列：
# - 'response_var': 因变量
# - 'fixed_var1', 'fixed_var2': 固定效应自变量
# - 'group_id': 分组变量 (例如，病人ID, 班级ID, 医院ID 等)

# 创建一个示例 DataFrame (请替换为您自己的数据)
# np.random.seed(42)
# n_groups = 10
# n_obs_per_group = 20
# total_obs = n_groups * n_obs_per_group

data = {
    'Y染色体浓度': df_boy['Y染色体浓度'],
    '孕周': df_boy['孕周'],
    '孕妇BMI': df_boy['孕妇BMI'],
    'GC含量': df_boy['GC含量'],
    '孕妇代码': df_boy['孕妇代码']
}
calculate_frame = pd.DataFrame(data)

# 为模型添加一些结构：
# 1. 固定效应：response_var = intercept + beta1*fixed_var1 + beta2*fixed_var2 + ...
# 2. 随机效应：假设每个 group_id 有一个随机截距 (random intercept)
#    这表示每个组的平均响应值会围绕全局平均值有所偏离。
#    公式中的 '0 + fixed_var1' 是为了示范如何给固定效应添加随机斜率，
#    但对于初学者，通常从随机截距模型开始。
#    一个简单的随机截距模型公式可以是: "response_var ~ fixed_var1 + fixed_var2"
#    statsmodels 默认如果指定了 groups，会尝试拟合随机截距。
#    如果需要更复杂的随机效应（如随机斜率），则需要在 formula 中明确指定，
#    例如 'response_var ~ fixed_var1 + fixed_var2 + (fixed_var1|group_id)'
#    这里我们先从最常见的随机截距模型开始：

# 定义模型公式：因变量 ~ 固定效应1 + 固定效应2 + ...
# statsmodels 会自动处理截距项
formula = "Y染色体浓度 ~ 孕周 + 孕妇BMI + GC含量"
model = smf.mixedlm(formula, data=calculate_frame, groups=calculate_frame["孕妇代码"])


# 创建并拟合线性混合模型
# groups 参数指定了用于定义随机效应的分组变量
# 默认情况下，它会拟合一个随机截距模型
result = model.fit()

# 打印模型摘要
print("--- 线性混合模型结果摘要 ---")
print(result.summary())




















# def get_mode(x):
#     """
#     Calculates the mode of a Series. Returns the first mode if multiple exist,
#     or NaN if the Series is empty.
#     """
#     if x.empty:
#         return np.nan
#     mode_values = x.mode()
#     return mode_values[0] if not mode_values.empty else np.nan
#
# # 确定分组的列
# group_cols = ['孕妇代码', '孕周']
#
# # 找出 DataFrame 中所有列的类型
# all_cols = df_boy.columns.tolist()
#
# # 分离出数值列和非数值列，排除分组列
# numeric_cols = df_boy.select_dtypes(include=np.number).columns.tolist()
# # 确保分组列不被重复处理
# numeric_cols = [col for col in numeric_cols if col not in group_cols]
#
# non_numeric_cols = df_boy.select_dtypes(exclude=np.number).columns.tolist()
# # 确保分组列不被重复处理（虽然通常它们是数值型）
# non_numeric_cols = [col for col in non_numeric_cols if col not in group_cols]
#
# # 创建聚合字典
# agg_dict = {}
#
# # 对于数值列，应用中位数聚合
# for col in numeric_cols:
#     agg_dict[col] = 'median'
#
# # 对于非数值列，应用众数聚合
# for col in non_numeric_cols:
#     agg_dict[col] = get_mode
#
# df_temp = df_boy.groupby(group_cols).agg(agg_dict).reset_index()
#
# df_boy = df_temp

#
# df_temp = df_boy[df_boy['Y染色体浓度'] <= 0.2]
# df_boy = df_temp
# for i, group in enumerate(groups):
#     df_group = df_boy[df_boy['孕妇代码'] == group]
#
# print(df_boy)
#
# calculate_frame = pd.DataFrame({
#     "孕周":df_boy["孕周"],
#     # "孕妇BMI": df_boy["孕妇BMI"],
#     "Y染色体浓度": df_boy["Y染色体浓度"]
# })
# #
# # print(calculate_frame)
# #
# # # 定义因变量 (y)
# # y = df_boy["Y染色体浓度"]
# #
# # # 定义自变量 (X)
# # # 选择 'age' 和 'income' 作为自变量
# # X = df_boy[['孕周', '孕妇BMI']]
# #
# # # 添加常数项 (截距)
# # # sm.add_constant() 会在自变量 X 的左侧添加一列全为 1 的列
# # X = sm.add_constant(X)
# #
# # # 使用 Ordinary Least Squares (OLS) 方法拟合模型
# # # OLS 是多元线性回归中最常用的方法
# # model = sm.OLS(y, X).fit()
# #
# # print(model.summary())
# print(calculate_frame)
#
# # calculate_frame_temp = pd.DataFrame([np.sqrt(calculate_frame.iloc[:, 0]), df_boy["Y染色体浓度"]]).T
# #
# # calculate_frame = calculate_frame_temp
#
#
# print(calculate_frame)
#
# pearson_corr = calculate_frame.corr(method='pearson')
# print("皮尔逊相关系数矩阵：\n", pearson_corr)
#
#
# spearman_corr = calculate_frame.corr(method='spearman')
# print("\n斯皮尔曼相关系数矩阵：\n", spearman_corr)
#
#
#
#
#
#
#
#
#
#
#
#
#
#
