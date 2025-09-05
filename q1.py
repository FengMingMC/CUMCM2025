import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
from scipy.stats import pearsonr, spearmanr
import statsmodels.api as sm
import seaborn as sns

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
    # "年龄": df_boy["年龄"],
    # "身高": df_boy["身高"],
    # "体重": df_boy["体重"],
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
# print(calculate_frame)
#
# # 定义因变量 (y)
# y = df_boy["Y染色体浓度"]
#
# # 定义自变量 (X)
# # 选择 'age' 和 'income' 作为自变量
# X = df_boy[['孕周', '孕妇BMI']]
#
# # 添加常数项 (截距)
# # sm.add_constant() 会在自变量 X 的左侧添加一列全为 1 的列
# X = sm.add_constant(X)
#
# # 使用 Ordinary Least Squares (OLS) 方法拟合模型
# # OLS 是多元线性回归中最常用的方法
# model = sm.OLS(y, X).fit()
#
# print(model.summary())
print(calculate_frame)

# calculate_frame_temp = pd.DataFrame([np.sqrt(calculate_frame.iloc[:, 0]), df_boy["Y染色体浓度"]]).T
#
# calculate_frame = calculate_frame_temp


# print(calculate_frame)
# np.set_printoptions(threshold=np.inf)
# np.set_printoptions(linewidth=np.inf)



pearson_corr_matrix = calculate_frame.corr(method='pearson')
print("皮尔逊相关系数矩阵：\n", pearson_corr_matrix)

# 2. 计算斯皮尔曼相关系数矩阵
spearman_corr_matrix = calculate_frame.corr(method='spearman')
print("\n斯皮尔曼相关系数矩阵：\n", spearman_corr_matrix)

# 3. 计算皮尔森相关系数的显著性 (p-value)
# 注意：pearsonr 函数一次只能计算两个变量之间的相关系数和 p-value
# 因此，我们需要遍历相关系数矩阵来获取所有配对的 p-value

# 创建一个空的 DataFrame 来存储 p-value
pearson_pvalue_matrix = pd.DataFrame(index=calculate_frame.columns, columns=calculate_frame.columns)

# 遍历 DataFrame 的列
for col1 in calculate_frame.columns:
    for col2 in calculate_frame.columns:
        if col1 == col2:
            pearson_pvalue_matrix.loc[col1, col2] = 1.0 # 自身相关系数的 p-value 为 1
        elif col1 < col2: # 避免重复计算
            # 使用 pearsonr 函数计算相关系数和 p-value
            corr, p_value = pearsonr(calculate_frame[col1], calculate_frame[col2])
            pearson_pvalue_matrix.loc[col1, col2] = p_value
            pearson_pvalue_matrix.loc[col2, col1] = p_value # 对称填入

print("\n皮尔逊相关系数的显著性 (p-value) 矩阵：\n", pearson_pvalue_matrix)

spearman_pvalue_matrix = pd.DataFrame(index=calculate_frame.columns, columns=calculate_frame.columns)

# 遍历 DataFrame 的列
for col1 in calculate_frame.columns:
    for col2 in calculate_frame.columns:
        if col1 == col2:
            spearman_pvalue_matrix.loc[col1, col2] = 1.0 # 自身相关系数的 p-value 为 1
        elif col1 < col2: # 避免重复计算
            # 使用 spearmanr 函数计算相关系数和 p-value
            corr, p_value = spearmanr(calculate_frame[col1], calculate_frame[col2])
            spearman_pvalue_matrix.loc[col1, col2] = p_value
            spearman_pvalue_matrix.loc[col2, col1] = p_value # 对称填入

print("\n斯皮尔曼相关系数的显著性 (p-value) 矩阵：\n", spearman_pvalue_matrix)



# sns.set_theme(style="white")

fig, axes = plt.subplots(1, 2, figsize=(18, 8))

# 绘制皮尔逊相关系数热力图
sns.heatmap(pearson_corr_matrix, annot=True, cmap='coolwarm', fmt=".2f",
            linewidths=.5, ax=axes[0])
axes[0].set_title('Pearson Correlation Heatmap')

# 绘制斯皮尔曼相关系数热力图
sns.heatmap(spearman_corr_matrix, annot=True, cmap='coolwarm', fmt=".2f",
            linewidths=.5, ax=axes[1])
axes[1].set_title('Spearman Correlation Heatmap')

plt.tight_layout()
plt.show()

# 如果您想将p值也标注在热力图上
fig, ax = plt.subplots(figsize=(10, 8))
# 创建一个注释矩阵，将显著性星号添加到相关系数值上
annotations = pearson_corr_matrix.copy()
for i in annotations.index:
    for j in annotations.columns:
        corr_val = annotations.loc[i, j]
        p_val = pearson_pvalue_matrix.loc[i, j]
        if p_val < 0.001:
            annotations.loc[i, j] = f'{corr_val:.2f}***'
        elif p_val < 0.01:
            annotations.loc[i, j] = f'{corr_val:.2f}**'
        elif p_val < 0.05:
            annotations.loc[i, j] = f'{corr_val:.2f}*'
        else:
            annotations.loc[i, j] = f'{corr_val:.2f}'

sns.heatmap(pearson_corr_matrix, annot=annotations, cmap='coolwarm', fmt='s', ax=ax, linewidths=.5)
ax.set_title('Pearson Correlation with Significance Stars\n(***: p<0.001, **: p<0.01, *: p<0.05)')
plt.show()

print("hello world")


calculate_frame.to_csv('data/final_data.csv', index=False)








