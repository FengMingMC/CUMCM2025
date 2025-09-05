import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
import statsmodels.api as sm
from scipy.stats import pearsonr, spearmanr # 导入 scipy.stats

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
        # 确保这里的正则表达式也能处理纯数字的孕周，如果你的数据格式可能包含这种情况
        try:
            return int(re.search(r'(\d+)w', pregnancy_week_str).group(1))
        except AttributeError:
            # 如果连 'w' 都找不到，可能需要更复杂的错误处理或数据清洗
            return np.nan # 返回 NaN 表示无效数据

df_boy['孕周'] = df_boy.apply(calculate_week, axis=1)

# 过滤掉孕周为 NaN 的行，因为它们无法参与后续计算
df_boy.dropna(subset=['孕周'], inplace=True)


calculate_frame = pd.DataFrame({
    "孕周": df_boy["孕周"],
    "孕妇BMI": df_boy["孕妇BMI"],
    "Y染色体浓度": df_boy["Y染色体浓度"]
})

# print(calculate_frame)

# 假设您之前是打算对 "孕周" 和 "Y染色体浓度" 进行相关性分析，
# 但后面的代码修改了 calculate_frame，所以我们重新构建一个用于相关性分析的 DataFrame
# 如果您需要对 "孕妇BMI" 和 "Y染色体浓度" 进行分析，请相应调整

# 重新构建用于相关性分析的 DataFrame，这里以 "孕周" 和 "Y染色体浓度" 为例
# 如果您想分析的是 sqrt(孕周) 和 Y染色体浓度，请使用您后面的代码
# calculate_frame_for_correlation = pd.DataFrame({
#     "孕周": df_boy["孕周"],
#     "Y染色体浓度": df_boy["Y染色体浓度"]
# })

# 根据您提供的最后一段代码，您是分析 sqrt(孕周) 和 Y染色体浓度
calculate_frame_for_correlation = pd.DataFrame({
    "sqrt_孕周": df_boy["孕周"], # 注意：这里假设 df_boy["孕周"] 总是大于等于0，否则 np.sqrt 会报错
    "Y染色体浓度": df_boy["Y染色体浓度"]
})


print("用于相关性分析的数据框：")
print(calculate_frame_for_correlation)

# --- 皮尔逊相关系数及 P 值 ---
# pearson_corr_matrix = calculate_frame_for_correlation.corr(method='pearson') # 旧方法，只输出系数
# print("皮尔逊相关系数矩阵：\n", pearson_corr_matrix)

# 使用 scipy.stats.pearsonr 获取相关系数和 P 值
pearson_coeff, pearson_p_value = pearsonr(calculate_frame_for_correlation.iloc[:, 0], calculate_frame_for_correlation.iloc[:, 1])
print("\n皮尔逊相关系数:")
print(f"  系数: {pearson_coeff:.4f}")
print(f"  P 值: {pearson_p_value:.4f}")

# --- 斯皮尔曼相关系数及 P 值 ---
# spearman_corr_matrix = calculate_frame_for_correlation.corr(method='spearman') # 旧方法，只输出系数
# print("\n斯皮尔曼相关系数矩阵：\n",spearman_corr_matrix)

# 使用 scipy.stats.spearmanr 获取相关系数和 P 值
spearman_coeff, spearman_p_value = spearmanr(calculate_frame_for_correlation.iloc[:, 0], calculate_frame_for_correlation.iloc[:, 1])
print("\n斯皮尔曼相关系数:")
print(f"  系数: {spearman_coeff:.4f}")
print(f"  P 值: {spearman_p_value:.4f}")

# 如果您还需要对其他列进行相关性分析，请重复上述 Pearsonr 和 Spearmanr 的调用。
# 例如，如果想分析 "孕妇BMI" 和 "Y染色体浓度" 的相关性：
# print("\n--- 孕妇BMI vs Y染色体浓度 ---")
# pearson_bmi_y_coeff, pearson_bmi_y_p_value = pearsonr(df_boy["孕妇BMI"], df_boy["Y染色体浓度"])
#print(f"皮尔逊相关系数: {pearson_bmi_y_coeff:.4f}, P 值: {pearson_bmi_y_p_value:.4f}")
#spearman_bmi_y_coeff, spearman_bmi_y_p_value = spearmanr(df_boy["孕妇BMI"], df_boy["Y染色体浓度"])
#print(f"斯皮尔曼相关系数: {spearman_bmi_y_coeff:.4f}, P 值: {spearman_p_value:.4f}")