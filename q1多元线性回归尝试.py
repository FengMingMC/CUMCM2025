import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
import statsmodels.api as sm

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


def pregnancyTimes(row):
    # 定义映射规则
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




# np.random.seed(0)
# n_samples = 100
# X1 = np.random.rand(n_samples) * 10
# X2 = np.random.rand(n_samples) * 5
# X3 = np.random.rand(n_samples) * 2
# X4 = np.random.rand(n_samples) * 8
# # 假设 Y 线性依赖于 X1, X3, X4, 并加上一些噪声
# y = 2 + 1.5 * X1 - 0.8 * X3 + 0.5 * X4 + np.random.randn(n_samples) * 2
#
# # 将数据放入 DataFrame
data = pd.DataFrame({
    'Y染色体浓度': df_boy['Y染色体浓度'],
    '孕周': df_boy['孕周'],
    'GC含量': df_boy['GC含量'],
    '孕妇BMI': df_boy['孕妇BMI'],
    '生产次数': df_boy['生产次数'],
    '怀孕次数': df_boy['PregnancyTimes']
})
# --- 实现一个简化的向前选择算法 ---
# 目标：找出最显著的变量加入模型

# 包含常数项的自变量集合
# X_all = data[['X1', 'X2', 'X3', 'X4']].copy()
# X_all = sm.add_constant(X_all)
X_cols = ['孕周', 'GC含量', '孕妇BMI', '生产次数', '怀孕次数']
X_all = sm.add_constant(data[X_cols])
y_data = data['Y染色体浓度']

# 记录当前模型中的变量 (最初只有常数项)
current_vars = ['const']
remaining_vars = X_cols.copy()
model_results = None

print("--- 开始向前选择 ---")

while remaining_vars:
    best_p_value = float('inf')
    best_var_to_add = None

    # 尝试将每个剩余变量添加到当前模型
    for var in remaining_vars:
        # 构建包含新变量的模型（加上当前已有的变量）
        vars_to_test = current_vars + [var]
        X_subset = X_all[vars_to_test]

        # 拟合模型
        model = sm.OLS(y_data, X_subset)
        try:
            results = model.fit()
            # 获取新加入变量的 p 值 (statsmodels 1.1+ 可以直接访问 pvalues 属性)
            # 注意：这里需要根据具体statsmodels版本找到对应新变量的p值
            # 通常是 results.pvalues[-1] (如果新变量是最后一个添加的)
            # 或者更可靠地：results.pvalues[var]
            p_value_var = results.pvalues[var]

            # 设定一个显著性阈值（例如 p < 0.05）
            if p_value_var < 0.05 and p_value_var < best_p_value:
                best_p_value = p_value_var
                best_var_to_add = var
        except Exception as e:
            print(f"Fitting model with {var} failed: {e}")
            continue # 尝试下一个变量

    # 如果找到了一个显著改进的模型
    if best_var_to_add:
        current_vars.append(best_var_to_add)
        remaining_vars.remove(best_var_to_add)
        print(f"  --> Add '{best_var_to_add}' (p={best_p_value:.4f}). Current vars: {current_vars}")
    else:
        # 没有找到可以显著改进模型的变量，停止
        print("  No more variables significantly improve the model. Stopping.")
        break

# 最终模型
if len(current_vars) > 1: # 至少有常数项和1个自变量
    final_X = X_all[current_vars]
    final_model = sm.OLS(y_data, final_X)
    model_results = final_model.fit()
    print("\n--- 最终模型 ---")
    print(model_results.summary())
else:
    print("\n没有显著的自变量被选入模型。")

# calculate_frame = pd.DataFrame({
#     # "孕周":df_boy["孕周"],
#     "孕妇BMI": df_boy["孕妇BMI"],
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
# calculate_frame_temp = pd.DataFrame([np.sqrt(calculate_frame.iloc[:, 0]), df_boy["Y染色体浓度"]]).T
#
# calculate_frame = calculate_frame_temp
#
#
# print(calculate_frame)
#
# pearson_corr = calculate_frame.corr(method='pearson')
# print("皮尔逊相关系数矩阵：\n", pearson_corr)
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




