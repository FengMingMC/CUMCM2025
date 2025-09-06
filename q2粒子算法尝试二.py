import pandas as pd
import numpy as np
import pyswarms as ps
from sympy import symbols, solve, Eq
from q1双对数模型建立 import calculate_frame

# 假设 calculate_frame 和 predictTime 已经定义
# from q1双对数模型建立 import calculate_frame
# global_bmi_min = calculate_frame['BMI'].min()
# global_bmi_max = calculate_frame['BMI'].max()
# a, b, c = 0.9004, -0.9195, 0.2299

# 示例数据和函数（请替换为你自己的）


global_bmi_min = calculate_frame['BMI'].min()
global_bmi_max = calculate_frame['BMI'].max()
a, b, c = 0.9004, -0.9195, 0.2299


# def get_time_for_bmi(target_bmi, dataframe):
#     closest_idx = (dataframe['BMI'] - target_bmi).abs().idxmin()
#     return dataframe.loc[closest_idx, '时间']


# a = 6.549496493147695e-05
# b = -7.0549519278202565e-06
# c = 2.3113341618794756e-05
d = 1.719914520891346e-05
e = -0.003313947522010834
f = -0.0011519253356130858
g = -0.0023610412520907254
h = 0.07583522622806302
i = 0.09279266520577661
j = -1.397845882473051

def get_time_for_bmi(BMI):
    # 确保 BMI 远离可能导致除零或无效计算的边界
    if BMI <= 0:
        return np.inf  # 或者一个非常大的惩罚值
    try:
        # time = symbols('t')
        # equation = e * time * BMI + f * time + g * BMI ** 2 + h * time + i * BMI + j - 0.04
        # result = solve(equation, time)
        # realresult = [x for x in result if isinstance(x, (int, float)) ]
        # positive_numbers = [x for x in realresult if x > 0 ]
        # return min(positive_numbers)
        return (0.04 / (a * (BMI ** b))) ** (1 / c)
    except (ValueError, OverflowError, ZeroDivisionError):
        return np.inf  # 同样返回一个非常大的惩罚值



def objective_function(boundaries_batch, dfInside: pd.DataFrame):
    """
    计算给定 BMI 边界划分的总加权时间。
    Args:
        boundaries_batch (np.ndarray): 形状为 (n_particles, dimensions) 的数组，
                                       包含所有粒子的边界值。
        dfInside (pd.DataFrame): 包含 'BMI' 和 '时间' 列的原始 DataFrame。
    Returns:
        np.ndarray: 包含每个粒子计算出的总加权时间的数组。
    """
    # 初始化一个数组来存储每个粒子的成本
    costs = np.zeros(boundaries_batch.shape[0])

    # 遍历每个粒子（每个粒子的边界值是一行）
    for i, particle_boundaries in enumerate(boundaries_batch):
        # 1. 确保边界值在有效范围内，并且有序 (针对当前粒子)
        # np.clip 处理单个粒子边界值，返回一个一维数组
        clipped_boundaries = np.clip(particle_boundaries, global_bmi_min, global_bmi_max)
        # sorted 接收一维数组，返回一个排序后的一维数组
        b1, b2, b3, b4 = sorted(clipped_boundaries)

        # 定义分组的上下限
        bins = [global_bmi_min, b1, b2, b3, b4, global_bmi_max]
        labels = ['group1', 'group2', 'group3', 'group4', 'group5']

        # 复制 DataFrame 以避免修改原始数据
        df_copy = dfInside.copy()
        # 使用 pd.cut 进行分组
        # duplicates='drop' 是一个好的处理方式，但需要注意分组数量可能少于5
        df_copy['group'] = pd.cut(df_copy['BMI'], bins=bins, labels=labels, include_lowest=True, duplicates='drop')

        current_particle_total_weighted_time = 0
        # 统计实际存在的分组数量
        actual_labels = df_copy['group'].dropna().unique()

        for group_label in labels:  # 遍历所有预期的标签
            group_df = df_copy[df_copy['group'] == group_label]

            if not group_df.empty:
                sample_size = len(group_df)
                bmi_median = group_df['BMI'].median()

                # 查找 BMI 中位数对应的时间
                time_for_median_bmi = get_time_for_bmi(bmi_median)

                # 如果 predictTime 返回了惩罚值，直接将此粒子成本设为高
                if np.isinf(time_for_median_bmi):
                    current_particle_total_weighted_time = np.inf
                    break  # 提前结束当前粒子计算，因为它已经是一个无效解

                current_particle_total_weighted_time += sample_size * time_for_median_bmi

        # 将当前粒子的总加权时间存储到 costs 数组中
        # 如果计算过程中出现 NaN 或 Inf，也将其设为无穷大
        if not np.isfinite(current_particle_total_weighted_time):
            costs[i] = np.inf
        else:
            costs[i] = current_particle_total_weighted_time

    return costs  # 返回一个 numpy 数组，包含所有粒子的成本


# --- 主程序部分 ---

df = calculate_frame
k = 4  # 需要 4 个边界来划分 5 组
dimensions = k

# 2. 定义搜索空间边界
# 每个边界值都应该在全局 BMI 范围内
# my_bounds 是一个元组，包含两个数组：每个维度的下界数组，每个维度的上界数组
lower_bounds = np.array([global_bmi_min] * dimensions)
upper_bounds = np.array([global_bmi_max] * dimensions)
my_bounds = (lower_bounds, upper_bounds)

# 3. 初始化 PSO 优化器
optimizer = ps.single.GlobalBestPSO(
    n_particles=50,  # 种群大小，可以调整
    dimensions=dimensions,  # 决策变量的数量
    options={'c1': 0.5, 'c2': 0.3, 'w': 0.9},  # PSO 参数，可调整
    bounds=my_bounds  # 搜索空间的边界
)

# 4. 执行优化
# 'df' 需要作为 kwarg 传递给 objective_function
# pyswarms 会自动处理批次调用

cost, pos = optimizer.optimize(objective_function, iters=100, verbose=True, dfInside=df)

print(f"最小总加权时间: {cost}")
# pos 是最优粒子找到的边界值，形状为 (dimensions,)
print(f"最优 BMI 边界值: {pos}")

# 5. 解析最优边界值
# 确保最优边界值在全局范围内，并且排序
optimal_boundaries = sorted(np.clip(pos, global_bmi_min, global_bmi_max))
print(f"最优分组边界 (BMI): {optimal_boundaries}")

# 使用最优边界进行最终分组和分析
bins = [global_bmi_min] + list(optimal_boundaries) + [global_bmi_max]
# 确保标签数量与实际分组数量匹配
labels = [f'Group_{i + 1}' for i in range(len(bins) - 1)]
df['final_group'] = pd.cut(df['BMI'], bins=bins, labels=labels, include_lowest=True, duplicates='drop')

print("\n最终分组结果的统计信息:")
# 使用 nunique() 来计算实际分组数量，以防 duplicates='drop' 删除了某些分组
actual_groups_count = df['final_group'].nunique()
if actual_groups_count < len(bins) - 1:
    print(f"注意：由于 BMI 边界值重复，实际分组数量为 {actual_groups_count}，少于预期的 {len(bins) - 1}。")

print(df.groupby('final_group').agg(
    sample_size=('BMI', 'size'),
    bmi_median=('BMI', 'median'),
    # 查找中位数对应的时间（再次调用）
    # 注意：get_time_for_bmi 查找的是最接近的，predictTime 是模型预测
    # 你需要明确你的需求是哪个
    time_median_bmi=pd.NamedAgg(column='BMI', aggfunc=lambda x: get_time_for_bmi(x.median()) if not pd.isna(
        x.median()) else np.nan)
))

# 验证计算出的最小总时间
final_total_weighted_time = 0
final_stats = df.groupby('final_group').agg(
    sample_size=('BMI', 'size'),
    bmi_median=('BMI', 'median')
)
for index, row in final_stats.iterrows():
    if not pd.isna(row['bmi_median']):
        # 使用 predictTime 来计算时间，与目标函数一致
        time_val = get_time_for_bmi(row['bmi_median'])
        if np.isfinite(time_val):
            final_total_weighted_time += row['sample_size'] * time_val
        else:
            # 如果预测时间是无效的，这表明这个分组方案可能不是最优的
            # 或者需要处理 predictTime 的返回值
            print(f"警告：在最终验证时，{index} 的 BMI 中位数 {row['bmi_median']} 预测时间无效。")

print(f"\n通过最优边界计算的最终总加权时间 (使用 predictTime): {final_total_weighted_time}")