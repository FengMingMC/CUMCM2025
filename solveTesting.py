# 导入所需的库
import numpy as np
import pandas as pd
import pyswarms as ps

# --- 1. 数据准备 ---
# 请将 'data.xlsx - 男胎检测数据.csv' 文件中的男胎数据加载到 DataFrame 中
# 确保文件路径正确
try:
    male_data = pd.read_excel('data/附件.xlsx', sheet_name = '男胎检测数据' )
except FileNotFoundError:
    print("错误：文件未找到。请确保 'data.xlsx - 男胎检测数据.csv' 文件存在。")
    exit()

# 将数据按BMI从小到大排序
sorted_male_data = male_data.sort_values(by='孕妇BMI')
sorted_male_data.reset_index(drop=True, inplace=True)
N = len(sorted_male_data)

# --- 2. 定义核心函数 ---
# 你的第一问回归模型参数 (请替换为实际值)
a, b, c = 0.01, 1.5, 2.0


def calculate_t(avg_bmi):
    """
    根据平均BMI，使用回归模型计算达到4%浓度所需的孕周t。
    """
    if avg_bmi <= 0:
        return np.inf

    try:
        t = (0.04 / (a * (avg_bmi ** b))) ** (1 / c)
        return t
    except (ZeroDivisionError, ValueError):
        return np.inf


def fitness_function_pyswarms(positions):
    """
    为pyswarms设计的适应度函数，计算总潜在风险。
    输入: positions (np.array), 形状为 (n_particles, dimensions)
    输出: 风险值数组 (np.array), 形状为 (n_particles,)
    """
    n_particles = positions.shape[0]
    all_risks = np.zeros(n_particles)

    for i in range(n_particles):
        # 将连续位置四舍五入并转换为唯一的整数分割点
        split_points = np.sort(np.unique(np.round(positions[i]).astype(int)))

        # 将分割点限制在有效范围内 [1, N-1]
        split_points = np.clip(split_points, 1, N - 1)

        # 确定每个分组的边界
        group_boundaries = [0] + list(split_points) + [N]
        total_risk = 0
        is_valid_solution = True

        # 遍历每个分组，计算其风险
        for j in range(len(group_boundaries) - 1):
            start_index = group_boundaries[j]
            end_index = group_boundaries[j + 1]
            group_size = end_index - start_index

            # 约束1: 样本数量至少为20
            if group_size < 20:
                is_valid_solution = False
                break

            # 计算该分组的平均BMI
            group_bmi_data = sorted_male_data.iloc[start_index:end_index]['孕妇BMI']
            avg_bmi = np.mean(group_bmi_data)

            # 计算该分组的t值
            t_value = calculate_t(avg_bmi)

            # 约束2: t值不能超过28周
            if t_value > 28:
                is_valid_solution = False
                break

            # 计算该分组的风险成本
            group_risk = (group_size / N) * t_value
            total_risk += group_risk

        # 如果不满足约束，给予巨大的惩罚
        if not is_valid_solution:
            all_risks[i] = np.inf
        else:
            all_risks[i] = total_risk

    return all_risks


# --- 3. 运行PSO算法寻找最优解 ---
results = {}

# 遍历可能的分组数量 k (例如从2到10)
for k_val in range(2, 11):
    print(f"\n--- 运行针对 k = {k_val} 的PSO ---")

    dimensions = k_val - 1  # 粒子的维度等于分割点数量

    # 定义搜索空间的边界，确保分割点在有效索引范围内
    bounds = (np.ones(dimensions), np.full(dimensions, N - 1))

    # 初始化优化器，设置参数
    options = {'c1': 2.0, 'c2': 2.0, 'w': 0.8}
    optimizer = ps.single.GlobalBestPSO(n_particles=30, dimensions=dimensions, options=options, bounds=bounds)

    # 运行优化过程
    cost, pos = optimizer.optimize(fitness_function_pyswarms, iters=200)

    # 存储结果
    results[k_val] = {
        'risk': cost,
        'split_points': pos
    }

# --- 4. 找到并展示全局最优解 ---
best_k = min(results, key=lambda k: results[k]['risk'])
final_result = results[best_k]
final_split_points = np.sort(np.unique(np.round(final_result['split_points']).astype(int)))

print("\n--- 最终最优解 ---")
print(f"最优分组数量 (k): {best_k}")
print(f"最小潜在风险: {final_result['risk']:.4f}")
print(f"最优分割点 (样本索引): {final_split_points}")

# 你可以根据最终分割点来进一步分析每个分组的BMI区间和t值
# 例如:
print("\n--- 最优分组方案详情 ---")
group_boundaries = [0] + list(final_split_points) + [N]
for i in range(len(group_boundaries) - 1):
    start_index = group_boundaries[i]
    end_index = group_boundaries[i + 1]
    group_data = sorted_male_data.iloc[start_index:end_index]

    min_bmi = group_data['孕妇BMI'].min()
    max_bmi = group_data['孕妇BMI'].max()
    avg_bmi = group_data['孕妇BMI'].mean()

    t_value = calculate_t(avg_bmi)

    print(
        f"第 {i + 1} 组: BMI 区间 [{min_bmi:.2f}, {max_bmi:.2f}]，样本数 {len(group_data)}，最佳NIPT时点 {t_value:.2f} 周")