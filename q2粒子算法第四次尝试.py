import pandas as pd
import numpy as np
import pyswarms as ps
import matplotlib.pyplot as plt
from pyswarms.utils.plotters import plot_cost_history

from q1数据清洗 import calculate_frame


bmiList = pd.DataFrame({"BMI":calculate_frame["孕妇BMI"]})
print(f"总样本量: {len(bmiList)}")
print(f"BMI 最小值: {bmiList["BMI"].min():.2f}")
print(f"BMI 最大值: {bmiList["BMI"].max():.2f}")
print(bmiList.head())


a, b, c = 0.9004, -0.9195, 0.2299
def time_func_of_bmi(bmi):
    # 假设BMI在22.5时时间最短（例如10个单位），其他情况时间更长
    return (0.04 / (a * (bmi ** b))) ** (1 / c)


# 将数据转换为numpy数组以提高计算速度
bmi_values = bmiList['BMI'].values
min_bmi = bmi_values.min()
max_bmi = bmi_values.max()
total_samples = len(bmi_values)

k = 5

def calculate_total_time(boundaries):
    """
    这是一个辅助函数，计算给定一组边界的总时间。
    为 PSO 目标函数调用。
    """
    # 强制排序，满足 b1 < b2 < b3 < b4 的约束
    sorted_boundaries = np.sort(boundaries)

    # 构造完整的5组边界
    full_boundaries = np.concatenate(([min_bmi], sorted_boundaries, [max_bmi]))

    total_weighted_time = 0

    for i in range(k):  # 遍历5个组
        lower_bound = full_boundaries[i]
        upper_bound = full_boundaries[i + 1]

        # 筛选出当前组的样本
        if i < k - 1 :
            mask = (bmi_values >= lower_bound) & (bmi_values < upper_bound)
        else:  # 最后一组包含上限
            mask = (bmi_values >= lower_bound) & (bmi_values <= upper_bound)

        group_samples = bmi_values[mask]
        sample_count = len(group_samples)

        # --- 约束检查 ---
        # 如果任何一组的样本量小于20，则返回一个巨大的惩罚值
        if sample_count < 20:
            return float('inf')

        # if sample_count > 300:
        #     return float('inf')

            # 计算该组的加权时间
        midpoint = (lower_bound + upper_bound) / 2
        time_at_midpoint = time_func_of_bmi(midpoint)
        weight = sample_count / total_samples
        weighted_time = time_at_midpoint * weight

        total_weighted_time += weighted_time

    return total_weighted_time


def pso_objective_function(particles):
    """
    pyswarms需要的目标函数。
    它接收一个粒子矩阵，为每个粒子（每一行）计算成本。
    """
    # particles 的形状是 (n_particles, n_dimensions)
    n_particles = particles.shape[0]
    costs = [calculate_total_time(p) for p in particles]
    return np.array(costs)


# --- PSO 配置 ---
options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}

# 边界值的搜索范围
# 每个边界值都可以在BMI的最小和最大值之间取值
lower_bound_search = [min_bmi] * (k-1)
upper_bound_search = [max_bmi] * (k-1)
bounds = (np.array(lower_bound_search), np.array(upper_bound_search))

# 实例化优化器
optimizer = ps.single.GlobalBestPSO(n_particles=50,
                                    dimensions=4,
                                    options=options,
                                    bounds=bounds)

# --- 执行优化 ---
best_cost, best_pos = optimizer.optimize(pso_objective_function, iters=100)

print(f"\n--- PSO 优化完成 ---")
print(f"找到的最小总时间 (Best Cost): {best_cost:.4f}")
print(f"对应的最佳切分点 (Best Position): {np.sort(best_pos)}")

# --- 结果分析与展示 ---
print("\n--- 最佳分组方案详情 ---")
sorted_best_boundaries = np.sort(best_pos)
final_boundaries = np.concatenate(([min_bmi], sorted_best_boundaries, [max_bmi]))

results_data = []
for i in range(k):
    lower = final_boundaries[i]
    upper = final_boundaries[i + 1]

    if i < k - 1:
        mask = (bmi_values >= lower) & (bmi_values < upper)
    else:
        mask = (bmi_values >= lower) & (bmi_values <= upper)

    count = np.sum(mask)
    midpoint_val = (lower + upper) / 2
    time_val = time_func_of_bmi(midpoint_val)
    weight_val = count / total_samples

    results_data.append({
        "组别": i + 1,
        "BMI范围": f"[{lower:.2f}, {upper:.2f})",
        "人数": count,
        "范围中点": f"{midpoint_val:.2f}",
        "对应时间": f"{time_val:.2f}",
        "权重": f"{weight_val:.2%}",
        "加权时间": f"{time_val * weight_val:.4f}"
    })

# 将最后一组的范围括号修正
results_data[-1]["BMI范围"] = f"[{final_boundaries[-2]:.2f}, {final_boundaries[-1]:.2f}]"

results_df = pd.DataFrame(results_data)
print(results_df.to_string())

# 绘制成本历史记录图，观察收敛情况
plot_cost_history(optimizer.cost_history)
plt.title("PSO Convergence Plot")
plt.xlabel("Iteration")
plt.ylabel("Total Weighted Time (Cost)")
plt.show()