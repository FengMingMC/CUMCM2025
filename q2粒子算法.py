import pandas as pd
import numpy as np
import pyswarms as ps

from q1双对数模型建立 import calculate_frame

bmiList = calculate_frame['BMI']

global_bmi_min = bmiList.min()
global_bmi_max = bmiList.max()

a, b, c = 0.9004, -0.9195, 0.2299


def predictTestTime(BMI):
    return (0.04 / (0.9004 * (BMI ** -0.9195))) ** (1 / 0.2299)


def objective_function(positions, bmi_data, time_mapping):
    # positions 是一个二维数组，形状为 (n_particles, n_dimensions)
    n_particles = positions.shape[0]
    costs = np.zeros(n_particles)

    for i in range(n_particles):
        # 获取第 i 个粒子的位置（分割点）
        split_points = positions[i]

        # 对分割点进行排序
        sorted_splits = np.sort(split_points)

        # 1. 确定5组的BMI边界
        min_bmi = bmi_data.min()
        max_bmi = bmi_data.max()

        # 将固定边界加入
        group_boundaries = [min_bmi] + sorted_splits.tolist() + [max_bmi]

        total_weighted_time = 0
        total_samples = len(bmi_data)
        valid_group = True

        # 2. 遍历每一组，进行计算
        for j in range(len(group_boundaries) - 1):
            lower_bound = group_boundaries[j]
            upper_bound = group_boundaries[j + 1]

            # 找到属于当前组的样本
            group_samples = bmi_data[(bmi_data >= lower_bound) & (bmi_data < upper_bound)]
            num_samples_in_group = len(group_samples)

            # 约束检查 1: 每组不少于20人
            if num_samples_in_group < 20 :
                # 如果违反约束，给这个粒子一个很大的惩罚值
                costs[i] = float('inf')
                valid_group = False
                break

            if num_samples_in_group > 500 :
                # 如果违反约束，给这个粒子一个很大的惩罚值
                costs[i] = float('inf')
                valid_group = False
                break

            # 3. 计算该组的中心点
            center = (lower_bound + upper_bound) / 2

            # 4. 找到对应的时间值
            group_time = time_mapping(center)

            # 5. 计算加权时间
            weight = num_samples_in_group / total_samples
            weighted_group_time = group_time * weight

            # 6. 累加到总时间
            total_weighted_time += weighted_group_time

        if valid_group:
            costs[i] = total_weighted_time

    return costs


# 其余代码保持不变...

# 假设你的bmi_data 是一个numpy数组或pandas Series
# 假设你的 time_mapping_func 已经定义

# 1. 确定搜索空间的边界
min_bmi = bmiList.min()
max_bmi = bmiList.max()

# PSO 需要优化4个变量 (p1, p2, p3, p4)
k = 5
dimensions = k - 1

# 定义 PSO 的搜索空间边界
lower_bounds = np.array([min_bmi] * dimensions)
upper_bounds = np.array([max_bmi] * dimensions)

# 2. 配置 PSO 优化器
options = {
    'c1': 0.5,  # 认知系数
    'c2': 0.5,  # 社会系数
    'w': 0.9,  # 惯性权重
}

# 3. 实例化 PSO 优化器
optimizer = ps.single.GlobalBestPSO(n_particles=50,
                                    dimensions=dimensions,
                                    options=options,
                                    bounds=(lower_bounds, upper_bounds))

# 4. 运行优化
cost, pos = optimizer.optimize(objective_function, iters=100, bmi_data=bmiList, time_mapping=predictTestTime)

# 5. 输出结果
min_bmi = bmiList.min()
max_bmi = bmiList.max()
sorted_optimal_splits = sorted(pos)
final_group_boundaries = [min_bmi] + sorted_optimal_splits + [max_bmi]

final_total_weighted_time = 0
print("最优分组方案：")
for i in range(len(final_group_boundaries) - 1):
    lower = final_group_boundaries[i]
    upper = final_group_boundaries[i + 1]
    group_samples = bmiList[(bmiList >= lower) & (bmiList < upper)]
    num_samples = len(group_samples)
    center = (lower + upper) / 2
    group_time = predictTestTime(center)
    weight = num_samples / len(bmiList)
    weighted_time = group_time * weight
    final_total_weighted_time += weighted_time

    print(
        f"  组 {i + 1}: BMI [{lower:.2f}, {upper:.2f}) - {num_samples} 人 (占比 {weight:.2%}) - 加权时间: {weighted_time:.2f}")

print(f"\n最小总加权时间: {final_total_weighted_time:.2f}")

# 3. 再次验证约束
if any([len(bmiList[(bmiList >= final_group_boundaries[i]) & (bmiList < final_group_boundaries[i + 1])]) < 20 for i in
        range(len(final_group_boundaries) - 1)]):
    print("警告：最优方案存在分组人数少于20人的情况，请检查目标函数中的惩罚项设置！")