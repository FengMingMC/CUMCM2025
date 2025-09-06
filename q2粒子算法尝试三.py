import pandas as pd
import numpy as np
import pyswarms as ps
from sympy import symbols, solve, Eq
from q1双对数模型建立 import calculate_frame




bmiList = calculate_frame['BMI']

global_bmi_min = bmiList.min()
global_bmi_max = bmiList.max()

# a, b, c = 0.9004, -0.9195, 0.2299
a = 6.549496493147695e-05
b = -7.0549519278202565e-06
c = 2.3113341618794756e-05
d = 1.719914520891346e-05
e = -0.003313947522010834
f = -0.0011519253356130858
g = -0.0023610412520907254
h = 0.07583522622806302
i = 0.09279266520577661
j = -1.397845882473051
def predictTestTime(BMI):
    # return (0.04 / (0.9004 * (BMI ** -0.9195))) ** (1 / 0.2299)
    time = symbols('t')
    equation = a*time**3 + b*time**2*BMI + c*time*BMI**2 + d*time**2 + e*time*BMI + f*time + g*BMI**2 + h*time + i*BMI + j - 0.04
    return (solve(equation, time))


def objective_function(split_points, bmi_data, time_mapping):
    # split_points 是 PSO 粒子当前的位置，例如 [p1, p2, p3, p4]
    # bmi_data 是你的 bmiList 数据（BMI值）
    # time_mapping 是一个字典或函数，用于将BMI值映射到时间

    # 1. 确定5组的BMI边界
    min_bmi = bmi_data.min()
    max_bmi = bmi_data.max()

    # 确保分割点有序，并将固定边界加入
    # 此处需要根据PSO输出的split_points来生成最终的组边界
    # 一个简单的方法是 PSO直接输出4个值，然后在函数内部排序并加入min/max
    sorted_splits = sorted(split_points)
    group_boundaries = [min_bmi] + sorted_splits + [max_bmi]

    total_weighted_time = 0
    total_samples = len(bmi_data)

    # 2. 遍历每一组，进行计算
    for i in range(len(group_boundaries) - 1):
        lower_bound = group_boundaries[i]
        upper_bound = group_boundaries[i+1]

        # 找到属于当前组的样本
        group_samples = bmi_data[(bmi_data >= lower_bound) & (bmi_data < upper_bound)]
        num_samples_in_group = len(group_samples)

        # 约束检查 1: 每组不少于20人
        if num_samples_in_group < 20:
            # 如果违反约束，返回一个很大的惩罚值，让PSO避开这个区域
            return float('inf')

        # 3. 计算该组的BMI上限加下限的和
        center = (lower_bound + upper_bound) / 2

        # 4. 找到对应的时间值
        # 假设你有一个函数或字典 time_mapping 来根据sum_of_bounds查找时间
        # 例如: group_time = get_time_from_bmi_sum(sum_of_bounds, time_mapping)
        # 这里的 get_time_from_bmi_sum 需要你根据实际情况实现，可能是一个插值或查找过程
        # 假设我们有一个函数 time_mapping_func(bmi_sum)
        group_time = time_mapping(center)

        # 5. 计算加权时间
        weight = num_samples_in_group / total_samples
        weighted_group_time = group_time * weight

        # 6. 累加到总时间
        total_weighted_time += weighted_group_time

    # 7. 返回需要最小化的总时间
    return total_weighted_time



# print(predictTestTime(global_bmi_min))


# 假设你的bmi_data 是一个numpy数组或pandas Series
# 假设你的 time_mapping_func 已经定义

# 1. 确定搜索空间的边界
min_bmi = bmiList.min()
max_bmi = bmiList.max()

# PSO 需要优化4个变量 (p1, p2, p3, p4)
k = 5
dimensions = k - 1

# 定义 PSO 的搜索空间边界
# 这里我们允许 p1, p2, p3, p4 在 min_bmi 和 max_bmi 之间任意取值，
# 并在目标函数中处理它们的大小关系和分组约束。
# 如果需要更严格的边界，可以调整lb和ub
lower_bounds = np.array([min_bmi] * dimensions)
upper_bounds = np.array([max_bmi] * dimensions)

# 2. 配置 PSO 优化器
options = {
    'c1': 0.5,  # 认知系数
    'c2': 0.5,  # 社会系数
    'w': 0.9,   # 惯性权重
    'k': 2,     # 邻域大小（对于全局最优 PSO，k=1）
    'p': 2,     # 邻域拓扑结构（对于全局最优 PSO，p=1）
    'min_velocity': -2, # 速度下限
    'max_velocity': 2,  # 速度上限
    'pop': 50,          # 粒子数量
    'max_iter': 100,    # 最大迭代次数
    'print_progress': True,
    'bounds': (lower_bounds, upper_bounds) # 设置边界
}

# 3. 实例化 PSO 优化器 (这里使用全局最优 PSO)
# optimizer = ps.single.GlobalBestPSO(n_particles=options['pop'],
#                                     dimensions=dimensions,
#                                     objective_func=objective_function,
#                                     config=options,
#                                     verbose=True,
#                                     # 传递额外的参数给目标函数
#                                     bmi_data=bmiList,
#                                     time_mapping_func=predictTestTime)
optimizer = ps.single.GlobalBestPSO(n_particles=options['pop'],
                                    dimensions=dimensions,
                                    options=options)

# 4. 运行优化
# 传递 objective_function 所需的额外参数
cost, pos = optimizer.optimize(objective_function, iters=100, bmi_data = bmiList, time_mapping = predictTestTime)
# 'pos' 将包含最优的 [p1, p2, p3, p4]
# 'cost' 将是最小的总加权时间

# 1. 重新构建最终的分组边界
min_bmi = bmiList.min()
max_bmi = bmiList.max()
sorted_optimal_splits = sorted(bmiList)
final_group_boundaries = [min_bmi] + sorted_optimal_splits + [max_bmi]

# 2. 计算最终的分组情况和总加权时间
final_total_weighted_time = 0
print("最优分组方案：")
for i in range(len(final_group_boundaries) - 1):
    lower = final_group_boundaries[i]
    upper = final_group_boundaries[i+1]
    group_samples = bmiList[(bmiList >= lower) & (bmiList < upper)]
    num_samples = len(group_samples)
    bmi_sum = lower + upper
    group_time = predictTestTime(bmi_sum)
    weight = num_samples / bmiList.shape[0]
    weighted_time = group_time * weight
    final_total_weighted_time += weighted_time

    print(f"  组 {i+1}: BMI [{lower:.2f}, {upper:.2f}) - {num_samples} 人 (占比 {weight:.2%}) - 加权时间: {weighted_time:.2f}")

print(f"\n最小总加权时间: {final_total_weighted_time:.2f}")

# 3. 再次验证约束
if any([len(bmiList[(bmiList >= final_group_boundaries[i]) & (bmiList < final_group_boundaries[i+1])]) < 20 for i in range(len(final_group_boundaries)-1)]):
    print("警告：最优方案存在分组人数少于20人的情况，请检查目标函数中的惩罚项设置！")