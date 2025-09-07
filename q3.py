import statistics

from scipy.stats import norm

from q1数据清洗 import *
import matplotlib.pyplot as plt
import pandas as pd

data = {
    'Week': calculate_frame["孕周"],
    'BMI': calculate_frame["孕妇BMI"],
    'Age': calculate_frame["年龄"],
    'Height': calculate_frame["身高"],
    'Weight': calculate_frame["体重"],
    "YContent": calculate_frame["Y染色体浓度"]}
data = pd.DataFrame(data)
def funcC(x_data,a,b,c,d,e,f,g):
    x,y,z,w,v = x_data
    return a + b*x + c*y + d * np.exp(z) + e*np.log10(w+1) + f * v + g * v**2

from scipy.optimize import curve_fit

initial_guess = [0.1,
                 0.1,
                 0.1,
                 0.1,
                 0.1,
                 0.1,
                 0.1]

x_data_packed = (np.array(data["Height"]), np.array(data["Weight"]), np.array(data["Age"]),  np.array(data["Week"]), np.array(data["BMI"]))
y_data = np.array(data["YContent"])

y_predicted = None
ss_res = None
ss_tot = None

try:
    # 进行拟合
    params, covariance = curve_fit(funcC, x_data_packed, y_data, p0=initial_guess)

    a_fit, b_fit, c_fit, d_fit, e_fit, f_fit, g_fit = params

    print("\n\n拟合得到的参数：")
    print(f"a = {a_fit}")
    print(f"b = {b_fit}")
    print(f"c = {c_fit}")
    print(f"d = {d_fit}")
    print(f"e = {e_fit}")
    print(f"f = {f_fit}")
    print(f"g = {g_fit}")

    y_predicted = funcC(x_data_packed, params[0], params[1], params[2], params[3], params[4], params[5], params[6])
    ss_res = np.sum((y_data - y_predicted) ** 2)
    ss_tot = np.sum((y_data - np.mean(y_data))**2) # 总平方和
    r_squared = 1 - (ss_res / ss_tot)
    print(f"R^2 = {r_squared}")

except RuntimeError as e:
    print(f"拟合失败：{e}")
    print("请检查您的数据和初始猜测参数。")

sample_stdev = statistics.stdev((y_data - y_predicted))
#
y_res = sample_stdev * (np.sum(y_data)/len(y_data)) ** 2

y_z = (0.04 - (np.sum(y_data)/len(y_data))) / np.sqrt(y_res)

p_value = norm.cdf(y_z)
print(f"对于 Z 值 {y_z}，P 值为: {p_value:.4f}")

print(y_res)


import pandas as pd
import numpy as np
import pyswarms as ps
import matplotlib.pyplot as plt
from pyswarms.utils.plotters import plot_cost_history

from q1数据清洗 import calculate_frame


# data = pd.DataFrame({"BMI":calculate_frame["孕妇BMI"]})
print(f"总样本量: {len(data)}")
print(f"BMI 最小值: {data["BMI"].min():.2f}")
print(f"BMI 最大值: {data["BMI"].max():.2f}")
# print(data.head())


# a, b, c = 0.9004, -0.9195, 0.2299
a = -0.6134707456120154
b = 0.0029772071399584684
c = -0.003370604781279176
d = -5.239532450624268e-21
e = 0.025739143057976022
f = 0.018424723231905298
g = -0.00016823880696470344
def time_func_of_bmi(bmi,height,weight,age,):
    # 假设BMI在22.5时时间最短（例如10个单位），其他情况时间更长
    return 10**((0.04-a-b*height-c*weight-d*np.exp(age)-f*bmi-g*bmi**2)/e) - 1

#
# 将数据转换为numpy数组以提高计算速度
bmi_values =    data['BMI'].values
# height_values = data['Height'].values
# weight_values = data['Weight'].values
# age_values =    data['Age'].values


min_bmi = data['BMI'].min()
max_bmi = data['BMI'].max()
total_samples = len(data['BMI'])

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
        data["Judge"] = None
        # 筛选出当前组的样本
        if i < k - 1 :
            data['Judge'] = (data['BMI'] >= lower_bound) & (data['BMI'] < upper_bound)
        else:  # '最后一组包含上'限
            data['Judge'] = (data['BMI'] >= lower_bound) & (data['BMI'] <= upper_bound)



        # data1 = data


        group_samples = data[data['Judge'] == True]

        sample_count = len(group_samples)

        # --- 约束检查 ---
        # 如果任何一组的样本量小于20，则返回一个巨大的惩罚值
        if sample_count < 20:
            return float('inf')
        if sample_count > 500:
            return float('inf')




            # 计算该组的加权时间
        # midpoint = (lower_bound + upper_bound) / 2
        # time_at_midpoint = time_func_of_bmi(midpoint)
        ave_age = sum(group_samples["Age"])/sample_count
        ave_height = sum(group_samples["Height"])/sample_count
        ave_weight = sum(group_samples["Weight"])/sample_count
        time_at_upper_bound = time_func_of_bmi(upper_bound,ave_height,ave_weight,ave_age) + 100 * p_value
        weight = sample_count / total_samples

        if time_at_upper_bound < 5:
            return float('inf')
        weighted_time = time_at_upper_bound * weight

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
best_cost, best_pos = optimizer.optimize(pso_objective_function, iters=1000)

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
        data_interval = data[(data["BMI"] >= lower) & (data["BMI"] < upper)]
        average_age = data_interval["Age"].mean()
        average_height = data_interval["Height"].mean()
        average_weight = data_interval["Weight"].mean()
    else:
        mask = (bmi_values >= lower) & (bmi_values <= upper)
        data_interval = data[(data["BMI"] >= lower) & (data["BMI"] <= upper)]
        average_age = data_interval["Age"].mean()
        average_height = data_interval["Height"].mean()
        average_weight = data_interval["Weight"].mean()
    count = np.sum(mask)
    # midpoint_val = (lower + upper) / 2
    time_val = time_func_of_bmi(upper,average_height,average_weight,average_age)
    weight_val = count / total_samples

    results_data.append({
        "组别": i + 1,
        "BMI范围": f"[{lower:.2f}, {upper:.2f})",
        "人数": count,
        "平均年龄": average_age,
        "平均身高": average_height,
        "平均体重": average_weight,
        # "范围中点": f"{midpoint_val:.2f}",
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


