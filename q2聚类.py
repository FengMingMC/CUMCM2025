# %% 1. 导入所需库
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
import warnings
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error
from scipy.stats import norm

# %% 2. 环境设置
# 设置绘图样式与中文字体，确保图表中的中文能够正确显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# sns.set_style("whitegrid")
# 忽略多项式拟合可能产生的警告
warnings.filterwarnings('ignore', message='Polyfit may be poorly conditioned')

# %% 3. 读取并准备数据
print("--- 步骤 1: 正在读取和预处理数据 ---")
file_path = '男胎检测数据.csv'
try:
    data = pd.read_csv(file_path, encoding='utf-8')
except UnicodeDecodeError:
    data = pd.read_csv(file_path, encoding='gbk')

# --- 关键修正：添加完整的列重命名和数据转换 ---
# 重命名所有需要的列
data.rename(columns={
    "孕妇代码": "PatientID",
    "检测孕周": "GestWeekStr",
    "Y染色体浓度": "Y_Concentration",
    "孕妇BMI": "BMI"
}, inplace=True)

# 定义一个函数来转换孕周格式
def parse_gest_week(week_str):
    if isinstance(week_str, str):
        if '+' in week_str:
            parts = week_str.split('w+')
            # 确保parts[1]不为空
            if parts[1]:
                return float(parts[0]) + float(parts[1]) / 7
        # 处理'12w'这种格式
        return float(week_str.replace('w', ''))
    return np.nan # 如果格式不正确，返回NaN

# 应用函数，创建数值型的孕周列 'GestWeek'
data['GestWeek'] = data['GestWeekStr'].apply(parse_gest_week)

# 移除核心变量中任何包含缺失值的行
required_cols = ['PatientID', 'GestWeek', 'Y_Concentration', 'BMI']
data.dropna(subset=required_cols, inplace=True)
print(f"数据预处理完成，共有 {len(data)} 条有效检测记录。")
# ----------------------------------------------------

# 计算每个孕妇的BMI平均值，这是我们聚类的对象
avg_bmi_per_patient = data.groupby('PatientID')['BMI'].mean().dropna()

# K-Means算法要求输入是二维数组，因此我们需要将数据变形
bmi_data_for_clustering = avg_bmi_per_patient.values.reshape(-1, 1)

print(f"将对 {len(avg_bmi_per_patient)} 位独立孕妇的平均BMI进行聚类分析。")

# %% 4. 使用“肘部法则”确定最佳聚类数量 (k)
print("\n--- 步骤 2: 正在使用肘部法则确定最佳k值 ---")
model = KMeans(init='k-means++', random_state=42, n_init='auto')
visualizer = KElbowVisualizer(model, k=(2, 10))
visualizer.fit(bmi_data_for_clustering)
visualizer.show() # 显示图形
optimal_k = visualizer.elbow_value_
print(f"肘部法则建议的最佳k值为: {optimal_k}")

# %% 5. 执行最终的K-Means++聚类
print(f"\n--- 步骤 3: 使用 k={optimal_k} 进行K-Means++聚类 ---")
kmeans = KMeans(n_clusters=optimal_k, init='k-means++', random_state=42, n_init='auto')
cluster_labels = kmeans.fit_predict(bmi_data_for_clustering)

# 将聚类结果整理到DataFrame中
results_df = avg_bmi_per_patient.to_frame(name='Avg_BMI').reset_index()
results_df['Cluster'] = cluster_labels

# 分析每个簇的BMI统计特征
bmi_ranges = results_df.groupby('Cluster')['Avg_BMI'].agg(['count', 'min', 'mean', 'max'])
print("聚类完成，各BMI区间的划分为：")
print(bmi_ranges)

# %% 6. 将聚类结果应用回原始数据
print("\n--- 步骤 4: 正在为后续分析准备数据 ---")
data_with_clusters = pd.merge(data, results_df[['PatientID', 'Cluster']], on='PatientID')
print("已将聚类标签合并回原始数据。")

# %% 7. 为每一个BMI群组建立预测模型
print("\n--- 步骤 5: 正在为每个BMI群组建立Y浓度-孕周的预测模型 ---")
models = {}
model_rmse = {}

for i in range(optimal_k):
    group_data = data_with_clusters[data_with_clusters['Cluster'] == i]
    
    X_group = group_data[['GestWeek']]
    y_group = group_data['Y_Concentration']
    
    degree = 2
    poly_model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    poly_model.fit(X_group, y_group)
    
    models[i] = poly_model
    y_pred = poly_model.predict(X_group)
    rmse = np.sqrt(mean_squared_error(y_group, y_pred))
    model_rmse[i] = rmse
    
    print(f"  - 群组 {i} 的模型训练完成，RMSE = {rmse:.4f}")

# %% 8. 定义风险函数
print("\n--- 步骤 6: 正在定义风险函数及求解最优时点 ---")
ALPHA = 0.8  # 两种风险的权重平衡因子 (可调整)
W_LOW = 1    # 早期窗口风险权重
W_MID = 5    # 中期窗口风险权重
W_HIGH = 20  # 晚期窗口风险权重

def calculate_total_risk(t, model, rmse, alpha=ALPHA):
    """计算总风险 R(t)"""
    # 预测Y浓度均值
    mu_t = model.predict(np.array([[t]]))[0]
    # 计算不准确风险
    r_inacc = norm.cdf(x=0.04, loc=mu_t, scale=rmse)
    # 计算窗口期风险
    if t <= 12: r_treat = W_LOW
    elif 13 <= t <= 27: r_treat = W_MID
    else: r_treat = W_HIGH
    # 加权求和
    return alpha * r_inacc + (1 - alpha) * r_treat

# %% 9. 求解最优化问题
weeks_to_test = np.arange(10, 25, 0.1)
optimal_results = {}
risk_curves = {}

for i in range(optimal_k):
    current_model = models[i]
    current_rmse = model_rmse[i]
    
    risks_data = [calculate_total_risk(t, current_model, current_rmse) for t in weeks_to_test]
    
    min_risk_index = np.argmin(risks_data)
    optimal_week = weeks_to_test[min_risk_index]
    min_risk_value = risks_data[min_risk_index]
    
    optimal_results[i] = {'Optimal_Week': optimal_week, 'Min_Risk': min_risk_value}
    risk_curves[i] = risks_data
    
    print(f"  - 群组 {i} 的最佳检测时间点为: {optimal_week:.2f} 周 (最小风险值为: {min_risk_value:.4f})")

# %% 10. 结果汇总与可视化
print("\n--- 步骤 7: 结果汇总与可视化 ---")
optimal_weeks_df = pd.DataFrame.from_dict(optimal_results, orient='index')
final_summary_table = bmi_ranges.join(optimal_weeks_df)

print("\n\n--- 问题二：最终解决方案汇总 ---")
print(final_summary_table)

# 风险曲线可视化
print("\n正在绘制各群组的风险曲线...")
fig, axes = plt.subplots(optimal_k, 1, figsize=(12, 5 * optimal_k), sharex=True, squeeze=False)

for i in range(optimal_k):
    ax = axes[i, 0]
    
    # 绘制总风险曲线
    ax.plot(weeks_to_test, risk_curves[i], label='总风险 (Total Risk)', color='red', linewidth=2.5)
    
    # 标记最低风险点
    optimal_week = optimal_results[i]['Optimal_Week']
    min_risk = optimal_results[i]['Min_Risk']
    ax.axvline(x=optimal_week, color='purple', linestyle=':', label=f'最佳时点: {optimal_week:.2f} 周')
    ax.plot(optimal_week, min_risk, 'r*', markersize=15)
    
    # 设置图表标题和标签
    cluster_info = final_summary_table.loc[i]
    title = (f"群组 {i} 的风险曲线 (BMI范围: [{cluster_info['min']:.2f}, {cluster_info['max']:.2f}], "
             f"人数: {int(cluster_info['count'])})")
    ax.set_title(title)
    ax.set_ylabel('风险值 (Risk Value)')
    ax.legend()
    ax.grid(True)

axes[-1, 0].set_xlabel('检测孕周 (Gestational Week)')
plt.suptitle('各BMI群组的NIPT检测时点风险分析', fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
