# %% 1. 导入所需库
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline, Pipeline 
import warnings

# %% 2. 环境设置
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")
warnings.filterwarnings('ignore')

# %% 3. 读取并准备数据
print("--- 步骤 1: 正在读取和预处理数据 ---")
file_path = '男胎检测数据.csv'
encodings_to_try = ['utf-8', 'utf-8-sig', 'gb18030', 'gbk']
data = None
for encoding in encodings_to_try:
    try:
        data = pd.read_csv(file_path, encoding=encoding)
        print(f"文件读取成功！使用的编码是: '{encoding}'")
        break
    except UnicodeDecodeError:
        continue
if data is None:
    raise ValueError("所有尝试的编码都无法正确解码文件。")

data.rename(columns={
    "孕妇代码": "PatientID", "检测孕周": "GestWeekStr",
    "Y染色体浓度": "Y_Concentration", "孕妇BMI": "BMI"
}, inplace=True)

def parse_gest_week(week_str):
    if isinstance(week_str, str):
        if '+' in week_str:
            parts = week_str.split('w+')
            if parts[1]: return float(parts[0]) + float(parts[1]) / 7
        return float(week_str.replace('w', ''))
    return np.nan

data['GestWeek'] = data['GestWeekStr'].apply(parse_gest_week)
required_cols = ['PatientID', 'GestWeek', 'Y_Concentration', 'BMI']
data = data[(data['Y_Concentration'] > 0) & (data['GestWeek'] > 0)]
data_clean = data[required_cols].dropna()
print(f"数据预处理完成，共 {len(data_clean)} 条有效检测记录。")

# %% 4. 步骤一：使用K-Means对孕妇进行分群
print("\n--- 步骤 2: 正在使用K-Means对孕妇进行BMI分群 ---")
avg_bmi_per_patient = data_clean.groupby('PatientID')['BMI'].mean()
bmi_for_clustering = avg_bmi_per_patient.values.reshape(-1, 1)
K = 4
kmeans = KMeans(n_clusters=K, init='k-means++', random_state=42, n_init='auto')
cluster_labels = kmeans.fit_predict(bmi_for_clustering)
results_df = avg_bmi_per_patient.to_frame(name='Avg_BMI').reset_index()
results_df['Cluster'] = cluster_labels
data_with_clusters = pd.merge(data_clean, results_df[['PatientID', 'Cluster']], on='PatientID')
print(f"已成功将 {len(avg_bmi_per_patient)} 位孕妇划分为 {K} 个群组。")

# %% 5. 步骤二和三：为各群组海选最优模型并求解
print("\n--- 步骤 3: 正在为各群组海选最优模型并求解推荐孕周 ---")
Y_THRESHOLD = 0.04
final_results_data = []

def get_candidate_models():
    return {
        "线性回归": LinearRegression(),
        "二阶多项式": make_pipeline(PolynomialFeatures(2, include_bias=False), LinearRegression()),
        "三阶多项式": make_pipeline(PolynomialFeatures(3, include_bias=False), LinearRegression()),
        "对数模型(lnY~X)": "log_y",
        "对数模型(Y~lnX)": "log_x"
    }

def evaluate_model(model, X, y):
    n = len(y)
    if isinstance(model, Pipeline):
        model_fit = model.fit(X, y)
        y_pred = model_fit.predict(X)
        k = model.named_steps['polynomialfeatures'].n_output_features_ + 1
    elif isinstance(model, str):
        if model == 'log_y':
            if np.any(y <= 0): return {'RMSE': np.inf, 'Adj_R2': -np.inf, 'AIC': np.inf, 'BIC': np.inf}
            model_fit = LinearRegression().fit(X, np.log(y))
            y_pred = np.exp(model_fit.predict(X))
            k = X.shape[1] + 1
        elif model == 'log_x':
            if np.any(X.values <= 0): return {'RMSE': np.inf, 'Adj_R2': -np.inf, 'AIC': np.inf, 'BIC': np.inf}
            model_fit = LinearRegression().fit(np.log(X), y)
            y_pred = model_fit.predict(np.log(X))
            k = X.shape[1] + 1
    else:
        model_fit = model.fit(X, y)
        y_pred = model_fit.predict(X)
        k = X.shape[1] + 1
            
    r2 = r2_score(y, y_pred)
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - k - 1) if (n - k - 1) > 0 else -np.inf
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    
    rss = np.sum((y - y_pred) ** 2) if np.sum((y - y_pred) ** 2) > 0 else 1e-9
    log_likelihood = -n/2 * np.log(2 * np.pi * rss/n) - n/2
    aic = 2*k - 2*log_likelihood
    bic = k * np.log(n) - 2*log_likelihood
    
    return {'RMSE': rmse, 'Adj_R2': adj_r2, 'AIC': aic, 'BIC': bic, 'model_fit': model_fit, 'type': model if not isinstance(model, str) else model}

def solve_for_t(model_info, threshold=Y_THRESHOLD, search_range=np.arange(5, 40, 0.1)):
    model_fit = model_info['model_fit']
    model_type = model_info['type']
    X_search = search_range.reshape(-1, 1)
    
    if model_type == 'log_y':
        predictions = np.exp(model_fit.predict(X_search))
    elif model_type == 'log_x':
        predictions = model_fit.predict(np.log(X_search))
    else:
        predictions = model_fit.predict(X_search)
        
    reach_indices = np.where(predictions >= threshold)[0]
    if len(reach_indices) > 0:
        return search_range[reach_indices[0]]
    return float('inf')

for i in sorted(data_with_clusters['Cluster'].unique()):
    print(f"\n--- 正在为 BMI分组 {i} 进行模型选择 ---")
    group_data = data_with_clusters[data_with_clusters['Cluster'] == i]
    X_group = group_data[['GestWeek']]
    y_group = group_data['Y_Concentration']
    
    candidate_models = get_candidate_models()
    model_performance = []
    
    for name, model in candidate_models.items():
        performance = evaluate_model(model, X_group, y_group.values)
        performance['name'] = name
        model_performance.append(performance)
        print(f"  - 模型 '{name}': Adj_R2={performance['Adj_R2']:.4f}, AIC={performance['AIC']:.2f}, BIC={performance['BIC']:.2f}")

    best_model_info = min(model_performance, key=lambda x: x['AIC'])
    print(f"  >>> 最佳模型为: '{best_model_info['name']}'")
    
    t_i = solve_for_t(best_model_info)
    clinical_t_i = max(t_i, 10.0)
    
    # #############################################################################
    # --- 这里是唯一的、最终的修正 ---
    # 我们直接在已经包含所有信息的 `results_df` 表中进行筛选和统计
    current_cluster_data = results_df[results_df['Cluster'] == i]
    group_stats = current_cluster_data['Avg_BMI'].agg(['count', 'min', 'max'])
    # #############################################################################
    
    final_results_data.append({
        "BMI分组": f"组 {i}", "孕妇人数": int(group_stats['count']),
        "BMI范围": f"({group_stats['min']:.2f}, {group_stats['max']:.2f}]",
        "最佳拟合模型": best_model_info['name'],
        "模型Adj.R²": f"{best_model_info['Adj_R2']:.3f}",
        "理论达标孕周(周)": f"{t_i:.2f}",
        "最终推荐孕周(周)": f"{clinical_t_i:.2f}"
    })

# %% 6. 汇总与展示最终结果
print("\n\n--- 问题二：最终解决方案汇总 ---")
final_report_df = pd.DataFrame(final_results_data)
final_report_df['sort_key'] = final_report_df['BMI分组'].map(results_df.groupby('Cluster')['Avg_BMI'].mean())
final_report_df.sort_values('sort_key', inplace=True)
final_report_df.drop('sort_key', axis=1, inplace=True)
final_report_df.set_index('BMI分组', inplace=True)
print(final_report_df.to_string())