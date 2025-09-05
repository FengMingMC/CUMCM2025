import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
from scipy.stats import pearsonr, spearmanr
import statsmodels.api as sm
import seaborn as sns
from scipy.optimize import curve_fit


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


groups = df_boy['孕妇代码'].unique()


def pregnancyTimes(row):
    # 怀孕次数转化
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

def get_mode(x):
    """
    Calculates the mode of a Series. Returns the first mode if multiple exist,
    or NaN if the Series is empty.
    """
    if x.empty:
        return np.nan
    mode_values = x.mode()
    return mode_values[0] if not mode_values.empty else np.nan

# 确定分组的列
group_cols = ['孕妇代码', '孕周']

# 找出 DataFrame 中所有列的类型
all_cols = df_boy.columns.tolist()

# 分离出数值列和非数值列，排除分组列
numeric_cols = df_boy.select_dtypes(include=np.number).columns.tolist()
# 确保分组列不被重复处理
numeric_cols = [col for col in numeric_cols if col not in group_cols]

non_numeric_cols = df_boy.select_dtypes(exclude=np.number).columns.tolist()
# 确保分组列不被重复处理（虽然通常它们是数值型）
non_numeric_cols = [col for col in non_numeric_cols if col not in group_cols]

# 创建聚合字典
agg_dict = {}

# 对于数值列，应用中位数聚合
for col in numeric_cols:
    agg_dict[col] = 'median'

# 对于非数值列，应用众数聚合
for col in non_numeric_cols:
    agg_dict[col] = get_mode

df_temp = df_boy.groupby(group_cols).agg(agg_dict).reset_index()

df_boy = df_temp


df_temp = df_boy[df_boy['Y染色体浓度'] <= 0.2]
df_boy = df_temp
for i, group in enumerate(groups):
    df_group = df_boy[df_boy['孕妇代码'] == group]

print(df_boy)




calculate_frame = pd.DataFrame({
    "孕周":df_boy["孕周"],
    "孕妇BMI": df_boy["孕妇BMI"],
    "BMI取自然对数": np.log(df_boy["孕妇BMI"]),
    "BMI取常用对数": np.log10(df_boy["孕妇BMI"]),
    "BMI开方": np.sqrt(df_boy["孕妇BMI"]),
    "GC含量": df_boy["GC含量"],
    "原始读段数": df_boy["原始读段数"],
    "在参考基因组上比对的比例": df_boy["在参考基因组上比对的比例"],
    "重复读段的比例": df_boy["重复读段的比例"],
    "怀孕次数": df_boy['PregnancyTimes'],
    "生产次数": df_boy["生产次数"],
    "X染色体浓度": df_boy["X染色体浓度"],
    "Y染色体浓度": df_boy["Y染色体浓度"]
})



def yPredict(time,bmi):
    return 0.9004 * time ** 0.2299 * bmi ** -0.9195

column_to_normalize = '孕妇BMI'
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
calculate_frame[column_to_normalize] = scaler.fit_transform(calculate_frame[[column_to_normalize]])

from sklearn.cluster import KMeans

# 设定你想要分的簇的数量
inertia = []
k_range = range(1, 11) # 尝试 k 从 1 到 10

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10) # n_init='auto' in newer versions
    kmeans.fit(calculate_frame[[column_to_normalize]].values)
    inertia.append(kmeans.inertia_)


plt.figure(figsize=(10, 6))
plt.plot(k_range, inertia, marker='o')
plt.title('肘部法求最优 k')
plt.xlabel('聚类数 (k)')
plt.ylabel('惯性（聚类内平方和）')
plt.xticks(k_range)
plt.grid(True)
plt.show()

k=5

kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)

calculate_frame['cluster_label'] = kmeans.fit_predict(calculate_frame[[column_to_normalize]].values)

print("\n聚类结果 DataFrame:")
print(calculate_frame)


# 可视化聚类结果
plt.figure(figsize=(10, 6))
# 绘制数据点，颜色由聚类标签决定
scatter = plt.scatter(df.index, df['normalized_column'], c=df['cluster_label'], cmap='viridis', s=50, alpha=0.7)

# 绘制聚类中心
centers = kmeans.cluster_centers_
plt.scatter(df.index, centers, c='red', s=200, alpha=0.9, marker='X', label='Cluster Centers')

plt.title('K-Means Clustering Results on Normalized Column')
plt.xlabel('Data Point Index')
plt.ylabel('Normalized Value')
plt.legend(*scatter.legend_elements(), title='Clusters')
plt.legend()
plt.grid(True)
plt.show()
