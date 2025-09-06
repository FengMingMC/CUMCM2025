import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
from scipy.stats import pearsonr, spearmanr
import statsmodels.api as sm
import seaborn as sns

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

df_girl = pd.read_excel('data/附件.xlsx', sheet_name = '女胎检测数据' )

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

df_girl['孕周'] = df_girl.apply(calculate_week, axis=1)

groups = df_girl['孕妇代码'].unique()


def BMIFill(row):
    if row["孕妇BMI"] is None:
        row["孕妇BMI"] = row["体重"]/((row["身高"])*10e-2)**2

df_girl.apply(BMIFill, axis=1)#BMI填补缺失


#唯一比对的读段数/原始读段数 过低值清除
df_temp = df_girl[df_girl["唯一比对的读段数"]/df_girl["原始读段数"]>=0.7]
df_girl = df_temp


def pregnancyTimes(row):
    # 怀孕次数转化
    if row["怀孕次数"] == "≥3":
        return 3
    else:
        return row["怀孕次数"]

    # 获取 '生产次数' 列的值
    return mapping[row["怀孕次数"]]

df_girl['PregnancyTimes'] = df_girl.apply(pregnancyTimes, axis=1)


# 中位数获取
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
all_cols = df_girl.columns.tolist()

# 分离出数值列和非数值列，排除分组列
numeric_cols = df_girl.select_dtypes(include=np.number).columns.tolist()
# 确保分组列不被重复处理
numeric_cols = [col for col in numeric_cols if col not in group_cols]

non_numeric_cols = df_girl.select_dtypes(exclude=np.number).columns.tolist()
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

df_temp = df_girl.groupby(group_cols).agg(agg_dict).reset_index()

df_girl = df_temp



df_girl = df_girl[df_girl['IVF妊娠'].notnull()]
def Method(row):
    mapping = {
        "自然受孕" : 0,
        "IVF（试管婴儿）" : 1
    }
    method_str = str(row["IVF妊娠"])
    return mapping[method_str]

df_girl['Method'] = df_girl.apply(Method, axis=1)#受孕方法


# def TripleChromosome(row):
#     if row["染色体的非整倍体"] is None:
#         return 0
#     else:
#         return 1
#
# df_girl['TripleChromosome'] = df_girl.apply(TripleChromosome, axis=1)

# 向量化操作，将 '染色体的非整倍体' 列中非 None 的值映射为 1，None 的值映射为 0
df_girl['TripleChromosome'] = df_girl["染色体的非整倍体"].notna().astype(int)

# print(df_girl)

