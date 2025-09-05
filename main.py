import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

# 设置pandas的显示选项，确保所有行和列都能完整显示
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# --- 文件内容加载 ---
# 假设您已将"附件.xlsx - 男胎检测数据.csv"和"附件.xlsx - 女胎检测数据.csv"上传
male_data_content = """{{file_content_fetcher.fetch(query="男胎检测数据", source_references=["attachment_1"])}}"""
female_data_content = """{{file_content_fetcher.fetch(query="女胎检测数据", source_references=["attachment_2"])}}"""

# 将文件内容转换为DataFrame
try:
    df_male = pd.read_csv(pd.io.common.StringIO(male_data_content))
    # 结合男胎和女胎数据，如果需要
    # df_female = pd.read_csv(pd.io.common.StringIO(female_data_content))
    # df = pd.concat([df_male, df_female], ignore_index=True)
    df = df_male
except pd.errors.ParserError as e:
    print(f"解析CSV文件时出错: {e}")
    # 如果解析失败，您可以尝试使用其他参数重新读取，例如 delimiter=',' 或 encoding='utf-8'


# --- 数据预处理和对数转换 ---
# 1. 将'检测孕周'转换为数值
def convert_gestational_week(week_str):
    if pd.isna(week_str):
        return np.nan
    try:
        parts = str(week_str).split('w+')
        weeks = int(parts[0])
        days = int(parts[1]) if len(parts) > 1 else 0
        return weeks + days / 7
    except (ValueError, IndexError):
        return np.nan


df['Gestational_Weeks'] = df['检测孕周'].apply(convert_gestational_week)

# 2. 选择相关列，并处理缺失值
df_filtered = df[['Gestational_Weeks', '孕妇BMI', 'Y染色体浓度']].dropna()

# 3. 筛选掉非正值，以进行对数转换
df_log = df_filtered[(df_filtered['Gestational_Weeks'] > 0) &
                     (df_filtered['孕妇BMI'] > 0) &
                     (df_filtered['Y染色体浓度'] > 0)].copy()

if df_log.empty:
    print("没有有效的正值数据进行对数转换，请检查原始数据。")
else:
    # 4. 对所有变量进行对数转换，以线性化模型
    df_log['log_Y_Conc'] = np.log(df_log['Y染色体浓度'])
    df_log['log_Gestational_Weeks'] = np.log(df_log['Gestational_Weeks'])
    df_log['log_BMI'] = np.log(df_log['孕妇BMI'])

    # --- 模型拟合 (d=0的情况) ---
    # 构建因变量和自变量
    Y = df_log['log_Y_Conc']
    X = df_log[['log_Gestational_Weeks', 'log_BMI']]

    # 添加截距项，对应于线性模型中的 log(a)
    X = sm.add_constant(X)

    # 拟合OLS（普通最小二乘）模型
    model = sm.OLS(Y, X).fit()

    # 打印模型摘要，其中包含显著性信息
    print("--- 模型拟合摘要 ---")
    print(model.summary())

    # --- 残差分析 ---
    # 计算预测值和残差
    y_pred = model.predict(X)
    residuals = model.resid

    # 绘制残差图，以检查模型假设
    plt.figure(figsize=(12, 6))

    # 残差与拟合值的散点图
    plt.subplot(1, 2, 1)
    sns.scatterplot(x=y_pred, y=residuals)
    plt.axhline(0, color='red', linestyle='--')
    plt.title('Residuals vs. Fitted')
    plt.xlabel('Fitted values (Predicted log(Y_Conc))')
    plt.ylabel('Residuals')

    # 残差的正态性Q-Q图
    plt.subplot(1, 2, 2)
    sm.qqplot(residuals, line='s', ax=plt.gca())
    plt.title('Normal Q-Q Plot of Residuals')

    plt.tight_layout()
    plt.show()

# --- 结果解读 ---
# 从模型中提取 a, b, c 的值
# a = exp(截距)
# b = 'log_Gestational_Weeks' 的系数
# c = 'log_BMI' 的系数

if 'model' in locals():
    a = np.exp(model.params['const'])
    b = model.params['log_Gestational_Weeks']
    c = model.params['log_BMI']

    print("\n--- 拟合结果 ---")
    print(f"拟合的模型为: Y染色体浓度 ≈ {a:.4f} * (孕期)^{b:.4f} * (孕妇BMI)^{c:.4f}")

    # 显著性分析
    print("\n--- 显著性分析 ---")
    if model.pvalues['log_Gestational_Weeks'] < 0.05:
        print(f"孕期变量的p值为 {model.pvalues['log_Gestational_Weeks']:.4f}，在0.05显著性水平下是显著的。")
    else:
        print(f"孕期变量的p值为 {model.pvalues['log_Gestational_Weeks']:.4f}，在0.05显著性水平下不显著。")

    if model.pvalues['log_BMI'] < 0.05:
        print(f"孕妇BMI变量的p值为 {model.pvalues['log_BMI']:.4f}，在0.05显著性水平下是显著的。")
    else:
        print(f"孕妇BMI变量的p值为 {model.pvalues['log_BMI']:.4f}，在0.05显著性水平下不显著。")