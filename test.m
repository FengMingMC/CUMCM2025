%% --- 0. 环境准备 ---
% 清理工作区、命令窗口和关闭所有图形窗口
clear; clc; close all;

%% --- 1. 数据导入与预处理 ---

% 定义文件名
data = readtable('男胎检测数据.csv','VariableNamingRule','preserve');

% 为了代码清晰和兼容性，将关键的中文列名重命名为英文
% 注意：请根据你实际读取后的列名进行精确匹配，特别是"唯一比对的读段数"可能因空格产生不同
data = renamevars(data, ...
    ["孕妇代码", "检测孕周", "Y染色体浓度", "孕妇BMI", "GC含量", "年龄"], ...
    ["PatientID", "GestWeekStr", "Y_Concentration", "BMI", "GC_Content", "Age"]);

% --- 数据清洗与格式转换 ---

% (1) 转换孕周格式：将 '11w+6' 字符串转换为 11.86 数值
weeks_numeric = zeros(height(data), 1);
for i = 1:height(data)
    % 使用sscanf解析'w+'格式的字符串
    parts = sscanf(data.GestWeekStr{i}, '%dw+%d');
    if length(parts) == 2
        weeks_numeric(i) = parts(1) + parts(2)/7;
    else % 兼容 "12w" 这种没有天数的格式
        parts = sscanf(data.GestWeekStr{i}, '%dw');
        weeks_numeric(i) = parts(1);
    end
end
% 将转换后的数值孕周作为一个新列添加到表中
data.GestWeek = weeks_numeric;

% (2) 移除缺失值：确保模型用到的所有变量都没有NaN值
% 我们关注的核心变量是 Y_Concentration, GestWeek, BMI, GC_Content, PatientID
% isnan()会检查NaN, ismissing()会检查更广泛的缺失值类型(如空字符串)
rows_to_remove = any(ismissing(data(:, {'Y_Concentration', 'GestWeek', 'BMI', 'GC_Content'})), 2);
data(rows_to_remove, :) = [];
fprintf('移除了 %d 行包含缺失值的数据后，剩余 %d 行数据用于分析。\n', sum(rows_to_remove), height(data));

% (3) 将孕妇ID转换为分类变量，这是混合效应模型的要求
data.PatientID = categorical(data.PatientID);


%% --- 2. 探索性数据分析(EDA): 可视化个体轨迹 ---

fprintf('正在绘制个体轨迹图 (Spaghetti Plot)...\n');
figure('Name', '个体轨迹图：Y染色体浓度 vs 检测孕周');
hold on;

% 获取所有唯一的孕妇ID
unique_patients = unique(data.PatientID);

% 循环遍历每一个孕妇，绘制她所有的数据点并连接成线
for i = 1:length(unique_patients)
    % 提取当前孕妇的所有数据
    patient_data = data(data.PatientID == unique_patients(i), :);
    
    % 按照孕周从小到大排序
    patient_data = sortrows(patient_data, 'GestWeek');
    
    % 绘制数据点和连接线
    plot(patient_data.GestWeek, patient_data.Y_Concentration, '-o', ...
        'MarkerSize', 4, 'LineWidth', 1);
end
hold off;

% 美化图形
title('Y染色体浓度随孕周变化的个体轨迹图');
xlabel('检测孕周 (周)');
ylabel('Y染色体浓度');
grid on;
fprintf('个体轨迹图绘制完成。\n\n');


%% --- 3. 建立并求解线性混合效应模型 (Linear Mixed-Effects Model) ---

% 核心步骤：使用 fitlme 函数来建立模型
% 公式解释: 'Y_Concentration ~ GestWeek + BMI + GC_Content + (1 | PatientID)'
%   - Y_Concentration: 因变量 (Response)
%   - ~: 定义后面的项为预测变量
%   - GestWeek + BMI + GC_Content: 固定效应 (Fixed Effects)。我们想研究的变量。
%   - (1 | PatientID): 随机效应 (Random Effects)。'1'代表截距，'| PatientID'代表
%     我们允许每个孕妇(PatientID)都有自己的基础截距水平。这正是处理"一人多点"的关键。

fprintf('正在建立线性混合效应模型...\n');
% 模型1：随机截距模型 (推荐，最常用且稳健)
lme_model = fitlme(data, 'Y_Concentration ~ GestWeek + BMI + GC_Content + Age + (1 | PatientID)');

% (可选) 模型2：随机截距和随机斜率模型 (更复杂，但能捕捉个体增长率差异)
% lme_model_slope = fitlme(data, 'Y_Concentration ~ GestWeek + BMI + GC_Content + (GestWeek | PatientID)');
% 这个模型假设每个孕妇不仅基础水平不同，浓度随孕周增长的速率也不同。
% 你可以在论文中对比两个模型，然后选择拟合优度更好（如AIC或BIC更小）的模型。

fprintf('模型建立完成。\n\n');

%% --- 4. 分析模型结果并回答问题 ---

% 显示整个模型的详细结果
disp('--- 线性混合效应模型结果 ---');
disp(lme_model);

% 提取并重点显示固定效应的结果，这是回答问题1的关键
fprintf('\n--- 固定效应分析 (Fixed Effects Analysis) ---\n');
disp('该表展示了各变量与Y染色体浓度的关系及其显著性：');
fixed_effects_table = lme_model.Coefficients;
disp(fixed_effects_table);

% --- 结果解读 ---
fprintf('\n--- 结果解读 ---\n');

% 检验孕周的显著性
p_value_gestweek = fixed_effects_table.pValue(strcmp(fixed_effects_table.Name, 'GestWeek'));
if p_value_gestweek < 0.05
    fprintf('1. 检测孕周 (GestWeek) 的影响: p-value = %.4f (< 0.05)，表明检测孕周与Y染色体浓度存在统计学上的【显著】正相关关系。\n', p_value_gestweek);
else
    fprintf('1. 检测孕周 (GestWeek) 的影响: p-value = %.4f (>= 0.05)，表明检测孕周与Y染色体浓度的关系不显著。\n', p_value_gestweek);
end
fprintf('   估计系数为 %.4f，即孕周每增加1周，Y染色体浓度平均增加约 %.4f。\n\n', fixed_effects_table.Estimate(strcmp(fixed_effects_table.Name, 'GestWeek')), fixed_effects_table.Estimate(strcmp(fixed_effects_table.Name, 'GestWeek')));

% 检验BMI的显著性
p_value_bmi = fixed_effects_table.pValue(strcmp(fixed_effects_table.Name, 'BMI'));
if p_value_bmi < 0.05
    fprintf('2. 孕妇BMI (BMI) 的影响: p-value = %.4f (< 0.05)，表明孕妇BMI与Y染色体浓度存在统计学上的【显著】负相关关系。\n', p_value_bmi);
else
    fprintf('2. 孕妇BMI (BMI) 的影响: p-value = %.4f (>= 0.05)，表明孕妇BMI与Y染色体浓度的关系不显著。\n', p_value_bmi);
end
fprintf('   估计系数为 %.4f，即BMI每增加1个单位，Y染色体浓度平均降低约 %.4f。\n\n', fixed_effects_table.Estimate(strcmp(fixed_effects_table.Name, 'BMI')), -fixed_effects_table.Estimate(strcmp(fixed_effects_table.Name, 'BMI')));


% 检验GC含量的显著性
p_value_gc = fixed_effects_table.pValue(strcmp(fixed_effects_table.Name, 'GC_Content'));
if p_value_gc < 0.05
    fprintf('3. GC含量 (GC_Content) 的影响: p-value = %.4f (< 0.05)，表明GC含量与Y染色体浓度存在统计学上的【显著】关系。\n', p_value_gc);
else
    fprintf('3. GC含量 (GC_Content) 的影响: p-value = %.4f (>= 0.05)，表明GC含量与Y染色体浓度的关系不显著。\n', p_value_gc);
    fprintf('我们没有足够的证据证明它的必要性，因此可以考虑将它从模型中剔除。\n\n');
end

% 检验年龄的显著性
p_value_age = fixed_effects_table.pValue(strcmp(fixed_effects_table.Name, 'Age'));
if p_value_age < 0.05
    fprintf('4. 年龄 (Age) 的影响: p-value = %.4f (< 0.05)，表明年龄与Y染色体浓度存在统计学上的【显著】关系。\n', p_value_age);
else
    fprintf('4. 年龄 (Age) 的影响: p-value = %.4f (>= 0.05)，表明年龄与Y染色体浓度的关系不显著。\n', p_value_age);
    disp('我们没有足够的证据证明它的必要性，因此可以考虑将它从模型中剔除。');

end
disp('--- 代码运行结束 ---');

% 分析随机效应
% 计算并显示每个孕妇的随机截距估计值
RE = randomEffects(lme_model);
% disp(RE);

%% --- 5. 结合模型结果简化模型 ---
% 运行只包含显著变量的简化模型
lme_reduced_model = fitlme(data, 'Y_Concentration ~ GestWeek + BMI + (1 | PatientID)');

% 显示简化模型的结果
disp(lme_reduced_model);

%% --- 6. 定量对比模型，选择最优模型 (最终兼容版) ---
fprintf('\n--- 模型对比分析 (Likelihood Ratio Test) ---\n');

% 调用compare函数，它会返回一个包含对比结果的表格(table或dataset)
% 'CheckNesting', true 是一个好习惯，确保比较的模型是嵌套的
comparison_table = compare(lme_reduced_model, lme_model, 'CheckNesting', true);

% 显示完整的对比表格，便于观察
disp('模型对比结果表：');
disp(comparison_table);

% --- 从对比表格中提取p值 (兼容table和dataset) ---
% 首先，用一种兼容的方式获取列名
if isa(comparison_table, 'table')
    % 新版MATLAB，使用VariableNames
    col_names = comparison_table.Properties.VariableNames;
else
    % 旧版MATLAB，使用VarNames
    col_names = comparison_table.Properties.VarNames;
end

% 然后，使用获取到的列名进行判断
if height(comparison_table) > 1 && ismember('pValue', col_names)
    % 提取p值（无论table还是dataset，都可以用点语法提取数据）
    pVal_compare = comparison_table.pValue(2);

    fprintf('\n全模型与简化模型的似然比检验 p-value = %.4f\n', pVal_compare);
    if pVal_compare > 0.05
        fprintf('p-value > 0.05，我们接受"两个模型没有显著差异"的原假设。\n');
        fprintf('这意味着剔除不显著变量是合理的，因此我们选择更简洁的【简化模型】作为最终模型。\n');
    else
        fprintf('p-value < 0.05，我们拒绝原假设，认为简化模型丢失了关键信息。\n');
        fprintf('这意味着剔除的变量中至少有一个是重要的，应重新审视或采用更复杂的【全模型】。\n');
    end
else
    fprintf('未能从对比表格中提取p值，请检查表格内容。\n');
end


% --- 同时，也可以直接从表格中解读AIC/BIC ---
fprintf('\nAIC/BIC 对比:\n');
% 这里的点语法对于table和dataset是通用的
fprintf('  - 全模型 (lme_model) AIC: %.2f\n', comparison_table.AIC(2));
fprintf('  - 简化模型 (lme_reduced_model) AIC: %.2f\n', comparison_table.AIC(1));
fprintf('  (AIC和BIC越小，模型越优)\n');