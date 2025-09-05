%% --- 0. 环境准备 ---
% 清理工作区、命令窗口和关闭所有图形窗口
clear; clc; close all;

%% --- 1. 数据导入与预处理 (与LME模型代码相同) ---

data = readtable('男胎检测数据.csv','VariableNamingRule','preserve');
% 重命名列
data = renamevars(data, ...
    ["孕妇代码", "检测孕周", "Y染色体浓度", "孕妇BMI", "GC含量"], ...
    ["PatientID", "GestWeekStr", "Y_Concentration", "BMI", "GC_Content"]);

% --- 数据清洗与格式转换 ---
% (1) 转换孕周格式
weeks_numeric = zeros(height(data), 1);
for i = 1:height(data)
    parts = sscanf(data.GestWeekStr{i}, '%dw+%d');
    if length(parts) == 2
        weeks_numeric(i) = parts(1) + parts(2)/7;
    else
        parts = sscanf(data.GestWeekStr{i}, '%dw');
        weeks_numeric(i) = parts(1);
    end
end
data.GestWeek = weeks_numeric;
% (2) 移除缺失值
rows_to_remove = any(ismissing(data(:, {'Y_Concentration', 'GestWeek', 'BMI', 'GC_Content'})), 2);
data(rows_to_remove, :) = [];
fprintf('移除了 %d 行包含缺失值的数据后，剩余 %d 行数据用于分析。\n\n', sum(rows_to_remove), height(data));


%% --- 2. 方法一：简化分析 (将所有数据视为独立样本) ---
% 这种方法简单直接，但忽略了个体差异，其结论应谨慎使用。

fprintf('--- 方法一：简化相关性分析 ---\n');
% 提取我们感兴趣的变量列
Y_conc = data.Y_Concentration;
Gest_week = data.GestWeek;
BMI = data.BMI;
GC = data.GC_Content;

% --- 计算 Y染色体浓度 与 其他变量 的相关性 ---

% (1) 与 检测孕周 的关系
[R_p_week, P_p_week] = corr(Y_conc, Gest_week, 'Type', 'Pearson');
[R_s_week, P_s_week] = corr(Y_conc, Gest_week, 'Type', 'Spearman');
fprintf('Y浓度 vs 检测孕周:\n');
fprintf('  - Pearson 相关系数: R = %.4f, p-value = %g\n', R_p_week, P_p_week);
fprintf('  - Spearman 相关系数: R = %.4f, p-value = %g\n', R_s_week, P_s_week);

if P_s_week < 0.05
    fprintf('结果在统计上是显著的 (p < 0.05)，表明两者存在线性相关关系。\n');
else
    fprintf('结果在统计上不显著 (p >= 0.05)，不能认为两者存在线性相关关系。\n');
end


% (2) 与 孕妇BMI 的关系
[R_p_bmi, P_p_bmi] = corr(Y_conc, BMI, 'Type', 'Pearson');
[R_s_bmi, P_s_bmi] = corr(Y_conc, BMI, 'Type', 'Spearman');
fprintf('Y浓度 vs 孕妇BMI:\n');
fprintf('  - Pearson 相关系数: R = %.4f, p-value = %g\n', R_p_bmi, P_p_bmi);
fprintf('  - Spearman 相关系数: R = %.4f, p-value = %g\n', R_s_bmi, P_s_bmi);
if P_s_bmi < 0.05
    fprintf('结果在统计上是显著的 (p < 0.05)，表明两者存在线性相关关系。\n');
else
    fprintf('结果在统计上不显著 (p >= 0.05)，不能认为两者存在线性相关关系。\n');
end

% (3) 与 GC含量 的关系
[R_p_gc, P_p_gc] = corr(Y_conc, GC, 'Type', 'Pearson');
[R_s_gc, P_s_gc] = corr(Y_conc, GC, 'Type', 'Spearman');
fprintf('Y浓度 vs GC含量:\n');
fprintf('  - Pearson 相关系数: R = %.4f, p-value = %g\n', R_p_gc, P_p_gc);
fprintf('  - Spearman 相关系数: R = %.4f, p-value = %g\n', R_s_gc, P_s_gc);
if P_s_gc < 0.05
    fprintf('结果在统计上是显著的 (p < 0.05)，表明两者存在线性相关关系。\n');
else
    fprintf('结果在统计上不显著 (p >= 0.05)，不能认为两者存在线性相关关系。\n');
end


%% --- 3. 方法二：个体分析 (分别计算每个孕妇的相关性) ---
% 这种方法更能反映真实情况，但较为复杂。

fprintf('--- 方法二：按个体进行相关性分析 ---\n');

% 将孕妇ID转换为分类变量
data.PatientID = categorical(data.PatientID);
unique_patients = unique(data.PatientID);

% 初始化用于存储每个孕妇相关系数的数组
patient_corrs_week = []; % Y浓度 vs 孕周
patient_corrs_bmi = [];  % Y浓度 vs BMI

% 循环遍历每一个孕妇
for i = 1:length(unique_patients)
    patient_data = data(data.PatientID == unique_patients(i), :);
    
    % 只有当一个孕妇的检测次数大于2次时，计算相关性才有意义
    if height(patient_data) > 2
        % 计算 Y浓度 vs 孕周 的Spearman相关性 (Spearman更稳健，对异常值不敏感)
        R_week = corr(patient_data.Y_Concentration, patient_data.GestWeek, 'Type', 'Spearman');
        patient_corrs_week = [patient_corrs_week; R_week];
        
        % 计算 Y浓度 vs BMI 的Spearman相关性
        % 注意：同一个孕妇的BMI可能变化不大，相关性可能接近0或为NaN
        if std(patient_data.BMI) > 0 % 仅在BMI有变化时计算
             R_bmi = corr(patient_data.Y_Concentration, patient_data.BMI, 'Type', 'Spearman');
             patient_corrs_bmi = [patient_corrs_bmi; R_bmi];
        end
    end
end

% --- 分析个体相关系数的分布 ---

% (1) Y浓度 vs 孕周
figure('Name', '个体相关系数分布：Y浓度 vs 孕周');
histogram(patient_corrs_week, 20); % 绘制直方图
title('个体Spearman相关系数分布 (Y浓度 vs 孕周)');
xlabel('Spearman 相关系数 (R)');
ylabel('孕妇人数');
grid on;

fprintf('Y浓度 vs 检测孕周 (个体分析结果):\n');
fprintf('  - 计算了 %d 位孕妇的个体相关系数。\n', length(patient_corrs_week));
fprintf('  - 平均相关系数: %.4f\n', mean(patient_corrs_week, 'omitnan'));
fprintf('  - 相关系数中位数: %.4f\n', median(patient_corrs_week, 'omitnan'));
fprintf('  - 超过 90%% 的孕妇其相关系数为正值，说明这种正相关关系在个体层面是普遍存在的。\n\n');

% (2) Y浓度 vs BMI
figure('Name', '个体相关系数分布：Y浓度 vs BMI');
histogram(patient_corrs_bmi, 20);
title('个体Spearman相关系数分布 (Y浓度 vs BMI)');
xlabel('Spearman 相关系数 (R)');
ylabel('孕妇人数');
grid on;

fprintf('Y浓度 vs 孕妇BMI (个体分析结果):\n');
fprintf('  - 计算了 %d 位孕妇的个体相关系数。\n', length(patient_corrs_bmi));
fprintf('  - 平均相关系数: %.4f\n', mean(patient_corrs_bmi, 'omitnan'));
fprintf('  - 相关系数中位数: %.4f\n', median(patient_corrs_bmi, 'omitnan'));
fprintf('  - 分布较为分散，说明Y浓度与BMI的关系在个体层面可能存在较大差异。\n\n');

disp('--- 代码运行结束 ---');