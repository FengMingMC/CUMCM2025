clear;close all;clc;

% --- 0.读取原始数据 ---
data = readtable('男胎检测数据.csv','VariableNamingRule','preserve');

% 修改表头，中文变英文
data = renamevars(data, ["孕妇代码", "检测孕周", "Y染色体浓度", "孕妇BMI"], ...
                        ["PatientID", "GestationalWeekStr", "Y_Concentration", "BMI"]);


% --- 1. 数据清洗与转换 ---
% 将字符串格式的孕周 ('11w+6') 转换为数值 (11.86)
% 使用 sscanf 来解析 'w+' 分隔的两个数字
weeks_numeric = zeros(height(data), 1);
for i = 1:height(data)
    parts = sscanf(data.GestationalWeekStr{i}, '%dw+%d');
    if length(parts) == 2 % 处理可能存在的 '11w+6' 这种格式
        weeks_numeric(i) = parts(1) + parts(2)/7;
    else % 处理可能存在的 '12w' 这种格式
        parts = sscanf(data.GestationalWeekStr{i}, '%dw');
        weeks_numeric(i) = parts(1);
    end
end
% 将转换后的数值孕周作为一个新列添加到表中
data.GestationalWeek = weeks_numeric;

% --- 2. 绘制 Y染色体浓度 vs. 检测孕周 的个体轨迹图 ---

% 创建一个新的图形窗口
figure;
hold on; % 允许在同一张图上叠加多条线

% 获取所有唯一的孕妇ID
unique_patients = unique(data.PatientID);

% 循环遍历每一个唯一的孕妇ID
for i = 1:length(unique_patients)
    % 提取当前孕妇的所有数据
    patient_data = data(strcmp(data.PatientID, unique_patients{i}), :);
    
    % 按照孕周从小到大排序，这是连接数据点的关键
    patient_data = sortrows(patient_data, 'GestationalWeek');
    
    % 绘制当前孕妇的数据点和连接线
    % '-o' 表示用圆圈标记数据点，并用实线连接
    plot(patient_data.GestationalWeek, patient_data.Y_Concentration, '-o', ...
        'MarkerSize', 4, 'LineWidth', 1);
end

% --- 3. 美化图形 ---
hold off; % 结束叠加绘图
title('Y染色体浓度随孕周变化的个体轨迹图');
xlabel('检测孕周 (周)');
ylabel('Y染色体浓度');
grid on; % 添加网格线
legend(unique_patients, 'Location', 'bestoutside'); % 添加图例，可能会很多，视情况决定是否显示

% --- 4.spearman & pearson ---
% 提取需要分析的两列数据
y_concentration = data.Y_Concentration;
gestational_week = data.GestationalWeek;
bmi = data.BMI;

% 计算Pearson相关系数和p值
% 'rows','complete' 这个参数对会自动忽略任何包含NaN（缺失值）的行
[r_gest, p_gest] = corr(y_concentration, gestational_week, 'rows', 'complete');

% 显示结果
fprintf('Y浓度与孕周的Pearson相关性分析:\n');
fprintf('相关系数 r = %.4f\n', r_gest);
fprintf('p-value = %.4f\n', p_gest);

% 根据p值判断显著性
if p_gest < 0.05
    fprintf('结果在统计上是显著的 (p < 0.05)，表明两者存在线性相关关系。\n');
else
    fprintf('结果在统计上不显著 (p >= 0.05)，不能认为两者存在线性相关关系。\n');
end

disp('-------------------------------');

[r_gest, p_gest] = corr(y_concentration, bmi, 'rows', 'complete');

% 显示结果
fprintf('Y浓度与BMI的Pearson相关性分析:\n');
fprintf('相关系数 r = %.4f\n', r_gest);
fprintf('p-value = %.4f\n', p_gest);

% 根据p值判断显著性
if p_gest < 0.05
    fprintf('结果在统计上是显著的 (p < 0.05)，表明两者存在线性相关关系。\n');
else
    fprintf('结果在统计上不显著 (p >= 0.05)，不能认为两者存在线性相关关系。\n');
end

disp('-------------------------------');

% 计算Spearman相关系数和p值
[rho_gest, p_gest_spearman] = corr(y_concentration, gestational_week, ...
                                 'type', 'Spearman', 'rows', 'complete');

% 显示结果
fprintf('\nY浓度与孕周的Spearman相关性分析:\n');
fprintf('相关系数 rho = %.4f\n', rho_gest);
fprintf('p-value = %.4f\n', p_gest_spearman);

if p_gest_spearman < 0.05
    fprintf('结果在统计上是显著的 (p < 0.05)，表明两者存在单调相关关系。\n');
else
    fprintf('结果在统计上不显著 (p >= 0.05)，不能认为两者存在单调相关关系。\n');
end

disp('-------------------------------');

% 计算Spearman相关系数和p值
[rho_bmi, p_bmi_spearman] = corr(y_concentration, bmi, ...
                               'type', 'Spearman', 'rows', 'complete');

% 显示结果
fprintf('\nY浓度与BMI的Spearman相关性分析:\n');
fprintf('相关系数 rho = %.4f\n', rho_bmi);
fprintf('p-value = %.4f\n', p_bmi_spearman);

if p_bmi_spearman < 0.05
    fprintf('结果在统计上是显著的 (p < 0.05)。\n');
else
    fprintf('结果在统计上不显著 (p >= 0.05)。\n');
end