% SOLVE_PROBLEM1_FINAL
% -------------------------------------------------------------------------
% 本脚本为2025年高教社杯A题问题1的最终精确解法。
% 模型：动态轮廓遮蔽 (Dynamic Silhouette Shielding)
% 该模型在每个时间步动态计算目标在导弹视角下的轮廓，并判断该轮廓是否被完全遮蔽。
% -------------------------------------------------------------------------

clear;
clc;
close all;

%% 1. 定义常量和初始条件
fprintf('正在初始化模型参数...\n');
% 物理常量
g = 9.8; % 重力加速度 (m/s^2)

% 导弹 M1
P_M1_0 = [20000, 0, 2000]; % M1 初始位置 (m)
v_M1 = 300; % M1 速度 (m/s)

% 无人机 FY1
P_FY1_0 = [17800, 0, 1800]; % FY1 初始位置 (m)
v_FY1 = 120; % FY1 速度 (m/s)

% 目标
P_FalseTarget = [0, 0, 0]; % 假目标位置 (m)
Cylinder_center_xy = [0, 200]; % 圆柱体中心轴的XY坐标
Cylinder_radius = 7;           % 圆柱体半径 (m)
Cylinder_height = 10;          % 圆柱体高度 (m)

% 烟幕
R_Smoke = 10; % 烟幕云团半径 (m)
v_Smoke_sink = 3; % 烟幕云团下沉速度 (m/s)
t_smoke_effective_duration = 20; % 烟幕有效持续时间 (s)

% 事件时间
t_drop = 1.5; % 投弹时间 (s)
t_detonation_delay = 3.6; % 弹体起爆延迟 (s)

%% 2. 计算关键事件的轨迹和坐标
fprintf('正在计算关键事件点...\n');
% 飞行方向向量
dir_M1 = (P_FalseTarget - P_M1_0) / norm(P_FalseTarget - P_M1_0);
dir_FY1 = (P_FalseTarget - P_FY1_0) / norm(P_FalseTarget - P_FY1_0);
V_FY1 = v_FY1 * dir_FY1;

% 投弹点
P_drop = P_FY1_0 + V_FY1 * t_drop;

% 起爆点 (抛体运动)
P_explosion = P_drop + [V_FY1(1)*t_detonation_delay, ...
                        V_FY1(2)*t_detonation_delay, ...
                        V_FY1(3)*t_detonation_delay - 0.5*g*t_detonation_delay^2];

% 烟幕有效时间窗口
t_explosion = t_drop + t_detonation_delay;
t_shield_start = t_explosion;
t_shield_end = t_explosion + t_smoke_effective_duration;

fprintf('烟幕弹于 t = %.2f s 在坐标 [%.2f, %.2f, %.2f] 起爆。\n', ...
    t_explosion, P_explosion(1), P_explosion(2), P_explosion(3));
fprintf('遮蔽效果判定区间为: t = [%.2f s, %.2f s]\n', t_shield_start, t_shield_end);


%% 3. 循环仿真并计算遮蔽时长 (精确模型)
fprintf('开始进行时间步长仿真...\n');
dt = 0.01; 
total_shielding_time = 0;

for t = t_shield_start : dt : t_shield_end
    
    % (a) 计算当前时刻导弹和烟幕中心位置
    P_M1_t = P_M1_0 + dir_M1 * v_M1 * t;
    P_Smoke_Center_t = P_explosion - [0, 0, v_Smoke_sink * (t - t_explosion)];
    
    % (b) 动态计算真目标的轮廓点
    P_M1_xy = [P_M1_t(1), P_M1_t(2)];
    [T1_xy, T2_xy] = find_tangent_points(P_M1_xy, Cylinder_center_xy, Cylinder_radius);
    
    if ~isnan(T1_xy(1)) % 仅当切点有效时进行判断
        % 构建四个三维动态轮廓点
        P_T1 = [T1_xy(1), T1_xy(2), 0];               % 左下
        P_T2 = [T1_xy(1), T1_xy(2), Cylinder_height]; % 左上
        P_T3 = [T2_xy(1), T2_xy(2), 0];               % 右下
        P_T4 = [T2_xy(1), T2_xy(2), Cylinder_height]; % 右上

        % (c) 判断所有轮廓视线是否均被遮蔽
        is_shielded_1 = is_segment_sphere_intersect(P_M1_t, P_T1, P_Smoke_Center_t, R_Smoke);
        is_shielded_2 = is_segment_sphere_intersect(P_M1_t, P_T2, P_Smoke_Center_t, R_Smoke);
        is_shielded_3 = is_segment_sphere_intersect(P_M1_t, P_T3, P_Smoke_Center_t, R_Smoke);
        is_shielded_4 = is_segment_sphere_intersect(P_M1_t, P_T4, P_Smoke_Center_t, R_Smoke);

        if is_shielded_1 && is_shielded_2 && is_shielded_3 && is_shielded_4
            total_shielding_time = total_shielding_time + dt;
        end
    end
end

fprintf('仿真完成。\n');

%% 4. 显示最终结果
fprintf('\n---------------------------------------------------\n');
fprintf('问题1最终求解结果：\n');
fprintf('烟幕干扰弹对M1的有效遮蔽总时长为: %.4f s\n', total_shielding_time);
fprintf('---------------------------------------------------\n');