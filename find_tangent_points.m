function [T1, T2] = find_tangent_points(P_ext, C_center, R)
    % 计算从外部点 P_ext 到一个圆的两个切点 T1 和 T2
    % P_ext: 外部点的二维坐标 [x, y]
    % C_center: 圆心的二维坐标 [x, y]
    % R: 圆的半径

    d_sq = sum((P_ext - C_center).^2);
    
    if d_sq < R^2
        T1 = [NaN, NaN]; T2 = [NaN, NaN];
        return;
    end

    % 向量：从圆心指向外部点
    vec_CP = P_ext - C_center;
    d = sqrt(d_sq);
    
    % 计算切点弦的中点 M
    a = R^2 / d;
    M = C_center + (a / d) * vec_CP;
    
    % 计算中点到切点的距离 h
    h = sqrt(max(0, R^2 - a^2)); % 使用max(0,...)防止浮点数误差导致负值

    % 计算从 M 到切点的向量
    vec_MT_perp = [vec_CP(2), -vec_CP(1)]; % 与 vec_CP 垂直的向量
    vec_MT = (h / norm(vec_MT_perp)) * vec_MT_perp;
    
    % 计算两个切点
    T1 = M + vec_MT;
    T2 = M - vec_MT;
end