function is_intersect = is_segment_sphere_intersect(P1, P2, Sc, R)
    % 判断线段 P1-P2 是否与球心为 Sc、半径为 R 的球体相交
    % P1: 线段起点
    % P2: 线段终点
    % Sc: 球心位置
    % R: 球体半径
    
    v = P2 - P1;      
    w = Sc - P1;      
    
    c1 = dot(w, v);
    if c1 <= 0
        dist_sq = dot(w, w);
    else
        c2 = dot(v, v);
        if c2 <= c1
            dist_sq = dot(P2 - Sc, P2 - Sc);
        else
            b = c1 / c2;
            Pb = P1 + b * v;
            dist_sq = dot(Sc - Pb, Sc - Pb);
        end
    end
    
    if dist_sq <= R^2
        is_intersect = true;
    else
        is_intersect = false;
    end
end