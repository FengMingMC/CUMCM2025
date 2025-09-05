import numpy as np
from sympy import symbols, solve, Eq

speedOfMs = 300
originPoint = np.array([0,0,0])
locationOfTrue = np.array([0,200,0])
gravity = -9.81

time = symbols('t')

locationOfMs1 = np.array([20000,0,2000])
locationOfFy1 = np.array([17800,0,1800])
speedOfFy1 = 120

# np.linalg.norm(locationOfFy1)
directionSpeedOfFy1 = speedOfFy1 * ((originPoint - locationOfFy1) / np.linalg.norm(originPoint - locationOfFy1))
directionSpeedOfMs1 = speedOfMs * ((originPoint - locationOfMs1) / np.linalg.norm(originPoint - locationOfMs1))
print(directionSpeedOfFy1)

timeOfFy1 = 1.5
timeOfFy1Boom = 3.6




releasePoint = locationOfFy1 - (timeOfFy1 + timeOfFy1Boom) * directionSpeedOfFy1
gravityDistant = np.array([0, 1 / 2 * gravity * timeOfFy1Boom * timeOfFy1Boom, 0])

boomingPoint = releasePoint + gravityDistant
boomingMsPoint = locationOfMs1 + (timeOfFy1 + timeOfFy1Boom) * directionSpeedOfMs1

def centerPointOfBooming(t,position:np.ndarray):
    return position - t * np.array([0,3,0])

def MsPosition(t,position:np.ndarray,directionSpeedOfMs1:np.ndarray):
    return position + t * directionSpeedOfMs1

def point_to_line_distance(p1: np.ndarray,
                           p2: np.ndarray,
                           p3: np.ndarray,
                           eps: float = 1e-12) -> np.ndarray:
    """
    计算点 p3 到由 p1、p2 确定的直线的距离。
    支持向量化：p3 可以是 (..., 3) 的数组，返回形状为 (...) 的距离数组。

    参数
    ----
    p1, p2 : (3,)  ndarray
        直线上两点
    p3     : (..., 3) ndarray
        待测点，可批量
    eps    : float
        避免除 0 的小数

    返回
    ----
    distance : (...) ndarray
        点到直线的欧氏距离
    """
    p1, p2, p3 = np.asarray(p1), np.asarray(p2), np.asarray(p3)
    # 方向向量
    line_vec = p2 - p1                        # (3,)
    # 向量 p1->p3
    p1p3 = p3 - p1                            # (..., 3)

    # 叉乘模长 ||(p3-p1) × dir||_2
    cross = np.cross(p1p3, line_vec, axis=-1)  # (..., 3)
    norm_cross = np.linalg.norm(cross, axis=-1)  # (...)

    # 直线方向向量模长
    len_line = np.linalg.norm(line_vec)

    # 避免直线退化为点
    if len_line < eps:
        return np.full(norm_cross.shape, np.inf)

    return norm_cross / len_line

print(point_to_line_distance(locationOfTrue, MsPosition(0, boomingMsPoint, directionSpeedOfMs1), centerPointOfBooming(0,boomingPoint )))



# print(boomingPoint)



# print(directionOfFy1)



