import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import h, c, k
from scipy.optimize import fsolve
from scipy.interpolate import interp1d

# 加载CIE 1931标准观察者数据
cmf = pd.read_csv('ciexyz31.csv', header=None, names=['wavelength', 'x_bar', 'y_bar', 'z_bar'])

# 加载光谱功率分布数据
spd = pd.read_excel('P1_processed_data.xlsx', sheet_name=0, header=0, names=['wavelength', 'power'])

# 合并前按波长排序
cmf = cmf.sort_values('wavelength')
spd = spd.sort_values('wavelength')

# 合并两个数据集
merged = pd.merge(spd, cmf, on='wavelength', how='inner')

# 计算XYZ三刺激值（使用梯形法数值积分）
delta_lambda = 1  # 波长间隔为1nm
X = np.trapezoid(merged['power'] * merged['x_bar'], dx=delta_lambda)
Y = np.trapezoid(merged['power'] * merged['y_bar'], dx=delta_lambda)
Z = np.trapezoid(merged['power'] * merged['z_bar'], dx=delta_lambda)

# 计算归一化常数（使Y=100）
k = 100 / Y

# 应用归一化得到最终XYZ三刺激值
X_norm = k * X
Y_norm = 100  # 归一化后Y=100
Z_norm = k * Z

# 计算色品坐标(xy)
sum_XYZ = X_norm + Y_norm + Z_norm
x = X_norm / sum_XYZ
y = Y_norm / sum_XYZ

# 计算CIE 1960 UCS坐标(uv)
denominator = X_norm + 15 * Y_norm + 3 * Z_norm
u = 4 * X_norm / denominator
v = 6 * Y_norm / denominator

# 打印基本结果
print("=" * 50)
print(f"XYZ三刺激值: X={X_norm:.6f}, Y={Y_norm:.6f}, Z={Z_norm:.6f}")
print(f"色品坐标: x={x:.6f}, y={y:.6f}")
print(f"CIE 1960 UCS坐标: u={u:.6f}, v={v:.6f}")
print("=" * 50)




# 1. 三角垂足插值法
def triangle_perpendicular_interpolation(u_c, v_c):
    # 生成黑体轨迹数据 (1000K - 25000K, 步长50K)
    T_min, T_max, T_step = 1000, 25000, 50
    temperatures = np.arange(T_min, T_max + T_step, T_step)
    u_bb = np.zeros_like(temperatures, dtype=float)
    v_bb = np.zeros_like(temperatures, dtype=float)

    # 计算黑体轨迹坐标
    for i, T in enumerate(temperatures):
        # 使用Chebyshev近似公式 T <= 4000K
        if T <= 4000:
            u_bb[i] = (0.860117757 + 1.54118254e-4 * T + 1.28641212e-7 * T ** 2) / \
                      (1 + 8.42420235e-4 * T + 7.08145163e-7 * T ** 2)
            v_bb[i] = (0.317398726 + 4.22806245e-5 * T + 4.20481691e-8 * T ** 2) / \
                     (1 - 2.89741816e-5 * T + 1.61456053e-7 * T ** 2)

        # 4000K < T < 7000K 区间
        if T > 4000 and T < 7000:
            u_bb[i] = (
                    -4.607e7 / T ** 3 +
                    2.9678e6 / T ** 2 +
                    99.11 / T +
                    0.244063
            )
            v_bb[i] = -3 * u_bb[i] ** 2 + 2.87 * u_bb[i] - 0.275

        # T >= 7000K 区间
        if T >= 7000:
            u_bb[i] = (
                    -2.0064e7 / T ** 3 +
                    1.9018e6 / T ** 2 +
                    247.48 / T +
                    0.23704
            )
            v_bb[i] = -3 * u_bb[i] ** 2 + 2.87 * u_bb[i] - 0.275
    # 计算待测点到所有黑体点的距离
    distances = np.sqrt((u_bb - u_c) ** 2 + (v_bb - v_c) ** 2)

    # 找到最近的两个黑体点
    idx = np.argsort(distances)[:2]
    T_j, T_j1 = temperatures[idx]
    u_j, v_j = u_bb[idx[0]], v_bb[idx[0]]
    u_j1, v_j1 = u_bb[idx[1]], v_bb[idx[1]]

    # 确保T_j < T_j1
    if T_j > T_j1:
        T_j, T_j1 = T_j1, T_j
        u_j, u_j1 = u_j1, u_j
        v_j, v_j1 = v_j1, v_j

    # 计算直线AB的方程
    slope_AB = (v_j1 - v_j) / (u_j1 - u_j) if u_j1 != u_j else 1e10

    # 计算垂足E
    if abs(slope_AB) > 1e5:  # 垂直线
        u_e = u_j
        v_e = v_c
    else:
        # 垂线斜率
        slope_perp = -1 / slope_AB if abs(slope_AB) > 1e-5 else 1e10

        # 计算垂足
        u_e = (slope_AB * u_j - slope_perp * u_c + v_c - v_j) / (slope_AB - slope_perp)
        v_e = slope_perp * (u_e - u_c) + v_c

    # 计算垂足到A和B的距离
    d1 = np.sqrt((u_e - u_j) ** 2 + (v_e - v_j) ** 2)
    d2 = np.sqrt((u_e - u_j1) ** 2 + (v_e - v_j1) ** 2)

    # 微倒度插值
    mired_j = 1e6 / T_j
    mired_j1 = 1e6 / T_j1
    mired0 = mired_j + d1 / (d1 + d2) * (mired_j1 - mired_j)

    # 计算相关色温
    cct = 1e6 / mired0
    return cct


# 2. 黑体轨迹的Chebyshev法
def chebyshev_method(u_c, v_c):

    # 定义u(T)和v(T)函数
    def u_T(T):
        num = 0.860117757 + 1.54118254e-4 * T + 1.28641212e-7 * T ** 2
        den = 1 + 8.42420235e-4 * T + 7.08145163e-7 * T ** 2
        return num / den

    def v_T(T):
        num = 0.317398726 + 4.22806245e-5 * T + 4.20481691e-8 * T ** 2
        den = 1 - 2.89741816e-5 * T + 1.61456053e-7 * T ** 2
        return num / den

    # 定义导数函数（数值微分）
    def du_dT(T, h=1.0):
        return (u_T(T + h) - u_T(T - h)) / (2 * h)

    def dv_dT(T, h=1.0):
        return (v_T(T + h) - v_T(T - h)) / (2 * h)

    # 定义方程 F(T) = 0
    def equation(T):
        u_t = u_T(T)
        v_t = v_T(T)
        du = du_dT(T)
        dv = dv_dT(T)
        return du * (u_t - u_c) + dv * (v_t - v_c)

    # 求解方程 (使用1000K-15000K范围)
    T_guess = 3000  # 初始猜测值
    cct = fsolve(equation, T_guess, xtol=0.1)[0]

    # 确保在合理范围内
    cct = np.clip(cct, 1000, 15000)
    return cct


# 3. 模拟黑体轨迹弧线法
def arc_simulating_method(u_c, v_c):
    # 定义参考点Q (根据论文表1)
    Q_low = np.array([0.328151, 0.1333451])  # 用于低色温范围
    Q_high = np.array([0.2861884, 0.246725])  # 用于高色温范围

    # 计算到两个参考点的向量
    vec_to_Q_low = np.array([u_c - Q_low[0], v_c - Q_low[1]])
    vec_to_Q_high = np.array([u_c - Q_high[0], v_c - Q_high[1]])

    # 计算负u轴方向向量
    neg_u_axis = np.array([-1, 0])

    # 计算角度θ (单位：度)
    def calc_angle(vec):
        dot_product = np.dot(vec, neg_u_axis)
        norm_vec = np.linalg.norm(vec)
        norm_neg_u = np.linalg.norm(neg_u_axis)
        cos_theta = dot_product / (norm_vec * norm_neg_u)
        theta_rad = np.arccos(cos_theta)
        return np.degrees(theta_rad)

    theta_low = calc_angle(vec_to_Q_low)
    theta_high = calc_angle(vec_to_Q_high)

    # 计算参数A和B (根据论文表1的多项式)
    # 低色温范围参数
    A_low = (24476 - 1690.96 * theta_low + 44.7172 * theta_low ** 2 -
             0.57152 * theta_low ** 3 + 3.5853e-3 * theta_low ** 4 -
             8.89785e-6 * theta_low ** 5)

    B_low = (-91627.3 + 6372.45 * theta_low - 169.335 * theta_low ** 2 +
             2.18036 * theta_low ** 3 - 0.0137504 * theta_low ** 4 +
             3.4292e-5 * theta_low ** 5)

    # 高色温范围参数
    A_high = (4.437 + 2.84 * theta_high + 0.022643 * theta_high ** 2 +
              4.039e-4 * theta_high ** 3 - 2.859e-6 * theta_high ** 4 -
              3.799e-6 * theta_high ** 5)

    B_high = (-746.3 + 71.16 * theta_high - 2.5412 * theta_high ** 2 +
              0.0474 * theta_high ** 3 - 5.128e-4 * theta_high ** 4 +
              2.604e-6 * theta_high ** 5)

    # 计算到两个参考点的距离
    d_low = np.linalg.norm(vec_to_Q_low)
    d_high = np.linalg.norm(vec_to_Q_high)

    # 计算微倒度
    mired_low = A_low + B_low * d_low
    mired_high = A_high + B_high * d_high

    # 计算色温 (单位K)
    cct_low = 1e6 / mired_low
    cct_high = 1e6 / mired_high

    # 根据色温范围选择合适的结果
    # 论文中方法适用于1667K-25000K (40-600微倒度)
    # 如果两个结果都在合理范围内，取平均值
    if 1667 <= cct_low <= 25000 and 1667 <= cct_high <= 25000:
        return (cct_low + cct_high) / 2
    elif 1667 <= cct_low <= 25000:
        return cct_low
    elif 1667 <= cct_high <= 25000:
        return cct_high
    else:
        # 如果都不在范围内，取平均值
        return (cct_low + cct_high) / 2


# 4. McCamy近似公式法
def mccamy_approximation(x, y):
    """McCamy近似公式法计算相关色温"""
    n = (x - 0.3320) / (y - 0.1858)
    cct = -437 * n ** 3 + 3601 * n ** 2 - 6861 * n + 5514.31
    return cct


# 计算四种方法的CCT
cct_triangle = triangle_perpendicular_interpolation(u, v)
cct_chebyshev = chebyshev_method(u, v)
cct_arc = arc_simulating_method(u, v)
cct_mccamy = mccamy_approximation(x, y)

# 打印结果
print("相关色温(CCT)计算结果:")
print(f"1. 三角垂足插值法: {cct_triangle:.2f} K")
print(f"2. 黑体轨迹的Chebyshev法: {cct_chebyshev:.2f} K")
print(f"3. 模拟黑体轨迹弧线法: {cct_arc:.2f} K")
print(f"4. McCamy近似公式法: {cct_mccamy:.2f} K")
print("=" * 50)

# 结果分析
methods = ['三角垂足插值', 'Chebyshev', '弧线模拟', 'McCamy']
cct_values = [cct_triangle, cct_chebyshev, cct_arc, cct_mccamy]

# # 绘制结果比较图
# plt.figure(figsize=(10, 6))
# plt.bar(methods, cct_values, color=['blue', 'green', 'orange', 'red'])
# plt.title('四种相关色温计算方法结果比较')
# plt.ylabel('相关色温 (K)')
# plt.grid(axis='y', linestyle='--', alpha=0.7)
#
# # 在柱子上方显示数值
# for i, v in enumerate(cct_values):
#     plt.text(i, v + 50, f'{v:.1f} K', ha='center', fontweight='bold')
#
# plt.tight_layout()
# plt.savefig('cct_comparison.png')
# plt.show()