import numpy as np
import CCT

# 5. Duv计算
def calculate_duv(u, v, T_min=1000, T_max=25000, T_step=50):

    # 生成黑体轨迹温度点
    temperatures = np.arange(T_min, T_max + T_step, T_step)
    u_bb = np.zeros_like(temperatures, dtype=float)
    v_bb = np.zeros_like(temperatures, dtype=float)

    # 计算黑体轨迹坐标
    for i, T in enumerate(temperatures):
        # 生成黑体光谱
        bb_spd = CCT.generate_blackbody_spd(T)

        # 计算黑体色度参数
        bb_params = CCT.calculate_color_parameters(bb_spd)

        if bb_params:
            u_bb[i] = bb_params['uv'][0]
            v_bb[i] = bb_params['uv'][1]
        else:
            # 如果计算失败，使用线性插值填充
            if i > 0:
                u_bb[i] = u_bb[i - 1]
                v_bb[i] = v_bb[i - 1]
            else:
                u_bb[i] = 0
                v_bb[i] = 0

    # 计算测试点到所有黑体点的距离
    distances = np.sqrt((u_bb - u) ** 2 + (v_bb - v) ** 2)

    # 找到最近的黑体点
    min_idx = np.argmin(distances)
    u0 = u_bb[min_idx]
    v0 = v_bb[min_idx]

    # 计算Duv
    duv = np.sqrt((u - u0) ** 2 + (v - v0) ** 2)

    # 计算黑体轨迹在最近点的切线
    if min_idx > 0 and min_idx < len(temperatures) - 1:
        # 取前后两点计算切线
        u_prev = u_bb[min_idx - 1]
        v_prev = v_bb[min_idx - 1]
        u_next = u_bb[min_idx + 1]
        v_next = v_bb[min_idx + 1]

        # 切线向量 (从低温到高温)
        tangent_vec = np.array([u_next - u_prev, v_next - v_prev])

        # 法向量 (顺时针旋转90度)
        normal_vec = np.array([-tangent_vec[1], tangent_vec[0]])

        # 测试点相对于最近点的向量
        test_vec = np.array([u - u0, v - v0])

        # 计算点积确定方向
        dot_product = np.dot(normal_vec, test_vec)
        sign = 1 if dot_product > 0 else -1
    else:
        # 如果在端点，使用v坐标确定方向
        sign = 1 if v > v0 else -1

    duv *= sign

    return duv