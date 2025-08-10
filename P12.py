import numpy as np
import pandas as pd
import colour
from colour import (
    SpectralDistribution, SpectralShape, MSDS_CMFS
)
import math
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

WMIN, WMAX, WSTEP = 380, 780, 1
shape = SpectralShape(WMIN, WMAX, WSTEP)
wls = np.arange(WMIN, WMAX + 1, WSTEP)

L_A = 64.0
Y_b = 20.0

def read_reflectances(file_path):
    """
    读取反射率数据，返回 (n_samples, n_wls) 数组和原始波长数组。
    自动判断数据格式：
    - 若csv文件首列名带'nm'，则认为是每列为波长，行是样品
    - 否则认为是第一列波长，后面列为样品反射率
    """
    try:
        df = pd.read_csv(file_path)
        if 'nm' in df.columns[0]:
            orig_wls = [float(c.replace('nm','')) for c in df.columns]
            refl_array = df.values  # (samples, wls)
            return refl_array, np.array(orig_wls)
    except Exception:
        pass

    df = pd.read_csv(file_path, header=None)
    wavelengths = df.iloc[:, 0].values
    refl_matrix = df.iloc[:, 1:].values
    refl_array = refl_matrix.T
    return refl_array, wavelengths

def read_spd_csv(file_path):
    """读取两列格式的SPD CSV，返回SpectralDistribution"""
    df = pd.read_csv(file_path)
    return SpectralDistribution(dict(zip(df.iloc[:, 0], df.iloc[:, 1])))

def read_spd_excel(file_path):
    """读取Excel格式SPD，要求含列'wavelength'和'power'"""
    df = pd.read_excel(file_path)
    return SpectralDistribution(dict(zip(df['wavelength'], df['power'])))

def interpolate_reflectances(refl_array, orig_wls, tgt_wls):
    refl_interp = []
    for refl in refl_array:
        f = interp1d(orig_wls, refl, kind='linear', bounds_error=False, fill_value=(refl[0], refl[-1]))
        refl_interp.append(f(tgt_wls))
    return np.vstack(refl_interp)

def compute_tm30(reflectance_file, test_spd_file, ref_spd_file, spd_excel=False):
    """
    计算TM-30的Rf和Rg

    参数:
    - reflectance_file: 反射率文件路径
    - test_spd_file: 测试光谱功率分布文件路径
    - ref_spd_file: 参考光谱功率分布文件路径（一般是D65）
    - spd_excel: 测试SPD是否为Excel格式（默认为False，即CSV）

    返回:
    - dict 包含 mean_dE, Rf_prime, Rf, Rg
    """
    refl_array, orig_wls = read_reflectances(reflectance_file)

    if spd_excel:
        sd_test = read_spd_excel(test_spd_file)
    else:
        sd_test = read_spd_csv(test_spd_file)

    sd_ref = read_spd_csv(ref_spd_file)

    refl_array = interpolate_reflectances(refl_array, orig_wls, wls)
    sd_test = sd_test.interpolate(shape)
    sd_ref = sd_ref.interpolate(shape)

    cmfs_10 = MSDS_CMFS['CIE 1964 10 Degree Standard Observer'].interpolate(shape)
    x10 = np.array([cmfs_10[wl][0] for wl in wls])
    y10 = np.array([cmfs_10[wl][1] for wl in wls])
    z10 = np.array([cmfs_10[wl][2] for wl in wls])

    st_vals = np.array([sd_test[wl] for wl in wls])
    sr_vals = np.array([sd_ref[wl] for wl in wls])
    k_t = 100.0 / np.trapezoid(st_vals * y10, wls)
    k_r = 100.0 / np.trapezoid(sr_vals * y10, wls)

    def compute_XYZ(refl_vals, sd_vals, k):
        S = sd_vals * refl_vals
        X = k * np.trapezoid(S * x10, wls)
        Y = k * np.trapezoid(S * y10, wls)
        Z = k * np.trapezoid(S * z10, wls)
        return np.array([X, Y, Z])

    n_samples = refl_array.shape[0]
    XYZ_test = np.zeros((n_samples, 3))
    XYZ_ref = np.zeros((n_samples, 3))

    for i in range(n_samples):
        refl = refl_array[i]
        XYZ_test[i] = compute_XYZ(refl, st_vals, k_t)
        XYZ_ref[i] = compute_XYZ(refl, sr_vals, k_r)

    one_refl = np.ones_like(wls)
    XYZ_white_test = compute_XYZ(one_refl, st_vals, k_t)
    XYZ_white_ref = compute_XYZ(one_refl, sr_vals, k_r)

    Jab_test = colour.XYZ_to_CAM02UCS(XYZ_test / 100.0, XYZ_w=XYZ_white_test / 100.0, L_A=L_A, Y_b=Y_b)
    Jab_ref = colour.XYZ_to_CAM02UCS(XYZ_ref / 100.0, XYZ_w=XYZ_white_ref / 100.0, L_A=L_A, Y_b=Y_b)

    delta_E = np.linalg.norm(Jab_test - Jab_ref, axis=1)
    mean_dE = np.mean(delta_E)

    Rf_prime = 100.0 - 6.73 * mean_dE
    Rf = 10.0 * math.log(math.exp(Rf_prime / 10.0) + 1.0)
    Rf = max(Rf, 0.0)

    a_test = Jab_test[:, 1]
    b_test = Jab_test[:, 2]
    a_ref = Jab_ref[:, 1]
    b_ref = Jab_ref[:, 2]

    hues = (np.degrees(np.arctan2(b_ref, a_ref)) + 360) % 360
    bin_width = 360.0 / 16.0
    bin_indices = (hues // bin_width).astype(int)

    mean_pts_test = []
    mean_pts_ref = []
    for bin_idx in range(16):
        mask = (bin_indices == bin_idx)
        if np.any(mask):
            mean_pts_test.append((np.mean(a_test[mask]), np.mean(b_test[mask])))
            mean_pts_ref.append((np.mean(a_ref[mask]), np.mean(b_ref[mask])))
        else:
            mean_pts_test.append((0.0, 0.0))
            mean_pts_ref.append((0.0, 0.0))

    poly_test = np.array(mean_pts_test)
    poly_ref = np.array(mean_pts_ref)

    def polygon_area(points):
        x = points[:, 0]
        y = points[:, 1]
        return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

    area_test = polygon_area(poly_test)
    area_ref = polygon_area(poly_ref)

    Rg = 100.0 * (area_test / area_ref) if area_ref > 0 else float('nan')

    return {
        'mean_dE': mean_dE,
        'Rf_prime': Rf_prime,
        'Rf': Rf,
        'Rg': Rg,
        'Jab_test': Jab_test,
        'Jab_ref': Jab_ref
    }

def visualize_rf_rg(Jab_test, Jab_ref):
    # 计算delta_E（色差）
    delta_E = np.linalg.norm(Jab_test - Jab_ref, axis=1)

    # 计算色相bin（16个bin）
    a_ref = Jab_ref[:, 1]
    b_ref = Jab_ref[:, 2]
    hues = (np.degrees(np.arctan2(b_ref, a_ref)) + 360) % 360
    bin_width = 360.0 / 16.0
    bin_indices = (hues // bin_width).astype(int)

    # 计算每bin平均a,b（多边形顶点）
    mean_pts_test = []
    mean_pts_ref = []
    for bin_idx in range(16):
        mask = (bin_indices == bin_idx)
        if np.any(mask):
            mean_pts_test.append((np.mean(Jab_test[mask,1]), np.mean(Jab_test[mask,2])))
            mean_pts_ref.append((np.mean(Jab_ref[mask,1]), np.mean(Jab_ref[mask,2])))
        else:
            mean_pts_test.append((0.0, 0.0))
            mean_pts_ref.append((0.0, 0.0))

    poly_test = np.array(mean_pts_test)
    poly_ref = np.array(mean_pts_ref)

    # 绘制色差 delta_E 散点图（Rf体现）
    plt.figure(figsize=(12,5))

    plt.subplot(1,2,1)
    plt.scatter(range(len(delta_E)), delta_E, c='tab:red', label='Color Difference (ΔE)')
    plt.xlabel('Sample Index')
    plt.ylabel('ΔE (CAM02-UCS)')
    plt.title('Color Difference per Sample (Reflects Rf)')
    plt.grid(True)
    plt.legend()

    # 绘制色度多边形图（Rg体现）
    plt.subplot(1,2,2)
    # 闭合多边形
    poly_test_closed = np.vstack([poly_test, poly_test[0]])
    poly_ref_closed = np.vstack([poly_ref, poly_ref[0]])

    plt.plot(poly_ref_closed[:,0], poly_ref_closed[:,1], 'b-', label='Reference SPD')
    plt.plot(poly_test_closed[:,0], poly_test_closed[:,1], 'r-', label='Test SPD')
    plt.scatter(poly_ref[:,0], poly_ref[:,1], c='blue')
    plt.scatter(poly_test[:,0], poly_test[:,1], c='red')
    plt.xlabel('a')
    plt.ylabel('b')
    plt.title('Chromaticity Polygon (Reflects Rg)')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')

    plt.tight_layout()
    plt.show()

def get_Rf_Rg(test_spd_df,
              reflectance_file='Data/IESTM30_15_Sspds.dat',
              ref_spd_file='Data/d65.csv'):
    """
    计算给定测试光源SPD（DataFrame格式）的 Rf 和 Rg。

    参数:
    - test_spd_df: pd.DataFrame，包含两列，第一列波长，第二列功率（或辐射度）
    - reflectance_file: 反射率文件路径（默认IESTM30标准测试样品）
    - ref_spd_file: 参考光源SPD文件路径（默认D65）

    返回:
    - (Rf, Rg) 两个float值
    """

    # 把测试SPD DataFrame转换成SpectralDistribution
    sd_test = SpectralDistribution(dict(zip(test_spd_df.iloc[:,0], test_spd_df.iloc[:,1])))

    # 读取反射率和参考SPD，调用已有函数的部分代码
    refl_array, orig_wls = read_reflectances(reflectance_file)
    sd_ref = read_spd_csv(ref_spd_file)

    # 插值处理
    refl_array = interpolate_reflectances(refl_array, orig_wls, wls)
    sd_test = sd_test.interpolate(shape)
    sd_ref = sd_ref.interpolate(shape)

    # CIE 1964 10度视标准观察者
    cmfs_10 = MSDS_CMFS['CIE 1964 10 Degree Standard Observer'].interpolate(shape)
    x10 = np.array([cmfs_10[wl][0] for wl in wls])
    y10 = np.array([cmfs_10[wl][1] for wl in wls])
    z10 = np.array([cmfs_10[wl][2] for wl in wls])

    st_vals = np.array([sd_test[wl] for wl in wls])
    sr_vals = np.array([sd_ref[wl] for wl in wls])
    k_t = 100.0 / np.trapezoid(st_vals * y10, wls)
    k_r = 100.0 / np.trapezoid(sr_vals * y10, wls)

    def compute_XYZ(refl_vals, sd_vals, k):
        S = sd_vals * refl_vals
        X = k * np.trapezoid(S * x10, wls)
        Y = k * np.trapezoid(S * y10, wls)
        Z = k * np.trapezoid(S * z10, wls)
        return np.array([X, Y, Z])

    n_samples = refl_array.shape[0]
    XYZ_test = np.zeros((n_samples, 3))
    XYZ_ref = np.zeros((n_samples, 3))

    for i in range(n_samples):
        refl = refl_array[i]
        XYZ_test[i] = compute_XYZ(refl, st_vals, k_t)
        XYZ_ref[i] = compute_XYZ(refl, sr_vals, k_r)

    one_refl = np.ones_like(wls)
    XYZ_white_test = compute_XYZ(one_refl, st_vals, k_t)
    XYZ_white_ref = compute_XYZ(one_refl, sr_vals, k_r)

    Jab_test = colour.XYZ_to_CAM02UCS(XYZ_test / 100.0, XYZ_w=XYZ_white_test / 100.0, L_A=L_A, Y_b=Y_b)
    Jab_ref = colour.XYZ_to_CAM02UCS(XYZ_ref / 100.0, XYZ_w=XYZ_white_ref / 100.0, L_A=L_A, Y_b=Y_b)

    delta_E = np.linalg.norm(Jab_test - Jab_ref, axis=1)
    mean_dE = np.mean(delta_E)

    Rf_prime = 100.0 - 6.73 * mean_dE
    Rf = 10.0 * math.log(math.exp(Rf_prime / 10.0) + 1.0)
    Rf = max(Rf, 0.0)

    a_test = Jab_test[:, 1]
    b_test = Jab_test[:, 2]
    a_ref = Jab_ref[:, 1]
    b_ref = Jab_ref[:, 2]

    hues = (np.degrees(np.arctan2(b_ref, a_ref)) + 360) % 360
    bin_width = 360.0 / 16.0
    bin_indices = (hues // bin_width).astype(int)

    mean_pts_test = []
    mean_pts_ref = []
    for bin_idx in range(16):
        mask = (bin_indices == bin_idx)
        if np.any(mask):
            mean_pts_test.append((np.mean(a_test[mask]), np.mean(b_test[mask])))
            mean_pts_ref.append((np.mean(a_ref[mask]), np.mean(b_ref[mask])))
        else:
            mean_pts_test.append((0.0, 0.0))
            mean_pts_ref.append((0.0, 0.0))

    poly_test = np.array(mean_pts_test)
    poly_ref = np.array(mean_pts_ref)

    def polygon_area(points):
        x = points[:, 0]
        y = points[:, 1]
        return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

    area_test = polygon_area(poly_test)
    area_ref = polygon_area(poly_ref)

    Rg = 100.0 * (area_test / area_ref) if area_ref > 0 else float('nan')

    return Rf, Rg

# 'Data/IESTM30_15_Sspds.dat' 'Data/TM30_CES99.csv'
results = compute_tm30(
    reflectance_file='Data/IESTM30_15_Sspds.dat',
    test_spd_file='Data/P1_processed_data.xlsx',
    ref_spd_file='Data/d65.csv',
    spd_excel=True
)

visualize_rf_rg(results['Jab_test'], results['Jab_ref'])
print(results['Rf'],results['Rg'])