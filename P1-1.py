import pandas as pd
import numpy as np
import Duv
import CCT
import matplotlib.pyplot as plt
import matplotlib
import os

matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei']   # 设置中文字体
matplotlib.rcParams['axes.unicode_minus'] = False     # 解决负号显示问题

# 获取当前.py文件所在目录，创建output文件夹
current_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(current_dir, 'output')
os.makedirs(output_dir, exist_ok=True)

# 加载光谱功率分布数据
spd = pd.read_excel('P1_processed_data.xlsx', sheet_name=0, header=0, names=['wavelength', 'power'])
results = CCT.calculate_color_parameters(spd, cmf_file='ciexyz31.csv', norm_Y=100)

# 打印基本结果
print("=" * 50)
print("计算结果:")
print("=" * 50)
print(f"归一化XYZ值:")
print(f"  X = {results['XYZ_norm'][0]:.6f}")
print(f"  Y = {results['XYZ_norm'][1]:.6f}")
print(f"  Z = {results['XYZ_norm'][2]:.6f}")
print(f"色品坐标 (xy):")
print(f"  x = {results['xy'][0]:.6f}")
print(f"  y = {results['xy'][1]:.6f}")
print(f"CIE 1960 UCS坐标 (uv):")
print(f"  u = {results['uv'][0]:.6f}")
print(f"  v = {results['uv'][1]:.6f}")
print("=" * 50)

# 计算四种方法的CCT
cct_triangle = CCT.triangle_perpendicular_interpolation(results['u'],results['v'])
cct_chebyshev = CCT.chebyshev_method(results['u'],results['v'])
cct_arc = CCT.arc_simulating_method(results['u'],results['v'])
cct_mccamy = CCT.mccamy_approximation(results['x'],results['y'])

# 打印结果
print("相关色温(CCT)计算结果:")
print(f"1. 三角垂足插值法: {cct_triangle:.2f} K")
print(f"2. 黑体轨迹的Chebyshev法: {cct_chebyshev:.2f} K")
print(f"3. 模拟黑体轨迹弧线法: {cct_arc:.2f} K")
print(f"4. McCamy近似公式法: {cct_mccamy:.2f} K")
print("=" * 50)

# 计算Duv
u, v = results['uv']
duv = Duv.calculate_duv(u, v)

# 打印Duv结果
print(f"Duv (色度偏差): {duv:.6f}")

# 解释Duv值
if abs(duv) < 0.0005:
    print("光源颜色非常接近黑体轨迹")
elif duv > 0:
    print("光源颜色在黑体轨迹上方（偏绿）")
else:
    print("光源颜色在黑体轨迹下方（偏紫）")

# 结果分析
methods = ['三角垂足插值', 'Chebyshev', '弧线模拟', 'McCamy']
cct_values = [cct_triangle, cct_chebyshev, cct_arc, cct_mccamy]

# ======== 可视化部分 ========

# 1. SPD 光谱曲线
plt.figure(figsize=(8, 5))
plt.plot(spd['wavelength'], spd['power'], color='blue')
plt.title('光谱功率分布 (SPD)')
plt.xlabel('波长 (nm)')
plt.ylabel('功率 (W/m²·sr·nm)')
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'SPD_curve.png'), dpi=300)
plt.show()

# 2. 四种方法 CCT 柱状图
plt.figure(figsize=(8, 5))
bars = plt.bar(methods, cct_values, color=['#4E79A7', '#F28E2B', '#E15759', '#76B7B2'])
plt.title('四种方法计算的相关色温 (CCT)')
plt.ylabel('色温 (K)')
plt.grid(axis='y', linestyle='--', alpha=0.7)
for bar, val in zip(bars, cct_values):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{val:.2f}",
             ha='center', va='bottom', fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'CCT_methods_comparison.png'), dpi=300)
plt.show()

# 3. CIE 1960 UCS 色度图

# 生成黑体轨迹
T_range = np.arange(1000, 25001, 500)
u_bb, v_bb = [], []
for T in T_range:
    bb_spd = CCT.generate_blackbody_spd(T)
    bb_params = CCT.calculate_color_parameters(bb_spd)
    u_bb.append(bb_params['u'])
    v_bb.append(bb_params['v'])

plt.figure(figsize=(7, 7))
plt.plot(u_bb, v_bb, '-k', label='黑体轨迹')
plt.scatter(results['u'], results['v'], color='red', label='测量点')
plt.title('CIE 1960 UCS 色度图')
plt.xlabel('u')
plt.ylabel('v')
plt.grid(True)
plt.axis('equal')

# 绘制 Duv 偏移方向箭头
plt.arrow(results['u'], results['v'], 0, duv,
          head_width=0.002, head_length=0.002, fc='green', ec='green', label='Duv 偏移')

plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'CIE1960_UCS.png'), dpi=300)
plt.show()

# 以三角垂足插值法为参考（无误差）
ref_name = methods[0]
ref_cct = cct_values[0]

# 需要分析的后三种方法索引
compare_methods = methods[1:]
compare_values = cct_values[1:]

# 计算误差
abs_errors = [abs(val - ref_cct) for val in compare_values]                    # 绝对误差 (K)
signed_errors = [val - ref_cct for val in compare_values]                      # 有向误差 (K)
rel_errors_pct = [100.0 * (val - ref_cct) / ref_cct for val in compare_values] # 相对误差百分比 (%)
abs_rel_pct = [100.0 * abs(err) / ref_cct for err in signed_errors]            # 绝对相对误差 (%)

def stats_from_errors(err_list):
    arr = np.array(err_list, dtype=float)
    bias = np.mean(arr)             # 平均偏差
    std = np.std(arr, ddof=0)       # 标准差
    rmse = np.sqrt(np.mean(arr**2))
    max_abs = np.max(np.abs(arr))
    return {'bias': bias, 'std': std, 'rmse': rmse, 'max_abs': max_abs}

stats_abs = stats_from_errors(signed_errors)
stats_rel = stats_from_errors(rel_errors_pct)

df_results = pd.DataFrame({
    'method': compare_methods,
    'cct_K': compare_values,
    'abs_error_K': abs_errors,
    'signed_error_K': signed_errors,
    'abs_rel_pct': abs_rel_pct,
    'signed_rel_pct': rel_errors_pct
})

print("="*60)
print("误差分析（参考：三角垂足插值法 —— 无误差基准）")
print(f"参考 CCT ({ref_name}): {ref_cct:.2f} K")
print(df_results.to_string(index=False, float_format='{:,.4f}'.format))
print("-"*60)
print("误差统计（绝对 K）:", {k: f"{v:.4f}" for k, v in stats_abs.items()})
print("误差统计（相对 %）:", {k: f"{v:.4f}" for k, v in stats_rel.items()})
print("="*60)

# ====== 可视化 1: 绝对误差柱状图（K） ======
plt.figure(figsize=(8,5))
bars = plt.bar(compare_methods, abs_errors)
plt.title('后三种方法相对于三角垂足插值法的绝对误差 (K)')
plt.ylabel('绝对误差 (K)')
plt.grid(axis='y', linestyle='--', alpha=0.6)
for bar, v in zip(bars, abs_errors):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f"{v:.2f}",
             ha='center', va='bottom', fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'CCT_abs_errors_K.png'), dpi=300)
plt.show()

# ====== 可视化 2: 相对误差百分比柱状图 (%) ======
plt.figure(figsize=(8,5))
bars2 = plt.bar(compare_methods, rel_errors_pct)
plt.title('后三种方法相对于三角垂足插值法的相对误差 (%)')
plt.ylabel('相对误差 (%)')
plt.grid(axis='y', linestyle='--', alpha=0.6)
for bar, v in zip(bars2, rel_errors_pct):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f"{v:.3f}%",
             ha='center', va='bottom', fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'CCT_rel_errors_pct.png'), dpi=300)
plt.show()
