from scipy.optimize import nnls
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os

matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei']   # 设置中文字体
matplotlib.rcParams['axes.unicode_minus'] = False     # 解决负号显示问题

# ================= 数据读取 =================
led_df = pd.read_excel('Data/ref.xlsx', sheet_name='Problem 2_LED_SPD')
led_df['wavelength'] = led_df['波长'].str.extract(r'(\d+)').astype(int)
led_df = led_df.set_index('wavelength').drop(columns=['波长'], errors='ignore')
led_df = led_df.reindex(range(380, 781), fill_value=0).astype(float)
led_matrix = led_df[['Blue', 'Green', 'Red', 'Warm White', 'Cold White']].values

sun_df = pd.read_excel('Data/ref.xlsx', sheet_name='Problem 3 SUN_SPD')
sun_df['wavelength'] = sun_df['波长'].str.extract(r'(\d+)').astype(int)
sun_df = sun_df.set_index('wavelength').drop(columns=['波长'], errors='ignore')
sun_df = sun_df.reindex(range(380, 781), fill_value=0).astype(float)

available_times_str = [str(t) for t in sun_df.columns]

# ================= 工具函数 =================
def find_closest_time(target_time, available_times_str):
    for t_str in available_times_str:
        if target_time in t_str:
            return t_str
    return None

# 拟合算法函数
def shape_then_value_fit(led_matrix, target, alpha=0.5):

    def linear_scale(y_fit, y_target):
        # 对拟合结果做线性缩放，使其最大最小值与目标匹配
        y_fit_min, y_fit_max = y_fit.min(), y_fit.max()
        y_tar_min, y_tar_max = y_target.min(), y_target.max()
        if y_fit_max - y_fit_min < 1e-12:
            return np.full_like(y_fit, y_tar_min)
        scale = (y_tar_max - y_tar_min) / (y_fit_max - y_fit_min)
        offset = y_tar_min - y_fit_min * scale
        return y_fit * scale + offset

    # 第一步：用 nnls 拟合数值部分，得到初始权重
    w_value, _ = nnls(led_matrix, target)

    # 计算拟合的光谱形状与目标的相关系数
    y_fit_raw = led_matrix @ w_value
    y_fit_scaled = linear_scale(y_fit_raw, target)

    # 计算相关系数
    def corr_coef(y_true, y_pred):
        eps = 1e-12
        y_true_mean = np.mean(y_true)
        y_pred_mean = np.mean(y_pred)
        numerator = np.sum((y_true - y_true_mean) * (y_pred - y_pred_mean))
        denominator = np.sqrt(np.sum((y_true - y_true_mean)**2) * np.sum((y_pred - y_pred_mean)**2)) + eps
        return numerator / denominator

    target_scaled = linear_scale(target, y_fit_raw)  # 目标缩放到拟合光谱范围
    w_shape, _ = nnls(led_matrix, target_scaled)

    # 最终权重按alpha加权
    w = alpha * w_shape + (1 - alpha) * w_value

    return w

# 生成权重随时间变化表
def generate_weights_for_times(times_to_generate, led_matrix, sun_df, fit_func):
    weights_dict = {}
    available_times_str = [str(t) for t in sun_df.columns]

    for time_str in times_to_generate:
        t_str = find_closest_time(time_str, available_times_str)
        if t_str is None:
            print(f"时间 {time_str} 不存在数据，跳过")
            continue

        idx = available_times_str.index(t_str)
        target = sun_df.iloc[:, idx].values
        weights = fit_func(led_matrix, target)
        weights_dict[time_str] = weights

    return weights_dict

# 生成时间序列
def generate_time_series_no_datetime(start='5:30', end='17:30', step_hours=1):
    times = []
    start_h, start_m = map(int, start.split(':'))
    end_h, end_m = map(int, end.split(':'))

    current_h, current_m = start_h, start_m
    while (current_h < end_h) or (current_h == end_h and current_m <= end_m):
        times.append(f"{current_h}:{current_m:02d}")
        current_h += step_hours
    return times

# 不同alpha三个时段的拟合光谱与太阳光谱对比图
def plot_spectrum_vs_alpha_subplots(times, alphas, led_matrix, sun_df, fit_func, save_dir="Output"):
    os.makedirs(save_dir, exist_ok=True)
    available_times_str = [str(t) for t in sun_df.columns]
    wav = np.arange(380, 781)
    led_colors = ['blue', 'green', 'red', 'orange', '#8ba7b8']

    fig, axs = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    fig.suptitle('不同 Alpha 值对光谱拟合的影响', fontsize=16)

    for j, time_str in enumerate(times):
        t_str = find_closest_time(time_str, available_times_str)
        if t_str is None:
            print(f"时间 {time_str} 不存在数据，跳过")
            continue

        idx = available_times_str.index(t_str)
        target = sun_df.iloc[:, idx].values

        ax = axs[j]
        ax.plot(wav, target, label='自然光谱', color='black', linewidth=2)

        for i, alpha in enumerate(alphas):
            weights = fit_func(led_matrix, target, alpha=alpha)
            fitted_spectrum = led_matrix @ weights
            ax.plot(wav, fitted_spectrum, linestyle='-',
                    label=f'alpha={alpha:.2f}', color=led_colors[i % len(led_colors)])

        ax.set_title(f"{time_str} 时段")
        ax.set_xlabel("波长 (nm)")
        if j == 0:
            ax.set_ylabel("光强")
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')

    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    file_path = os.path.join(save_dir, "alpha_vs_spectrum.pdf")
    plt.savefig(file_path, format="pdf")
    print(f"已保存 PDF：{file_path}")
    plt.show()

# 确定alpha三个时段的拟合光谱与太阳光谱对比图
def plot_fits_for_selected_times_subplots(times, led_matrix, sun_df, fit_func, save_dir="Output"):
    os.makedirs(save_dir, exist_ok=True)
    available_times_str = [str(t) for t in sun_df.columns]
    wav = np.arange(380, 781)

    fig, axs = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    fig.suptitle('自然光与拟合光谱对比', fontsize=16)

    for j, time_str in enumerate(times):
        t_str = find_closest_time(time_str, available_times_str)
        if t_str is None:
            print(f"时间 {time_str} 不存在数据，跳过")
            continue

        idx = available_times_str.index(t_str)
        target = sun_df.iloc[:, idx].values
        weights = fit_func(led_matrix, target)
        fitted_spectrum = led_matrix @ weights

        ax = axs[j]
        ax.plot(wav, target, label='自然光', color='black', linewidth=2)
        ax.plot(wav, fitted_spectrum, label='拟合光线', linestyle='--', color='red')
        ax.set_title(f"{time_str} 时段")
        ax.set_xlabel("波长 (nm)")
        if j == 0:
            ax.set_ylabel("光强")
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    file_path = os.path.join(save_dir, "fitted_vs_sun.pdf")
    plt.savefig(file_path, format="pdf")
    print(f"已保存 PDF：{file_path}")
    plt.show()

# 生成权重随时间变化图
def plot_led_weights_over_time(weights_results, times_to_plot, save_dir="Output"):
    os.makedirs(save_dir, exist_ok=True)

    def time_str_to_float(t):
        h, m = map(int, t.split(':'))
        return h + m / 60

    times_numeric = [time_str_to_float(t) for t in times_to_plot]
    weights_array = np.array([weights_results[t] for t in times_to_plot])  # (时间数, 5)

    plt.figure(figsize=(10,6))

    led_names = ['Blue', 'Green', 'Red', 'Warm White', 'Cold White']
    led_colors = ['blue', 'green', 'red', 'orange', '#8ba7b8']

    for i in range(weights_array.shape[1]):
        plt.plot(times_numeric, weights_array[:, i], marker='o', label=led_names[i], color=led_colors[i])

    plt.xticks(times_numeric, times_to_plot, rotation=45)
    plt.xlabel('时间')
    plt.ylabel('权重')
    plt.title('LED五个通道权重随时间变化')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    file_path = os.path.join(save_dir, "led_weights_over_time.pdf")
    plt.savefig(file_path, format="pdf")
    print(f"已保存 PDF：{file_path}")
    plt.show()

# 生成权重随时间变化表
def save_led_weights_table(weights_results, times_to_plot, filename="Output\weights.xlsx"):

    # LED通道名称
    led_names = ['Blue', 'Green', 'Red', 'Warm White', 'Cold White']

    # 生成 DataFrame
    weights_array = np.array([weights_results[t] for t in times_to_plot])
    weights_df = pd.DataFrame(weights_array, index=times_to_plot, columns=led_names)

    # 保留小数
    weights_df = weights_df.round(4)

    # 打印表格
    print("LED通道权重表：")
    print(weights_df)

    # 保存到 Excel
    weights_df.to_excel(filename)
    print(f"权重表已保存到 {filename}")

    return weights_df

# ================= 主执行逻辑 =================
times_to_plot = generate_time_series_no_datetime('5:30', '17:30', 1)
selected_times = ['5:30', '12:30', '17:30']
weights_results = generate_weights_for_times(times_to_plot, led_matrix, sun_df, shape_then_value_fit)

# 生成权重随时间变化表并保存
save_led_weights_table(weights_results, times_to_plot)

# 生成权重随时间变化图
plot_led_weights_over_time(weights_results, times_to_plot)

# 三个时段的拟合光谱与太阳光谱对比图
alphas = np.linspace(0, 1, 5)
plot_spectrum_vs_alpha_subplots(selected_times, alphas, led_matrix, sun_df, shape_then_value_fit)
plot_fits_for_selected_times_subplots(selected_times, led_matrix, sun_df, shape_then_value_fit)