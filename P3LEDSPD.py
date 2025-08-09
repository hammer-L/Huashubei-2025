import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# 设置中文和负号显示
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei']
matplotlib.rcParams['axes.unicode_minus'] = False

# 读取LED光谱数据
led_df = pd.read_excel('Data/ref.xlsx', sheet_name='Problem 2_LED_SPD')

# 预处理波长列
led_df['wavelength'] = led_df['波长'].str.extract(r'(\d+)')
led_df['wavelength'] = pd.to_numeric(led_df['wavelength'], errors='coerce')
led_df = led_df.dropna(subset=['wavelength'])
led_df['wavelength'] = led_df['wavelength'].astype(int)

# 设置波长为索引，删除原“波长”列
led_df = led_df.set_index('wavelength')
led_df = led_df.drop(columns=['波长'], errors='ignore')

# 补全波长，填充0
wav = np.arange(380, 781)
led_df = led_df.reindex(wav, fill_value=0)

# 取五个通道数据
channels = ['Blue', 'Green', 'Red', 'Warm White', 'Cold White']
led_df = led_df.astype(float)

# 颜色对应关系
led_colors = {
    'Blue': 'blue',
    'Green': 'green',
    'Red': 'red',
    'Warm White': 'orange',
    'Cold White': '#8ba7b8'
}

# 画图
plt.figure(figsize=(10,6))
for ch in channels:
    plt.plot(wav, led_df[ch].values, label=ch, color=led_colors[ch])
plt.title("LED五个通道光谱基底")
plt.xlabel("波长 (nm)")
plt.ylabel("光强 (mW/m²/nm)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
