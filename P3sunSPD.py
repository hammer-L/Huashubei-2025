
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei']   # 设置中文字体
matplotlib.rcParams['axes.unicode_minus'] = False     # 解决负号显示问题

# 读取太阳光谱数据
sun_df = pd.read_excel('Data/ref.xlsx', sheet_name='Problem 3 SUN_SPD')

# 预处理波长列，提取数字波长
sun_df['wavelength'] = sun_df['波长'].str.extract(r'(\d+)')
sun_df['wavelength'] = pd.to_numeric(sun_df['wavelength'], errors='coerce')
sun_df = sun_df.dropna(subset=['wavelength'])
sun_df['wavelength'] = sun_df['wavelength'].astype(int)

# 设波长为索引，删除原波长列
sun_df = sun_df.set_index('wavelength')
sun_df = sun_df.drop(columns=['波长'], errors='ignore')

# 补全波长范围，380-780nm，空缺补0
wav = np.arange(380, 781)
sun_df = sun_df.reindex(wav, fill_value=0)

# 确保数值类型
sun_df = sun_df.astype(float)

# 获取所有时间点列名
time_columns = sun_df.columns.tolist()

# 画所有时间点的原始光谱
plt.figure(figsize=(12, 8))
for time in time_columns:
    spd = sun_df[time].values
    plt.plot(wav, spd, label=time)

plt.title('各时段太阳光谱原始光强展示')
plt.xlabel('波长 (nm)')
plt.ylabel('光强 (mW/m²/nm)')
plt.legend(loc='upper right', fontsize='small', ncol=2)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
