import pandas as pd
import numpy as np
from scipy.optimize import minimize
# 读取Excel文件第二张表
df = pd.read_excel('ref.xlsx', sheet_name=1)
spd = pd.read_excel('Data/ref.xlsx', sheet_name=2, header=0, names=['wavelength', 'power'])
# 解析波长列，提取数字部分
df['wavelength'] = df['波长'].str.extract(r'(\d+)').astype(int)

# 取各通道SPD列
blue = df['Blue']
green = df['Green']
red = df['Red']
warm_white = df['Warm White']
cold_white = df['Cold White']

# 定义权重（这里举例）
w_blue = 1.0
w_green = 1.0
w_red = 1.0
w_WW = 1.0
w_CW = 1.0

# 计算合成SPD
df['SPD_total'] = (w_blue * blue + w_green * green + w_red * red + w_WW * warm_white + w_CW * cold_white)

# 最终结果df包含'wavelength'和'SPD_total'列
print(df[['wavelength', 'SPD_total']])