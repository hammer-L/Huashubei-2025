import pandas as pd
import os
import re  # 用于正则表达式处理
os.chdir(r"C:\Users\MrFoxxxx\Desktop\MathematicalModel\Huashubei-2025")

# 读取Excel文件
df = pd.read_excel('ref.xlsx', sheet_name='Problem 1', header=0)

# 处理波长
def extract_wavelength(value):
    if isinstance(value, str):
        match = re.search(r'^(\d+\.?\d*)', value)
        if match:
            return float(match.group(1))
    return value

df['波长'] = df['波长'].apply(extract_wavelength)
df['光强'] = pd.to_numeric(df['光强'], errors='coerce')

# 重命名
df = df.rename(columns={
    '波长': '波长(nm)',
    '光强': '光谱功率(W/nm)'
})

df.to_excel('P1_processed_data.xlsx', index=False)

