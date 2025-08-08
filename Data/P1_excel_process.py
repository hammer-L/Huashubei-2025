import pandas as pd
import os
os.chdir(r"/home/workshop/Huashubei-2025")
import re  # 用于正则表达式处理

# 读取Excel文件
df = pd.read_excel('ref.xlsx', sheet_name='Problem 2_LED_SPD', header=0)  # header=0 表示第一行是列标题

# 处理波长列：提取括号前的数字
def extract_wavelength(value):
    if isinstance(value, str):
        match = re.search(r'^(\d+\.?\d*)', value)
        if match:
            return float(match.group(1))
    return value

# 应用处理函数到波长列（跳过标题行）
df['波长'] = df['波长'].apply(extract_wavelength)
df.to_excel('processed_data.xlsx', index=False)

