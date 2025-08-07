import pandas as pd

# 读取Excel文件
file_path = 'processed_data.xlsx'
df = pd.read_excel(file_path, sheet_name='Sheet1')

# 提取波长和光强数据为NumPy数组
wavelengths = df['波长(nm)'].to_numpy()
intensity = df['光谱功率(W/nm)'].to_numpy()
