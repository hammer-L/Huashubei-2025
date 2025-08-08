import numpy as np
import pandas as pd
import os
import openpyxl

mode1 = "Lucas et al., 2014"
mode2 = "CIE S 026"

def get_mel_DER_Lucas(numpy_data):

    input_excel = "Data/CIE S 026 alpha-opic Toolbox Lucas.xlsx"

    output_file = "Data/CIE S 026 alpha-opic Toolbox_modified.xlsx"
    
    # 加载Excel文件
    wb = openpyxl.load_workbook(input_excel, data_only=True)
    
    # 获取Inputs工作表
    inputs_sheet = wb['Inputs']
    
    # 将numpy数据填充到C24到C424单元格
    for i, value in enumerate(numpy_data):
        inputs_sheet[f'C{24 + i}'] = float(value)
    
    # 保存修改后的Excel文件
    wb.save(output_file)
    
    # 读取Advanced Outputs工作表中的J40值
    advanced_outputs_sheet = wb['Advanced Outputs']
    j40_value = advanced_outputs_sheet['J40'].value
    
    return j40_value

def get_mel_DER_CIE(numpy_data):

    input_excel = "Data/CIE S 026 alpha-opic Toolbox CIE.xlsx"

    output_file = "Data/CIE S 026 alpha-opic Toolbox_modified.xlsx"
    
    # 加载Excel文件
    wb = openpyxl.load_workbook(input_excel, data_only=True)
    
    # 获取Inputs工作表
    inputs_sheet = wb['Inputs']
    
    # 将numpy数据填充到C24到C424单元格
    for i, value in enumerate(numpy_data):
        inputs_sheet[f'C{24 + i}'] = float(value)
    
    # 保存修改后的Excel文件
    wb.save(output_file)
    
    # 读取Advanced Outputs工作表中的J40值
    advanced_outputs_sheet = wb['Advanced Outputs']
    j40_value = advanced_outputs_sheet['J40'].value
    
    return j40_value

"""
test

os.chdir(r"/home/workshop/Huashubei-2025")

df_p1 = pd.read_excel('Data/P1_processed_data.xlsx')
light_intensity = df_p1["光强"].to_numpy()
print (len(light_intensity))

result = get_mel_DER(light_intensity, mode1)
print(f"从Advanced Outputs的J40读取的值是: {result}")
"""