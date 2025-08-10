import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
from typing import Tuple, Optional, Dict

sep_df = pd.read_excel('Data/P2_processed_data.xlsx')
print (sep_df.head())

def linear_combine_spd(spd_df: pd.DataFrame, w: np.ndarray) -> pd.DataFrame:
   
    # 确保顺序
    channels = ['Blue', 'Green', 'Red', 'Warm White', 'Cold White']
    assert len(w) == len(channels)
    
    # 取出各通道光谱矩阵 (波长数 × 通道数)
    channel_power = spd_df[channels].values 
    
    # 线性叠加：波长点数 × 通道数 乘以 通道数 × 1
    combined_power = np.dot(channel_power, w)

    #return
    result_df = pd.DataFrame({
        'wavelength': spd_df['wavelength'],
        'power': combined_power
    })
    
    return result_df

#weight init
w = np.array((1,1,1,1,1))

#Configeration
CCT_BOUNDS = (5500.0, 6500.0)   # [min, max] K
RG_BOUNDS = (95.0, 105.0)       # [min, max]
WEIGHT_BOUNDS = (0.0, 1.0)      # each weight in [0,1]
NUM_RANDOM_STARTS = 30          # multi-start次数

FEASIBILITY_TOL = 1e-5        # 约束容忍度（数值判断）
MAXITER = 300

from Interface.mel_DER import get_mel_DER_Lucas, get_mel_DER_CIE
from Interface.CCT import get_cct
from Interface.Duv import get_duv
from Interface.Rf_Rg import get_rf_rg

class MetricCache:
    def __init__(self, sep_df: pd.DataFrame):
        self.sep_df = sep_df
        self.cache = {}

    def key_of(self, w: np.ndarray, digits: int = 8):
        # 把权重归一化为 tuple，便于哈希
        return tuple(np.round(w, digits).tolist())
        
    def compute(self, w: np.ndarray):
        k = self.key_of(w)
        if k in self.cache:
            return self.cache[k]

        # 线性组合
        combined_spd_df = linear_combine_spd(self.sep_df, w)

        cct=0
        duv=0
        rf=0
        rg=0
        mel=0
        
        # 计算各项指标
        cct = float(get_cct(combined_spd_df))
        duv = float(get_duv(combined_spd_df))
        (rf, rg) = get_rf_rg(combined_spd_df)
        mel = float(get_mel_DER_CIE(combined_spd_df))
        """
        try:
            cct = float(get_cct(combined_spd_df))
            duv = float(get_duv(combined_spd_df))
            rf = float(get_rf(combined_spd_df))
            rg = float(get_rg(combined_spd_df))
            mel = float(get_mel_DER_Lucas(combined_spd_df))
        except Exception as e:
            print(f"计算指标时出错: {e}")
            cct, duv, rf, rg, mel = 0, 0, 0, 0, 0
        
        """    
      
        val = {'sep_df': self.sep_df,
               'cct': cct,
               'duv': duv,
               'rf': rf,
               'rg': rg,
               'mel': mel}
        
        self.cache[k] = val
        return val

def optimize_daytime_lighting(sep_df, initial_weights=None, max_iter= MAXITER, ftol=FEASIBILITY_TOL):
    
    # 创建指标缓存
    cache = MetricCache(sep_df)
    metrics = cache.compute(w)
    
    # 设置初始权重
    if initial_weights is None:
        initial_weights = np.array([0.1, 0.2, 0.1, 0.2, 0.4])

    # 定义目标函数
    def objective(w):
        rf = metrics['rf']
        return -rf 

    # 定义约束函数
    def cct_constraint(w):
        cct = metrics['cct']
        return min(cct - 5500, 6500 - cct)

    def rg_constraint(w):
        rg = metrics['rg']
        return min(rg - 95, 105 - rg)
    

    # 定义权重非负约束
    constraints = [
    {'type': 'ineq', 'fun': lambda w: w},            # w >= 0
    {'type': 'ineq', 'fun': cct_constraint},
    {'type': 'ineq', 'fun': rg_constraint},
]

    # 优化边界（权重范围）
    bounds = [(0, 1) for _ in range(5)]

    # 执行优化
    result = minimize(
        objective,
        initial_weights,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': max_iter, 'ftol': ftol}
    )
    if result.success:
        optimal_weights = result.x
        metrics = cache.compute_metrics(optimal_weights)    

        # 归一化权重（使总和为1）
        normalized_weights = optimal_weights / np.sum(optimal_weights)

        # print
        print("\n优化成功！")
        print(f"最优权重: {normalized_weights}")
        print(f"权重分配: 蓝光={normalized_weights[0]:.4f}, 绿光={normalized_weights[1]:.4f}, "
              f"红光={normalized_weights[2]:.4f}, 暖白={normalized_weights[3]:.4f}, "
              f"冷白={normalized_weights[4]:.4f}")
        print(f"相关色温 (CCT): {metrics['cct']:.1f} K")
        print(f"显色指数 (Rf): {metrics['rf']:.2f}")
        print(f"色域指数 (Rg): {metrics['rg']:.2f}")
        print(f"视黑素日光效率比 (mel-DER): {metrics['mel']:.3f}")

        # 返回结果
        return {
            'weights': normalized_weights,
            'cct': metrics['cct'],
            'rf': metrics['rf'],
            'rg': metrics['rg'],
            'mel': metrics['mel'],
            'spd': cache.compute(optimal_weights)['spd_df']
        }

    # 优化失败
    else:
        print("优化失败:", result.message)
        return None


def optimize_nighttime_lighting(sep_df, initial_weights=None, max_iter=MAXITER, ftol=FEASIBILITY_TOL):
    # 创建指标缓存
    cache = MetricCache(sep_df)
    
    # 设置初始权重
    if initial_weights is None:
        initial_weights = np.array([0.1, 0.2, 0.1, 0.2, 0.4])

    # 目标函数：最小化 mel-DER
    def objective(w):
        metrics = cache.compute(w)
        mel = metrics['mel']
        return mel  # 越小越好

    # 色温约束: 3000 ± 500 K
    def cct_constraint(w):
        metrics = cache.compute(w)
        cct = metrics['cct']
        return min(cct - 2500, 3500 - cct)  # 都 >= 0 时满足

    # 显色指数约束: Rf >= 80
    def rf_constraint(w):
        metrics = cache.compute(w)
        rf = metrics['rf']
        return rf - 80  # >= 0 满足约束

    # 权重非负约束
    constraints = [
        {'type': 'ineq', 'fun': lambda w: w},  # w >= 0
        {'type': 'ineq', 'fun': cct_constraint},
        {'type': 'ineq', 'fun': rf_constraint}
    ]

    # 权重范围限制
    bounds = [(0, 1) for _ in range(5)]

    # 执行优化
    result = minimize(
        objective,
        initial_weights,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': max_iter, 'ftol': ftol}
    )
    
    if result.success:
        optimal_weights = result.x
        metrics = cache.compute(optimal_weights)
        
        # 归一化权重
        normalized_weights = optimal_weights / np.sum(optimal_weights)

        print("\n优化成功！")
        print(f"最优权重: {normalized_weights}")
        print(f"权重分配: 蓝光={normalized_weights[0]:.4f}, 绿光={normalized_weights[1]:.4f}, "
              f"红光={normalized_weights[2]:.4f}, 暖白={normalized_weights[3]:.4f}, "
              f"冷白={normalized_weights[4]:.4f}")
        print(f"相关色温 (CCT): {metrics['cct']:.1f} K")
        print(f"显色指数 (Rf): {metrics['rf']:.2f}")
        print(f"视黑素日光效率比 (mel-DER): {metrics['mel']:.3f}")

        return {
            'weights': normalized_weights,
            'cct': metrics['cct'],
            'rf': metrics['rf'],
            'mel': metrics['mel'],
            'spd': metrics['spd_df'] if 'spd_df' in metrics else cache.compute(optimal_weights)['spd_df']
        }
    else:
        print("优化失败:", result.message)
        return None

optimize_daytime_lighting(sep_df)

