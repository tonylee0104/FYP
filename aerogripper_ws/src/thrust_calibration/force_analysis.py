#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
力传感器数据分析程序
分析阶梯形力数据，计算每个阶梯的平均力值
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

def load_and_clean_data(file_path):
    """
    加载并清理CSV数据
    """
    try:
        # 读取CSV文件，跳过第一行（中文标题）
        df = pd.read_csv(file_path, skiprows=1, header=None)
        
        # 设置列名
        df.columns = ['时间', 'x力', 'y力', 'z力', '合力']
        
        # 清理时间列（移除开头的单引号）
        df['时间'] = df['时间'].str.replace("'", "")
        
        # 转换时间列为datetime格式
        df['时间'] = pd.to_datetime(df['时间'])
        
        # 转换力列为数值类型
        for col in ['x力', 'y力', 'z力', '合力']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # 移除无效数据
        df = df.dropna()
        
        print(f"成功加载数据，共 {len(df)} 行")
        return df
        
    except Exception as e:
        print(f"加载数据时出错: {e}")
        return None

def detect_force_steps(force_data, window_size=50, threshold_factor=0.1):
    """
    检测力数据的阶梯变化
    使用滑动窗口和变化检测来识别阶梯
    """
    # 计算合力的变化率
    force_diff = np.abs(np.diff(force_data))
    
    # 使用滑动窗口计算局部标准差
    local_std = []
    for i in range(len(force_data)):
        start = max(0, i - window_size // 2)
        end = min(len(force_data), i + window_size // 2)
        local_std.append(np.std(force_data[start:end]))
    
    local_std = np.array(local_std)
    
    # 检测变化点（当局部标准差超过阈值时）
    threshold = np.mean(local_std) * threshold_factor
    change_points = np.where(local_std > threshold)[0]
    
    # 合并相近的变化点
    if len(change_points) > 1:
        merged_points = [change_points[0]]
        for point in change_points[1:]:
            if point - merged_points[-1] > window_size:
                merged_points.append(point)
        change_points = np.array(merged_points)
    
    return change_points

def segment_force_data(df, change_points):
    """
    根据变化点分割力数据
    """
    segments = []
    
    # 添加开始点
    all_points = [0] + list(change_points) + [len(df)]
    
    for i in range(len(all_points) - 1):
        start_idx = all_points[i]
        end_idx = all_points[i + 1]
        
        if end_idx - start_idx > 10:  # 至少10个数据点
            segment_data = df.iloc[start_idx:end_idx]
            segments.append({
                'start_time': segment_data.iloc[0]['时间'],
                'end_time': segment_data.iloc[-1]['时间'],
                'start_idx': start_idx,
                'end_idx': end_idx,
                'data': segment_data,
                'mean_force': segment_data['合力'].mean(),
                'std_force': segment_data['合力'].std(),
                'mean_x': segment_data['x力'].mean(),
                'mean_y': segment_data['y力'].mean(),
                'mean_z': segment_data['z力'].mean()
            })
    
    return segments

def analyze_force_steps(df):
    """
    分析力数据的阶梯
    """
    print("开始分析力数据阶梯...")
    
    # 获取合力数据
    force_data = df['合力'].values
    
    # 检测阶梯变化点
    change_points = detect_force_steps(force_data)
    print(f"检测到 {len(change_points)} 个变化点")
    
    # 分割数据
    segments = segment_force_data(df, change_points)
    print(f"分割为 {len(segments)} 个数据段")
    
    return segments

def plot_force_analysis(df, segments):
    """
    绘制力数据分析图
    """
    plt.figure(figsize=(15, 10))
    
    # 绘制原始合力数据
    plt.subplot(2, 1, 1)
    plt.plot(df['时间'], df['合力'], 'b-', alpha=0.7, label='原始合力数据')
    
    # 标记每个阶梯
    colors = plt.cm.tab10(np.linspace(0, 1, len(segments)))
    for i, segment in enumerate(segments):
        segment_data = segment['data']
        plt.plot(segment_data['时间'], segment_data['合力'], 
                color=colors[i], linewidth=2, 
                label=f'阶梯 {i+1}: {segment["mean_force"]:.3f}N')
        
        # 标记阶梯边界
        plt.axvline(x=segment['start_time'], color=colors[i], linestyle='--', alpha=0.5)
    
    plt.axvline(x=segment['end_time'], color=colors[-1], linestyle='--', alpha=0.5)
    
    plt.xlabel('时间')
    plt.ylabel('合力 (N)')
    plt.title('力传感器数据阶梯分析')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # 绘制阶梯平均力值
    plt.subplot(2, 1, 2)
    step_numbers = range(1, len(segments) + 1)
    mean_forces = [seg['mean_force'] for seg in segments]
    
    bars = plt.bar(step_numbers, mean_forces, color=colors[:len(segments)])
    plt.xlabel('阶梯编号')
    plt.ylabel('平均合力 (N)')
    plt.title('各阶梯平均力值')
    plt.grid(True, alpha=0.3)
    
    # 在柱状图上添加数值标签
    for i, (bar, force) in enumerate(zip(bars, mean_forces)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{force:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()

def print_analysis_results(segments):
    """
    打印分析结果
    """
    print("\n" + "="*80)
    print("力传感器数据分析结果")
    print("="*80)
    
    print(f"{'阶梯':<4} {'开始时间':<20} {'结束时间':<20} {'平均合力(N)':<12} {'标准差':<10} {'x力':<8} {'y力':<8} {'z力':<8}")
    print("-"*100)
    
    total_force = 0
    for i, segment in enumerate(segments):
        start_time = segment['start_time'].strftime('%H:%M:%S')
        end_time = segment['end_time'].strftime('%H:%M:%S')
        
        print(f"{i+1:<4} {start_time:<20} {end_time:<20} "
              f"{segment['mean_force']:<12.3f} {segment['std_force']:<10.3f} "
              f"{segment['mean_x']:<8.3f} {segment['mean_y']:<8.3f} {segment['mean_z']:<8.3f}")
        
        total_force += segment['mean_force']
    
    print("-"*100)
    print(f"总计 {len(segments)} 个阶梯，平均力值范围: {min([s['mean_force'] for s in segments]):.3f}N - {max([s['mean_force'] for s in segments]):.3f}N")
    print(f"所有阶梯平均力值: {total_force/len(segments):.3f}N")

def main():
    """
    主函数
    """
    # 数据文件路径
    file_path = "20250811-201025.csv"
    
    print("力传感器数据分析程序")
    print("="*50)
    
    # 加载数据
    df = load_and_clean_data(file_path)
    if df is None:
        return
    
    # 显示数据基本信息
    print(f"数据时间范围: {df['时间'].min()} 到 {df['时间'].max()}")
    print(f"数据采样间隔: 约 {(df['时间'].max() - df['时间'].min()).total_seconds() / len(df):.3f} 秒")
    print(f"合力范围: {df['合力'].min():.3f}N 到 {df['合力'].max():.3f}N")
    
    # 分析力数据阶梯
    segments = analyze_force_steps(df)
    
    if len(segments) == 0:
        print("未检测到明显的力数据阶梯，请检查数据或调整参数")
        return
    
    # 打印分析结果
    print_analysis_results(segments)
    
    # 绘制分析图
    try:
        plot_force_analysis(df, segments)
    except Exception as e:
        print(f"绘图时出错: {e}")
        print("跳过绘图步骤")
    
    # 保存结果到CSV文件
    try:
        results_df = pd.DataFrame([
            {
                '阶梯编号': i+1,
                '开始时间': seg['start_time'],
                '结束时间': seg['end_time'],
                '平均合力(N)': seg['mean_force'],
                '标准差': seg['std_force'],
                '平均x力(N)': seg['mean_x'],
                '平均y力(N)': seg['mean_y'],
                '平均z力(N)': seg['mean_z']
            }
            for i, seg in enumerate(segments)
        ])
        
        output_file = "force_analysis_results.csv"
        results_df.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"\n分析结果已保存到: {output_file}")
        
    except Exception as e:
        print(f"保存结果时出错: {e}")

if __name__ == "__main__":
    main()
