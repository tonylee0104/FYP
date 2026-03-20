#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
力传感器数据分析程序（简化版）
只使用Python标准库，分析阶梯形力数据
"""

import csv
import datetime
import statistics
from collections import defaultdict

def load_csv_data(file_path):
    """
    加载CSV数据文件
    """
    data = []
    encodings = ['gbk', 'gb2312', 'utf-8', 'latin1']
    
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as file:
                # 跳过第一行标题
                next(file)
                
                for line_num, line in enumerate(file, 2):
                    try:
                        # 移除开头的单引号并分割
                        line = line.strip().replace("'", "")
                        parts = line.split(',')
                        
                        if len(parts) >= 5:
                            # 解析时间
                            time_str = parts[0]
                            try:
                                time_obj = datetime.datetime.strptime(time_str, '%Y-%m-%d %H:%M:%S.%f')
                            except ValueError:
                                try:
                                    time_obj = datetime.datetime.strptime(time_str, '%Y-%m-%d %H:%M:%S')
                                except ValueError:
                                    print(f"第{line_num}行时间格式错误: {time_str}")
                                    continue
                            
                            # 解析力值
                            try:
                                x_force = float(parts[1])
                                y_force = float(parts[2])
                                z_force = float(parts[3])
                                total_force = float(parts[4])
                                
                                data.append({
                                    'time': time_obj,
                                    'x_force': x_force,
                                    'y_force': y_force,
                                    'z_force': z_force,
                                    'total_force': total_force
                                })
                            except ValueError as e:
                                print(f"第{line_num}行力值解析错误: {e}")
                                continue
                                
                    except Exception as e:
                        print(f"第{line_num}行处理错误: {e}")
                        continue
            
            print(f"使用 {encoding} 编码成功加载 {len(data)} 行数据")
            return data
            
        except UnicodeDecodeError:
            continue
        except Exception as e:
            print(f"使用 {encoding} 编码读取文件时出错: {e}")
            continue
    
    print("所有编码都失败，无法读取文件")
    return None

def detect_force_changes(data, window_size=100, threshold_factor=0.05):
    """
    检测力值变化点
    """
    if len(data) < window_size * 2:
        print("数据量不足，无法进行阶梯检测")
        return []
    
    # 计算滑动窗口内的标准差
    local_stds = []
    half_window = window_size // 2
    
    for i in range(len(data)):
        start_idx = max(0, i - half_window)
        end_idx = min(len(data), i + half_window)
        
        window_data = [data[j]['total_force'] for j in range(start_idx, end_idx)]
        if len(window_data) > 1:
            local_stds.append(statistics.stdev(window_data))
        else:
            local_stds.append(0)
    
    # 计算阈值
    mean_std = statistics.mean(local_stds)
    threshold = mean_std * threshold_factor
    
    # 找到变化点
    change_points = []
    for i in range(1, len(local_stds) - 1):
        if local_stds[i] > threshold:
            # 检查是否是局部最大值
            if (local_stds[i] > local_stds[i-1] and 
                local_stds[i] > local_stds[i+1]):
                change_points.append(i)
    
    # 合并相近的变化点
    if len(change_points) > 1:
        merged_points = [change_points[0]]
        for point in change_points[1:]:
            if point - merged_points[-1] > window_size:
                merged_points.append(point)
        change_points = merged_points
    
    return change_points

def segment_data(data, change_points):
    """
    根据变化点分割数据
    """
    segments = []
    
    # 添加开始和结束点
    all_points = [0] + change_points + [len(data)]
    
    for i in range(len(all_points) - 1):
        start_idx = all_points[i]
        end_idx = all_points[i + 1]
        
        if end_idx - start_idx > 20:  # 至少20个数据点
            segment_data = data[start_idx:end_idx]
            
            # 计算统计信息
            total_forces = [d['total_force'] for d in segment_data]
            x_forces = [d['x_force'] for d in segment_data]
            y_forces = [d['y_force'] for d in segment_data]
            z_forces = [d['z_force'] for d in segment_data]
            
            segments.append({
                'start_time': segment_data[0]['time'],
                'end_time': segment_data[-1]['time'],
                'start_idx': start_idx,
                'end_idx': end_idx,
                'data_points': len(segment_data),
                'mean_total_force': statistics.mean(total_forces),
                'std_total_force': statistics.stdev(total_forces) if len(total_forces) > 1 else 0,
                'mean_x_force': statistics.mean(x_forces),
                'mean_y_force': statistics.mean(y_forces),
                'mean_z_force': statistics.mean(z_forces),
                'min_total_force': min(total_forces),
                'max_total_force': max(total_forces)
            })
    
    return segments

def print_results(segments):
    """
    打印分析结果
    """
    print("\n" + "="*100)
    print("力传感器数据分析结果")
    print("="*100)
    
    print(f"{'阶梯':<4} {'开始时间':<20} {'结束时间':<20} {'数据点数':<8} {'平均合力(N)':<12} {'标准差':<10} {'x力':<8} {'y力':<8} {'z力':<8} {'范围':<15}")
    print("-"*100)
    
    total_force_sum = 0
    for i, segment in enumerate(segments):
        start_time = segment['start_time'].strftime('%H:%M:%S')
        end_time = segment['end_time'].strftime('%H:%M:%S')
        force_range = f"{segment['min_total_force']:.3f}-{segment['max_total_force']:.3f}"
        
        print(f"{i+1:<4} {start_time:<20} {end_time:<20} {segment['data_points']:<8} "
              f"{segment['mean_total_force']:<12.3f} {segment['std_total_force']:<10.3f} "
              f"{segment['mean_x_force']:<8.3f} {segment['mean_y_force']:<8.3f} {segment['mean_z_force']:<8.3f} "
              f"{force_range:<15}")
        
        total_force_sum += segment['mean_total_force']
    
    print("-"*100)
    print(f"总计 {len(segments)} 个阶梯")
    if segments:
        min_force = min([s['mean_total_force'] for s in segments])
        max_force = max([s['mean_total_force'] for s in segments])
        print(f"平均力值范围: {min_force:.3f}N - {max_force:.3f}N")
        print(f"所有阶梯平均力值: {total_force_sum/len(segments):.3f}N")

def save_results_to_csv(segments, output_file="force_analysis_results.csv"):
    """
    保存结果到CSV文件
    """
    try:
        with open(output_file, 'w', newline='', encoding='utf-8-sig') as file:
            writer = csv.writer(file)
            
            # 写入标题行
            writer.writerow([
                '阶梯编号', '开始时间', '结束时间', '数据点数',
                '平均合力(N)', '标准差', '平均x力(N)', '平均y力(N)', '平均z力(N)',
                '最小合力(N)', '最大合力(N)'
            ])
            
            # 写入数据行
            for i, segment in enumerate(segments):
                writer.writerow([
                    i + 1,
                    segment['start_time'].strftime('%Y-%m-%d %H:%M:%S'),
                    segment['end_time'].strftime('%Y-%m-%d %H:%M:%S'),
                    segment['data_points'],
                    f"{segment['mean_total_force']:.6f}",
                    f"{segment['std_total_force']:.6f}",
                    f"{segment['mean_x_force']:.6f}",
                    f"{segment['mean_y_force']:.6f}",
                    f"{segment['mean_z_force']:.6f}",
                    f"{segment['min_total_force']:.6f}",
                    f"{segment['max_total_force']:.6f}"
                ])
        
        print(f"\n分析结果已保存到: {output_file}")
        
    except Exception as e:
        print(f"保存结果时出错: {e}")

def main():
    """
    主函数
    """
    print("力传感器数据分析程序（简化版）")
    print("="*50)
    
    # 数据文件路径
    file_path = "20250811-201025.csv"
    
    # 加载数据
    data = load_csv_data(file_path)
    if not data:
        return
    
    # 显示数据基本信息
    print(f"数据时间范围: {data[0]['time']} 到 {data[-1]['time']}")
    print(f"数据采样间隔: 约 {(data[-1]['time'] - data[0]['time']).total_seconds() / len(data):.3f} 秒")
    
    total_forces = [d['total_force'] for d in data]
    print(f"合力范围: {min(total_forces):.3f}N 到 {max(total_forces):.3f}N")
    
    # 检测力值变化点
    print("\n开始检测力值变化点...")
    change_points = detect_force_changes(data)
    print(f"检测到 {len(change_points)} 个变化点")
    
    if not change_points:
        print("未检测到明显的力值变化，尝试调整参数...")
        # 尝试不同的参数
        change_points = detect_force_changes(data, window_size=50, threshold_factor=0.02)
        print(f"使用调整后参数检测到 {len(change_points)} 个变化点")
    
    # 分割数据
    segments = segment_data(data, change_points)
    print(f"分割为 {len(segments)} 个数据段")
    
    if not segments:
        print("无法分割数据，请检查数据或调整参数")
        return
    
    # 打印结果
    print_results(segments)
    
    # 保存结果
    save_results_to_csv(segments)
    
    # 显示每个阶梯的详细信息
    print("\n详细分析:")
    for i, segment in enumerate(segments):
        print(f"\n阶梯 {i+1}:")
        print(f"  时间范围: {segment['start_time'].strftime('%H:%M:%S')} - {segment['end_time'].strftime('%H:%M:%S')}")
        print(f"  数据点数: {segment['data_points']}")
        print(f"  平均合力: {segment['mean_total_force']:.6f}N ± {segment['std_total_force']:.6f}N")
        print(f"  平均x力: {segment['mean_x_force']:.6f}N")
        print(f"  平均y力: {segment['mean_y_force']:.6f}N")
        print(f"  平均z力: {segment['mean_z_force']:.6f}N")
        print(f"  合力范围: {segment['min_total_force']:.6f}N - {segment['max_total_force']:.6f}N")

if __name__ == "__main__":
    main()
