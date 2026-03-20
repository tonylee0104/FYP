#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
力传感器数据分析程序（修正版）
根据实际实验情况：10个阶梯，每个阶梯约2秒，力值在0.3N范围内
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

def remove_invalid_data(data, threshold=0.1):
    """
    移除开始和结尾的无效数据
    无效数据定义为合力值小于阈值的部分
    """
    if not data:
        return data
    
    # 找到第一个有效数据点（合力 > 阈值）
    start_idx = 0
    for i, point in enumerate(data):
        if abs(point['total_force']) > threshold:
            start_idx = i
            break
    
    # 找到最后一个有效数据点
    end_idx = len(data) - 1
    for i in range(len(data) - 1, -1, -1):
        if abs(data[i]['total_force']) > threshold:
            end_idx = i
            break
    
    # 保留有效数据范围，并向前后各扩展一些数据点
    buffer = 50  # 缓冲区
    start_idx = max(0, start_idx - buffer)
    end_idx = min(len(data) - 1, end_idx + buffer)
    
    valid_data = data[start_idx:end_idx + 1]
    print(f"移除无效数据后保留: {len(valid_data)} 行 (从第{start_idx}行到第{end_idx}行)")
    
    return valid_data

def detect_force_steps_corrected(data, target_steps=10, step_duration_seconds=2):
    """
    根据实际实验情况检测力值阶梯
    - 目标：10个阶梯
    - 每个阶梯约2秒
    - 力值变化在0.3N范围内
    """
    if len(data) < 100:
        print("数据量不足，无法进行阶梯检测")
        return []
    
    # 计算采样频率
    total_time = (data[-1]['time'] - data[0]['time']).total_seconds()
    sample_rate = len(data) / total_time
    samples_per_step = int(step_duration_seconds * sample_rate)
    
    print(f"采样频率: {sample_rate:.1f} Hz")
    print(f"每个阶梯预期数据点数: {samples_per_step}")
    
    # 使用K-means聚类来识别阶梯
    # 将数据按时间分成大致相等的段
    total_samples = len(data)
    step_size = total_samples // target_steps
    
    # 为每个阶梯找到最佳分割点
    change_points = []
    for i in range(1, target_steps):
        # 在预期位置附近寻找最佳分割点
        expected_point = i * step_size
        search_start = max(0, expected_point - step_size // 2)
        search_end = min(total_samples, expected_point + step_size // 2)
        
        # 在搜索范围内找到力值变化最大的点
        max_change = 0
        best_point = expected_point
        
        for j in range(search_start, search_end):
            if j < len(data) - 1:
                # 计算前后数据的力值差异
                before_force = data[max(0, j - samples_per_step // 2):j]
                after_force = data[j:min(len(data), j + samples_per_step // 2)]
                
                if before_force and after_force:
                    before_mean = statistics.mean([p['total_force'] for p in before_force])
                    after_mean = statistics.mean([p['total_force'] for p in after_force])
                    change = abs(after_mean - before_mean)
                    
                    if change > max_change:
                        max_change = change
                        best_point = j
        
        change_points.append(best_point)
    
    # 排序并去重
    change_points = sorted(list(set(change_points)))
    
    print(f"检测到 {len(change_points)} 个变化点")
    return change_points

def segment_data_corrected(data, change_points, target_steps=10):
    """
    根据变化点分割数据，确保得到10个阶梯
    """
    segments = []
    
    # 添加开始和结束点
    all_points = [0] + change_points + [len(data)]
    
    for i in range(len(all_points) - 1):
        start_idx = all_points[i]
        end_idx = all_points[i + 1]
        
        if end_idx - start_idx > 10:  # 至少10个数据点
            segment_data = data[start_idx:end_idx]
            
            # 计算统计信息
            total_forces = [d['total_force'] for d in segment_data]
            x_forces = [d['x_force'] for d in segment_data]
            y_forces = [d['y_force'] for d in segment_data]
            z_forces = [d['z_force'] for d in segment_data]
            
            # 计算持续时间
            duration = (segment_data[-1]['time'] - segment_data[0]['time']).total_seconds()
            
            segments.append({
                'step_number': i + 1,
                'start_time': segment_data[0]['time'],
                'end_time': segment_data[-1]['time'],
                'duration_seconds': duration,
                'start_idx': start_idx,
                'end_idx': end_idx,
                'data_points': len(segment_data),
                'mean_total_force': statistics.mean(total_forces),
                'std_total_force': statistics.stdev(total_forces) if len(total_forces) > 1 else 0,
                'mean_x_force': statistics.mean(x_forces),
                'mean_y_force': statistics.mean(y_forces),
                'mean_z_force': statistics.mean(z_forces),
                'min_total_force': min(total_forces),
                'max_total_force': max(total_forces),
                'force_range': max(total_forces) - min(total_forces)
            })
    
    return segments

def print_results_corrected(segments):
    """
    打印修正后的分析结果
    """
    print("\n" + "="*120)
    print("力传感器数据分析结果（修正版）- 10个阶梯分析")
    print("="*120)
    
    print(f"{'阶梯':<4} {'开始时间':<20} {'结束时间':<20} {'持续时间(s)':<10} {'数据点数':<8} {'平均合力(N)':<12} {'标准差':<10} {'力值范围':<10} {'x力':<8} {'y力':<8} {'z力':<8}")
    print("-"*120)
    
    total_force_sum = 0
    for segment in segments:
        start_time = segment['start_time'].strftime('%H:%M:%S')
        end_time = segment['end_time'].strftime('%H:%M:%S')
        
        print(f"{segment['step_number']:<4} {start_time:<20} {end_time:<20} {segment['duration_seconds']:<10.1f} "
              f"{segment['data_points']:<8} {segment['mean_total_force']:<12.3f} {segment['std_total_force']:<10.3f} "
              f"{segment['force_range']:<10.3f} {segment['mean_x_force']:<8.3f} {segment['mean_y_force']:<8.3f} {segment['mean_z_force']:<8.3f}")
        
        total_force_sum += segment['mean_total_force']
    
    print("-"*120)
    print(f"总计 {len(segments)} 个阶梯")
    if segments:
        min_force = min([s['mean_total_force'] for s in segments])
        max_force = max([s['mean_total_force'] for s in segments])
        print(f"平均力值范围: {min_force:.3f}N - {max_force:.3f}N")
        print(f"所有阶梯平均力值: {total_force_sum/len(segments):.3f}N")
        print(f"力值变化范围: {max_force - min_force:.3f}N")

def save_results_to_csv_corrected(segments, output_file="force_analysis_corrected_results.csv"):
    """
    保存修正后的结果到CSV文件
    """
    try:
        with open(output_file, 'w', newline='', encoding='utf-8-sig') as file:
            writer = csv.writer(file)
            
            # 写入标题行
            writer.writerow([
                '阶梯编号', '开始时间', '结束时间', '持续时间(秒)', '数据点数',
                '平均合力(N)', '标准差', '力值范围', '平均x力(N)', '平均y力(N)', '平均z力(N)',
                '最小合力(N)', '最大合力(N)'
            ])
            
            # 写入数据行
            for segment in segments:
                writer.writerow([
                    segment['step_number'],
                    segment['start_time'].strftime('%Y-%m-%d %H:%M:%S'),
                    segment['end_time'].strftime('%Y-%m-%d %H:%M:%S'),
                    f"{segment['duration_seconds']:.1f}",
                    segment['data_points'],
                    f"{segment['mean_total_force']:.6f}",
                    f"{segment['std_total_force']:.6f}",
                    f"{segment['force_range']:.6f}",
                    f"{segment['mean_x_force']:.6f}",
                    f"{segment['mean_y_force']:.6f}",
                    f"{segment['mean_z_force']:.6f}",
                    f"{segment['min_total_force']:.6f}",
                    f"{segment['max_total_force']:.6f}"
                ])
        
        print(f"\n修正后的分析结果已保存到: {output_file}")
        
    except Exception as e:
        print(f"保存结果时出错: {e}")

def main():
    """
    主函数
    """
    print("力传感器数据分析程序（修正版）")
    print("="*60)
    print("目标：识别10个阶梯，每个阶梯约2秒，力值在0.3N范围内")
    print("="*60)
    
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
    
    # 移除无效数据
    print("\n开始移除无效数据...")
    valid_data = remove_invalid_data(data, threshold=0.05)
    
    if not valid_data:
        print("没有有效数据，无法继续分析")
        return
    
    # 检测力值阶梯
    print("\n开始检测力值阶梯...")
    change_points = detect_force_steps_corrected(valid_data, target_steps=10, step_duration_seconds=2)
    
    if not change_points:
        print("未检测到明显的力值变化，尝试调整参数...")
        return
    
    # 分割数据
    segments = segment_data_corrected(valid_data, change_points, target_steps=10)
    print(f"分割为 {len(segments)} 个数据段")
    
    if not segments:
        print("无法分割数据，请检查数据或调整参数")
        return
    
    # 打印结果
    print_results_corrected(segments)
    
    # 保存结果
    save_results_to_csv_corrected(segments)
    
    # 显示每个阶梯的详细信息
    print("\n详细分析:")
    for segment in segments:
        print(f"\n阶梯 {segment['step_number']}:")
        print(f"  时间范围: {segment['start_time'].strftime('%H:%M:%S')} - {segment['end_time'].strftime('%H:%M:%S')}")
        print(f"  持续时间: {segment['duration_seconds']:.1f} 秒")
        print(f"  数据点数: {segment['data_points']}")
        print(f"  平均合力: {segment['mean_total_force']:.6f}N ± {segment['std_total_force']:.6f}N")
        print(f"  力值范围: {segment['force_range']:.6f}N")
        print(f"  平均x力: {segment['mean_x_force']:.6f}N")
        print(f"  平均y力: {segment['mean_y_force']:.6f}N")
        print(f"  平均z力: {segment['mean_z_force']:.6f}N")
        print(f"  合力范围: {segment['min_total_force']:.6f}N - {segment['max_total_force']:.6f}N")

if __name__ == "__main__":
    main()
