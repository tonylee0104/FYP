#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import math

def create_allocation_matrix(thrust_arm_length, torque_coeff):
    """创建控制分配矩阵"""
    sqrt2_2 = math.sqrt(2.0) / 2.0
    
    allocation_matrix = np.array([
        [-0.5,          0.5,          -0.5,           0.5],      # roll
        [0.5,         -0.5,          -0.5,           0.5],      # pitch
        [-sqrt2_2,       -sqrt2_2,       -sqrt2_2,        -sqrt2_2],  # thrust
        [0.5*thrust_arm_length - 0.5*torque_coeff,  -0.5*thrust_arm_length + 0.5*torque_coeff,  -0.5*thrust_arm_length + 0.5*torque_coeff,  0.5*thrust_arm_length - 0.5*torque_coeff],  # roll moment
        [0.5*thrust_arm_length + 0.5*torque_coeff,  -0.5*thrust_arm_length - 0.5*torque_coeff,   0.5*thrust_arm_length + 0.5*torque_coeff, -0.5*thrust_arm_length - 0.5*torque_coeff],  # pitch moment
        [-sqrt2_2*torque_coeff,  -sqrt2_2*torque_coeff,  sqrt2_2*torque_coeff,  sqrt2_2*torque_coeff]   # yaw moment
    ])
    
    return allocation_matrix

def weighted_pseudo_inverse(allocation_matrix, yaw_weight):
    """计算加权伪逆"""
    W = np.eye(6)
    W[5,5] = yaw_weight  # yaw控制权重
    
    # 加权伪逆: W * A^T * (A * W * A^T)^(-1)
    mix_matrix = W @ allocation_matrix.T @ np.linalg.inv(allocation_matrix @ W @ allocation_matrix.T)
    
    return mix_matrix

def standard_pseudo_inverse(allocation_matrix):
    """计算标准伪逆"""
    return np.linalg.pinv(allocation_matrix)

def analyze_control_allocation(allocation_matrix, mix_matrix, control_demand):
    """分析控制分配效果"""
    # 计算电机输出
    motor_output = mix_matrix @ control_demand
    
    # 验证控制分配
    actual_control = allocation_matrix @ motor_output
    
    # 计算误差
    error = control_demand - actual_control
    
    return motor_output, actual_control, error

def plot_comparison(yaw_weights, motor_outputs, control_errors):
    """绘制比较图"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. 电机输出比较
    x = np.arange(4)
    width = 0.8 / len(yaw_weights)
    
    for i, weight in enumerate(yaw_weights):
        ax1.bar(x + i*width, motor_outputs[i], width, label=f'yaw_weight={weight}')
    
    ax1.set_xlabel('电机编号')
    ax1.set_ylabel('电机输出')
    ax1.set_title('不同yaw_weight下的电机输出比较')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 控制误差比较
    control_names = ['Roll', 'Pitch', 'Thrust', 'Roll_M', 'Pitch_M', 'Yaw_M']
    x = np.arange(6)
    
    for i, weight in enumerate(yaw_weights):
        ax2.bar(x + i*width, control_errors[i], width, label=f'yaw_weight={weight}')
    
    ax2.set_xlabel('控制通道')
    ax2.set_ylabel('控制误差')
    ax2.set_title('不同yaw_weight下的控制误差比较')
    ax2.set_xticks(x + width * (len(yaw_weights)-1)/2)
    ax2.set_xticklabels(control_names)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Yaw控制权重对电机输出的影响
    yaw_motor_weights = []
    for i, weight in enumerate(yaw_weights):
        # 计算yaw控制对每个电机的影响权重
        yaw_row = mix_matrices[i][:, 5]  # 第6列对应yaw控制
        yaw_motor_weights.append(np.abs(yaw_row))
    
    yaw_motor_weights = np.array(yaw_motor_weights)
    
    for i in range(4):
        ax3.plot(yaw_weights, yaw_motor_weights[:, i], 'o-', label=f'Motor {i+1}')
    
    ax3.set_xlabel('yaw_weight')
    ax3.set_ylabel('Yaw控制权重')
    ax3.set_title('Yaw控制权重对电机输出的影响')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_xscale('log')
    
    # 4. 控制分配矩阵的条件数
    condition_numbers = []
    for weight in yaw_weights:
        W = np.eye(6)
        W[5,5] = weight
        weighted_matrix = allocation_matrix @ W
        cond_num = np.linalg.cond(weighted_matrix)
        condition_numbers.append(cond_num)
    
    ax4.plot(yaw_weights, condition_numbers, 'o-', color='red')
    ax4.set_xlabel('yaw_weight')
    ax4.set_ylabel('条件数')
    ax4.set_title('加权矩阵的条件数')
    ax4.grid(True, alpha=0.3)
    ax4.set_xscale('log')
    
    plt.tight_layout()
    plt.show()

def main():
    """主函数"""
    print("=== Yaw控制权重测试程序 ===")
    
    # 参数设置
    thrust_arm_length = 0.055
    torque_coeff = 0.01
    yaw_weights = [1.0, 0.5, 0.2, 0.1, 0.05]
    
    # 控制需求 (roll, pitch, thrust, roll_moment, pitch_moment, yaw_moment)
    control_demand = np.array([0.1, 0.1, 0.5, 0.01, 0.01, 0.01])
    
    print(f"推力臂长度: {thrust_arm_length} m")
    print(f"扭矩系数: {torque_coeff}")
    print(f"控制需求: {control_demand}")
    print()
    
    # 创建控制分配矩阵
    allocation_matrix = create_allocation_matrix(thrust_arm_length, torque_coeff)
    print("控制分配矩阵:")
    print(allocation_matrix)
    print()
    
    # 测试不同的yaw_weight值
    motor_outputs = []
    control_errors = []
    mix_matrices = []
    
    for yaw_weight in yaw_weights:
        print(f"测试 yaw_weight = {yaw_weight}")
        
        # 计算加权伪逆
        mix_matrix = weighted_pseudo_inverse(allocation_matrix, yaw_weight)
        mix_matrices.append(mix_matrix)
        
        # 分析控制分配效果
        motor_output, actual_control, error = analyze_control_allocation(
            allocation_matrix, mix_matrix, control_demand)
        
        motor_outputs.append(motor_output)
        control_errors.append(error)
        
        print(f"  电机输出: {motor_output}")
        print(f"  实际控制: {actual_control}")
        print(f"  控制误差: {error}")
        print(f"  误差范数: {np.linalg.norm(error):.6f}")
        print()
    
    # 绘制比较图
    try:
        plot_comparison(yaw_weights, motor_outputs, control_errors)
        print("图表已生成，请查看窗口。")
    except Exception as e:
        print(f"绘图失败: {e}")
        print("请确保已安装matplotlib库")
    
    # 总结和建议
    print("=== 调参建议 ===")
    print("1. 对于torque_coeff=0.01的小型无人机:")
    print("   - 建议yaw_weight = 0.1")
    print("   - 这样可以显著降低yaw控制的敏感度")
    print()
    print("2. 如果yaw控制仍然过于敏感:")
    print("   - 可以进一步降低yaw_weight到0.05")
    print("   - 但要注意不要过低，避免数值不稳定")
    print()
    print("3. 如果yaw控制响应太慢:")
    print("   - 可以增加yaw_weight到0.2或0.5")
    print("   - 在稳定性和响应性之间找到平衡")
    print()
    print("4. 监控指标:")
    print("   - 控制误差范数应该保持较小")
    print("   - 电机输出变化应该平滑")
    print("   - 避免电机输出饱和")

if __name__ == "__main__":
    main()
