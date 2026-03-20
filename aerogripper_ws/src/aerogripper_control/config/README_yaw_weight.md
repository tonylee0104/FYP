# Yaw控制权重参数 (yaw_weight) 说明

## 问题背景

在无人机控制中，当控制分配矩阵的扭矩系数(`torque_coeff`)较小时，yaw方向的控制会变得过度敏感，导致：

1. **Yaw控制权重过大**: 在伪逆计算中，yaw方向的力矩需求被过度放大
2. **控制不平衡**: Roll、Pitch和Yaw三个方向的控制效果不均衡
3. **飞行不稳定**: Yaw方向的微小误差可能导致大幅度的电机输出变化

## 解决方案

通过引入`yaw_weight`参数，使用**加权伪逆**来平衡控制分配：

```cpp
// 加权矩阵 W
Eigen::Matrix<double,6,6> W = Eigen::Matrix<double,6,6>::Identity();
W(5,5) = yaw_weight_; // 第6行（yaw控制）使用yaw_weight_权重

// 加权伪逆计算
mix_matrix_ = (W * allocation_matrix_.transpose() * 
               (allocation_matrix_ * W * allocation_matrix_.transpose()).inverse());
```

## 参数含义

- **yaw_weight = 1.0**: 标准伪逆，无权重调整
- **yaw_weight < 1.0**: 降低yaw控制权重，减少yaw敏感度
- **yaw_weight > 1.0**: 增加yaw控制权重，提高yaw敏感度

## 推荐设置

### 对于小型无人机 (torque_coeff ≈ 0.01)
```yaml
yaw_weight: 0.1  # 降低yaw控制权重到1/10
```

### 对于中型无人机 (torque_coeff ≈ 0.05)
```yaml
yaw_weight: 0.3  # 降低yaw控制权重到1/3
```

### 对于大型无人机 (torque_coeff ≈ 0.1)
```yaml
yaw_weight: 0.5  # 降低yaw控制权重到1/2
```

## 调参步骤

1. **初始设置**: 从`yaw_weight = 0.1`开始
2. **飞行测试**: 观察yaw方向的响应特性
3. **逐步调整**: 
   - 如果yaw响应太慢: 增加`yaw_weight`值
   - 如果yaw响应太敏感: 减少`yaw_weight`值
4. **平衡优化**: 确保roll、pitch、yaw三个方向的控制效果均衡

## 数学原理

### 标准伪逆
```
mix_matrix = allocation_matrix^T * (allocation_matrix * allocation_matrix^T)^(-1)
```

### 加权伪逆
```
W = diag([1, 1, 1, 1, 1, yaw_weight])
mix_matrix = W * allocation_matrix^T * (allocation_matrix * W * allocation_matrix^T)^(-1)
```

### 效果分析
- 当`yaw_weight < 1`时，yaw方向的力矩需求在伪逆计算中被"压缩"
- 这相当于给yaw控制添加了一个"阻尼器"
- 结果是yaw方向的电机输出变化更加平缓

## 实际应用示例

### 配置文件设置
```yaml
aerogripper_control:
  ros__parameters:
    torque_coeff: 0.01        # 扭矩系数很小
    yaw_weight: 0.1          # 对应的yaw权重也较小
```

### 代码中的效果
```cpp
// 原始控制需求
Eigen::Vector3d tau_des(0.1, 0.1, 0.1); // roll, pitch, yaw力矩需求

// 使用加权伪逆后
// yaw方向的力矩需求被压缩到原来的1/10
// 实际yaw控制效果: 0.1 * 0.1 = 0.01
```

## 注意事项

1. **权重范围**: 建议`yaw_weight`在0.05到1.0之间
2. **数值稳定性**: 过小的权重可能导致数值计算不稳定
3. **控制性能**: 权重过小可能影响yaw方向的跟踪性能
4. **实时性**: 加权伪逆计算比标准伪逆稍慢，但影响很小

## 故障排除

### 如果yaw控制仍然过于敏感
- 进一步减少`yaw_weight`值
- 检查`torque_coeff`是否设置合理
- 调整姿态控制器的PID参数

### 如果yaw控制响应太慢
- 增加`yaw_weight`值
- 检查是否有其他限制因素
- 优化姿态控制器的响应特性

## 总结

`yaw_weight`参数是解决无人机yaw控制过度敏感问题的有效工具。通过合理设置这个参数，可以实现：

- **控制平衡**: Roll、Pitch、Yaw三个方向控制效果均衡
- **飞行稳定**: 减少yaw方向的振荡和过冲
- **性能优化**: 在保持控制精度的同时提高飞行稳定性

建议在实际应用中，根据无人机的具体特性和飞行需求，逐步调整`yaw_weight`参数，找到最佳的控制平衡点。
