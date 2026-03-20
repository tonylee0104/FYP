# Aerogripper Control Package

## 参数系统说明

本包已重构为使用集中式参数管理系统。所有参数现在通过main函数从yaml配置文件读取，然后分发到各个类中。

## 代码结构

### 主要组件

1. **main.cpp**: 主程序入口，包含参数获取和节点管理
2. **PoseController**: 姿态控制器，负责飞行器姿态控制
3. **TetherObserver**: 绳索观测器，负责绳索拉力观测
4. **Mixer**: 混控器，负责电机控制分配

### 参数管理

- **ControllerParams结构体**: 定义所有参数的统一数据结构
- **get_all_parameters()函数**: 从yaml文件读取所有参数并返回结构体
- **set_params()函数**: 各个类中用于接收参数的接口函数

## 使用方法

### 1. 启动节点

使用launch文件启动节点：

```bash
ros2 launch aerogripper_control test.launch.py
```

### 2. 参数配置

所有参数都在 `config/config.yaml` 文件中配置，使用ROS2标准的参数格式：

```yaml
aerogripper_control:
  ros__parameters:
    # 控制参数
    kp_pos: [2.0, 2.0, 3.0]
    kv_pos: [0.6, 0.6, 0.8]
    k_R: [1.0, 1.0, 1.0]
    kp_att: [8.0, 8.0, 5.0]
    ki_att: [0.0, 0.0, 0.0]
    kd_att: [0.1, 0.1, 0.1]

    # 物理参数
    mass: 0.2
    inertia: [0.01, 0.01, 0.005]
    thrust_coeff: 0.0002
    thrust_arm_length: 0.05
    gravity_arm_length: 0.05
    torque_coeff: 0.01
    rpm_coeff: 1.0
```

**注意**: 
- 使用ROS2标准的 `ros__parameters` 格式
- 所有参数都在 `aerogripper_control` 命名空间下
- 相同名称的参数（如 `mass`）在多个类中共享同一个值
- 这种设计避免了参数重复，使配置更加简洁

### 3. 参数分发机制

- **main.cpp**: 调用 `get_all_parameters()` 获取所有参数，然后通过 `set_params()` 分发到各个类
- **PoseController**: 通过 `set_params()` 函数接收控制参数
- **TetherObserver**: 通过 `set_params()` 函数接收观测参数  
- **Mixer**: 通过 `set_params()` 函数接收物理参数

### 4. 运行时参数修改

如需修改参数，请编辑 `config/config.yaml` 文件，然后重新启动节点。

## 架构优势

1. **集中管理**: 所有参数在一个文件中管理
2. **类型安全**: 参数类型在编译时确定
3. **易于维护**: 参数修改不需要重新编译代码
4. **清晰分离**: 参数读取和业务逻辑分离
5. **模块化设计**: 参数获取逻辑封装在独立函数中，便于维护和测试
6. **参数名称一致性**: yaml文件中的参数名称与代码中的读取方式完全匹配

## 注意事项

- 确保yaml文件格式正确
- 参数名称必须与代码中的声明完全匹配（使用点号分隔的扁平化格式）
- 修改参数后需要重启节点才能生效
- 所有参数都有默认值，确保程序在配置文件缺失时仍能正常运行
