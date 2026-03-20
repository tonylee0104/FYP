# 参数文件使用说明

## 📁 文件结构

```
workspace/aerogripper_ws/src/aerogripper_control/
├── config/
│   ├── config.yaml              # 主配置文件 (当前使用)
│   ├── controller_params.yaml   # 完整参数配置 (参考用)
│   └── README_yaw_weight.md     # Yaw权重参数说明
├── launch/
│   └── test.launch.py          # Launch文件
└── src/
    └── pose_controller.cpp      # 控制器实现
```

## 🚀 启动方式

### 1. 使用Launch文件启动 (推荐)
```bash
cd workspace/aerogripper_ws
source install/setup.bash
ros2 launch aerogripper_control test.launch.py
```

### 2. 直接运行节点
```bash
cd workspace/aerogripper_ws
source install/setup.bash
ros2 run aerogripper_control pose_controller
```

## 📋 参数文件配置

### 主配置文件: config.yaml
- **路径**: `config/config.yaml`
- **状态**: ✅ 当前使用中
- **包含**: 所有必要的控制器参数
- **特点**: 与launch文件匹配，自动加载

### 参考配置文件: controller_params.yaml
- **路径**: `config/controller_params.yaml`
- **状态**: 📖 参考文档
- **包含**: 完整的参数说明和示例
- **特点**: 详细的注释和推荐值

## 🔧 参数加载机制

### 1. Launch文件中的路径配置
```python
# launch/test.launch.py
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # 获取包的共享目录路径
    pkg_share = get_package_share_directory('aerogripper_control')
    
    # 构建config.yaml的完整路径
    config_file = os.path.join(pkg_share, 'config', 'config.yaml')
    
    # 启动节点时传入参数文件
    Node(
        package='aerogripper_control',
        executable='aerogripper_control_node',
        name='aerogripper_control',
        parameters=[config_file],  # 自动加载参数文件
        output='screen'
    )
```

### 2. 代码中的参数声明
```cpp
// src/pose_controller.cpp
PoseController::PoseController() : Node("aerogripper_control")
{
    // 声明参数（这些是默认值，会被yaml文件覆盖）
    this->declare_parameter("kp_pos", std::vector<double>{2.0, 2.0, 3.0});
    this->declare_parameter("kv_pos", std::vector<double>{0.6, 0.6, 0.8});
    this->declare_parameter("k_R", std::vector<double>{1.0, 1.0, 1.0});
    this->declare_parameter("kp_att", std::vector<double>{8.0, 8.0, 5.0});
    this->declare_parameter("ki_att", std::vector<double>{0.0, 0.0, 0.0});
    this->declare_parameter("kd_att", std::vector<double>{0.1, 0.1, 0.1});
    this->declare_parameter("mass", 0.110);
    this->declare_parameter("inertia", std::vector<double>{0.01, 0.01, 0.005});
    this->declare_parameter("thrust_coeff", 1.99);
    this->declare_parameter("thrust_arm_length", 0.055);
    this->declare_parameter("gravity_arm_length", 0.045);
    this->declare_parameter("torque_coeff", 0.01);
    this->declare_parameter("rpm_coeff", 1.0);
    this->declare_parameter("yaw_weight", 1.0);  // 新增的yaw权重参数
    this->declare_parameter("compensation_factor", 1.5);
    this->declare_parameter("filter_coefficient", 0.3);
    this->declare_parameter("max_throttle_delta", 0.1);
    
    // 等待一小段时间确保参数加载完成
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    
    // 从yaml文件加载参数并设置
    loadAndSetParameters();
}
```

## 📊 参数优先级

1. **YAML文件参数** (最高优先级)
2. **代码中的默认值** (最低优先级)

这意味着：
- 如果YAML文件中定义了参数，会覆盖代码中的默认值
- 如果YAML文件中没有定义参数，会使用代码中的默认值

## 🛠️ 修改参数

### 方法1: 修改YAML文件 (推荐)
```bash
# 编辑配置文件
nano workspace/aerogripper_ws/src/aerogripper_control/config/config.yaml

# 重新构建
cd workspace/aerogripper_ws
colcon build --packages-select aerogripper_control

# 重新启动
ros2 launch aerogripper_control test.launch.py
```

### 方法2: 运行时动态修改
```bash
# 查看当前参数
ros2 param list /aerogripper_control

# 修改单个参数
ros2 param set /aerogripper_control yaw_weight 0.05

# 查看参数值
ros2 param get /aerogripper_control yaw_weight
```

### 方法3: 保存参数到文件
```bash
# 保存当前参数到文件
ros2 param dump /aerogripper_control > current_params.yaml

# 从文件加载参数
ros2 param load /aerogripper_control current_params.yaml
```

## 🔍 调试参数

### 1. 查看所有参数
```bash
ros2 param list /aerogripper_control
```

### 2. 查看参数值
```bash
ros2 param get /aerogripper_control yaw_weight
ros2 param get /aerogripper_control torque_coeff
```

### 3. 查看参数描述
```bash
ros2 param describe /aerogripper_control yaw_weight
```

### 4. 监控参数变化
```bash
ros2 param monitor /aerogripper_control
```

## ⚠️ 注意事项

1. **文件路径**: 确保launch文件中的路径正确
2. **参数名称**: 参数名称必须与代码中声明的一致
3. **数据类型**: 确保YAML文件中的数据类型正确
4. **重新构建**: 修改YAML文件后需要重新构建项目
5. **节点名称**: 确保节点名称与参数命名空间匹配

## 🚨 常见问题

### Q: 参数没有加载怎么办？
A: 检查launch文件中的路径是否正确，确保config.yaml文件存在

### Q: 如何添加新参数？
A: 在代码中声明参数，然后在YAML文件中设置值

### Q: 参数值不正确怎么办？
A: 使用`ros2 param get`检查实际加载的值，使用`ros2 param set`动态调整

### Q: 如何备份参数配置？
A: 使用`ros2 param dump`命令保存当前参数到文件

## 📝 总结

- **当前使用**: `config/config.yaml` (通过launch文件自动加载)
- **参数声明**: 在`pose_controller.cpp`构造函数中
- **启动方式**: 推荐使用`ros2 launch aerogripper_control test.launch.py`
- **修改方式**: 编辑YAML文件后重新构建和启动
- **调试工具**: 使用`ros2 param`系列命令
