# 自定义消息类型使用指南

## 概述

我已经成功为你创建了一个自定义消息类型 `PoseReference`，它包含了五个类型的信息：

1. **位置** (position) - `geometry_msgs/Point`
2. **线性速度** (linear_velocity) - `geometry_msgs/Vector3`
3. **线性加速度** (linear_acceleration) - `geometry_msgs/Vector3`
4. **姿态四元数** (orientation) - `geometry_msgs/Quaternion`
5. **角速度** (angular_velocity) - `geometry_msgs/Vector3`

## 文件结构

```
workspace/aerogripper_ws/
├── src/
│   ├── aerogripper_msgs/           # 新的消息包
│   │   ├── msg/
│   │   │   └── PoseReference.msg   # 消息定义
│   │   ├── src/
│   │   │   └── test_publisher.cpp  # 测试发布者
│   │   ├── CMakeLists.txt
│   │   └── package.xml
│   └── aerogripper_control/        # 原始控制包
│       ├── include/
│       │   ├── pose_controller.h   # 已更新
│       │   └── mixer.h             # 已更新
│       └── src/
│           └── pose_controller.cpp # 已更新
```

## 使用方法

### 1. 编译消息包

```bash
cd ~/workspace/aerogripper_ws
colcon build --packages-select aerogripper_msgs
```

### 2. 编译控制包

```bash
colcon build --packages-select aerogripper_control
```

### 3. 运行测试

```bash
# 终端1：运行测试发布者
source install/setup.bash
ros2 run aerogripper_msgs test_publisher

# 终端2：查看消息
ros2 topic echo /pose_reference
```

### 4. 在你的代码中使用

```cpp
#include "aerogripper_msgs/msg/pose_reference.hpp"

// 创建订阅者
auto subscription = node->create_subscription<aerogripper_msgs::msg::PoseReference>(
    "pose_reference", 10,
    [](const aerogripper_msgs::msg::PoseReference::SharedPtr msg) {
        // 提取所有五个类型的信息
        Eigen::Vector3d pos(msg->position.x, msg->position.y, msg->position.z);
        Eigen::Vector3d vel(msg->linear_velocity.x, msg->linear_velocity.y, msg->linear_velocity.z);
        Eigen::Vector3d acc(msg->linear_acceleration.x, msg->linear_acceleration.y, msg->linear_acceleration.z);
        Eigen::Quaterniond q(msg->orientation.w, msg->orientation.x, msg->orientation.y, msg->orientation.z);
        Eigen::Vector3d omega(msg->angular_velocity.x, msg->angular_velocity.y, msg->angular_velocity.z);
        
        // 使用这些数据进行控制计算
    });
```

## 消息定义详情

```msg
# PoseReference.msg
geometry_msgs/Point position
geometry_msgs/Vector3 linear_velocity
geometry_msgs/Vector3 linear_acceleration
geometry_msgs/Quaternion orientation
geometry_msgs/Vector3 angular_velocity
```

## 优势

1. **类型安全**: 所有字段都有明确的类型定义
2. **完整性**: 一个消息包含所有需要的参考信息
3. **一致性**: 使用标准的ROS 2消息格式
4. **可扩展性**: 可以轻松添加新的字段
5. **工具支持**: 支持ROS 2的所有工具（rviz2, rqt等）

## 注意事项

- 确保在编译前先编译 `aerogripper_msgs` 包
- 在使用消息的包中添加对 `aerogripper_msgs` 的依赖
- 消息字段的顺序很重要，不要随意更改

## 下一步

你现在可以：
1. 使用这个新的消息类型替换原来的多个消息
2. 在 `pose_controller.cpp` 中实现 `reference_callback_custom` 函数
3. 创建发布者来发送这种消息类型
4. 根据需要扩展消息定义 