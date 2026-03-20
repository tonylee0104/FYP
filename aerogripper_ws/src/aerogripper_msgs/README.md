# Aerogripper Messages

这个包包含了用于aerogripper控制系统的自定义消息类型。

## 消息类型

### PoseReference.msg

`PoseReference`消息包含了完整的姿态参考信息，包括：

- `position`: 位置参考 (x, y, z)
- `linear_velocity`: 线性速度参考 (vx, vy, vz)  
- `linear_acceleration`: 线性加速度参考 (ax, ay, az)
- `orientation`: 姿态四元数参考 (w, x, y, z)
- `angular_velocity`: 角速度参考 (wx, wy, wz)

## 使用方法

### 1. 发布消息

```cpp
#include "aerogripper_msgs/msg/pose_reference.hpp"

// 创建发布者
auto publisher = node->create_publisher<aerogripper_msgs::msg::PoseReference>(
    "pose_reference", 10);

// 创建消息
auto msg = aerogripper_msgs::msg::PoseReference();
msg.position.x = 1.0;
msg.position.y = 2.0;
msg.position.z = 3.0;
msg.linear_velocity.x = 0.1;
msg.linear_velocity.y = 0.2;
msg.linear_velocity.z = 0.3;
msg.linear_acceleration.x = 0.01;
msg.linear_acceleration.y = 0.02;
msg.linear_acceleration.z = 0.03;
msg.orientation.w = 1.0;
msg.orientation.x = 0.0;
msg.orientation.y = 0.0;
msg.orientation.z = 0.0;
msg.angular_velocity.x = 0.1;
msg.angular_velocity.y = 0.2;
msg.angular_velocity.z = 0.3;

// 发布消息
publisher->publish(msg);
```

### 2. 订阅消息

```cpp
#include "aerogripper_msgs/msg/pose_reference.hpp"

// 创建订阅者
auto subscription = node->create_subscription<aerogripper_msgs::msg::PoseReference>(
    "pose_reference", 10,
    [](const aerogripper_msgs::msg::PoseReference::SharedPtr msg) {
        // 处理消息
        double x = msg->position.x;
        double vx = msg->linear_velocity.x;
        double ax = msg->linear_acceleration.x;
        // ... 其他处理
    });
```

## 编译

```bash
cd ~/workspace/aerogripper_ws
colcon build --packages-select aerogripper_msgs
```

## 依赖

- `geometry_msgs`: 用于基础几何类型
- `rclcpp`: ROS 2 C++客户端库 