# ControllerState 消息类型

## 概述

`ControllerState` 是一个自定义的ROS2消息类型，用于统一发布控制器的内部状态数据。相比为每个变量单独创建topic，这种方式更加方便和高效。

## 消息结构

```yaml
# 时间戳
timestamp: builtin_interfaces/Time

# 时间步长 (dt)
dt: float64

# 期望值
desired_acceleration: geometry_msgs/Vector3      # a_des
desired_angular_velocity: geometry_msgs/Vector3  # omega_des
desired_thrust: geometry_msgs/Vector3            # T_des
desired_torque: geometry_msgs/Vector3            # tau_des

# 当前状态
current_position: geometry_msgs/Point
current_velocity: geometry_msgs/Vector3
current_orientation: geometry_msgs/Quaternion
current_euler_angles: geometry_msgs/Vector3        # 欧拉角 (roll, pitch, yaw) [弧度]
current_angular_velocity: geometry_msgs/Vector3

# 参考值
reference_position: geometry_msgs/Point
reference_velocity: geometry_msgs/Vector3
reference_orientation: geometry_msgs/Quaternion
reference_euler_angles: geometry_msgs/Vector3      # 欧拉角 (roll, pitch, yaw) [弧度]

# 输出值
throttle_output: float64[4]                     # 四个电机的油门值

# 观测值
tether_force_bodyframe: geometry_msgs/Vector3   # 绳子拉力（机体坐标系）

## 欧拉角说明

消息中包含了姿态的四元数和欧拉角两种表示方式：

- **四元数** (`current_orientation`, `reference_orientation`): 避免万向锁问题，适合计算
- **欧拉角** (`current_euler_angles`, `reference_euler_angles`): 直观易理解，顺序为 ZYX (yaw-pitch-roll)
  - `x`: roll (横滚角) [弧度]
  - `y`: pitch (俯仰角) [弧度]  
  - `z`: yaw (偏航角) [弧度]

**注意**: 欧拉角使用 ZYX 顺序，这是航空领域常用的约定。

### 欧拉角使用示例

```cpp
// 在订阅者中处理欧拉角数据
void controller_state_callback(const aerogripper_msgs::msg::ControllerState::SharedPtr msg)
{
    // 获取当前姿态欧拉角
    double roll = msg->current_euler_angles.x;    // 横滚角 [弧度]
    double pitch = msg->current_euler_angles.y;   // 俯仰角 [弧度]
    double yaw = msg->current_euler_angles.z;     // 偏航角 [弧度]
    
    // 转换为度数（可选）
    double roll_deg = roll * 180.0 / M_PI;
    double pitch_deg = pitch * 180.0 / M_PI;
    double yaw_deg = yaw * 180.0 / M_PI;
    
    // 计算姿态误差
    double roll_error = msg->reference_euler_angles.x - roll;
    double pitch_error = msg->reference_euler_angles.y - pitch;
    double yaw_error = msg->reference_euler_angles.z - yaw;
    
    // 处理偏航角的周期性（-π 到 π）
    while (yaw_error > M_PI) yaw_error -= 2 * M_PI;
    while (yaw_error < -M_PI) yaw_error += 2 * M_PI;
    
    // 使用误差进行控制...
}
```
```

## 使用方法

### 1. 发布消息

在您的控制器代码中，可以这样发布消息：

```cpp
#include "aerogripper_msgs/msg/controller_state.hpp"

// 创建发布者
auto controller_state_pub_ = this->create_publisher<aerogripper_msgs::msg::ControllerState>(
    "/aerogripper/controller_state", 10);

// 创建并填充消息
auto msg = std::make_unique<aerogripper_msgs::msg::ControllerState>();
msg->timestamp = this->now();
msg->dt = dt;
msg->desired_acceleration.x = a_des(0);
msg->desired_acceleration.y = a_des(1);
msg->desired_acceleration.z = a_des(2);
// ... 设置其他字段

// 发布消息
controller_state_pub_->publish(*msg);
```

### 2. 订阅消息

在其他节点中，可以这样订阅消息：

```cpp
#include "aerogripper_msgs/msg/controller_state.hpp"

// 创建订阅者
auto controller_state_sub_ = this->create_subscription<aerogripper_msgs::msg::ControllerState>(
    "/aerogripper/controller_state", 10,
    std::bind(&YourNode::controller_state_callback, this, std::placeholders::_1));

// 回调函数
void controller_state_callback(const aerogripper_msgs::msg::ControllerState::SharedPtr msg)
{
    // 处理接收到的数据
    double dt = msg->dt;
    double ax = msg->desired_acceleration.x;
    double ay = msg->desired_acceleration.y;
    double az = msg->desired_acceleration.z;
    
    // ... 处理其他数据
}
```

### 3. 命令行查看

使用ROS2命令行工具查看消息：

```bash
# 查看topic列表
ros2 topic list

# 查看消息内容
ros2 topic echo /aerogripper/controller_state

# 查看消息类型
ros2 topic info /aerogripper/controller_state

# 查看消息频率
ros2 topic hz /aerogripper/controller_state
```

## 优势

1. **代码维护性** - 只需要维护一个发布者和一个消息类型
2. **数据一致性** - 所有相关数据在同一个消息中，时间戳一致
3. **网络效率** - 减少topic数量，降低ROS2系统的开销
4. **调试方便** - 可以用一个工具查看所有相关数据
5. **扩展性好** - 需要添加新字段时，只需修改消息定义

## 测试

项目包含了一个测试节点 `test_controller_state`，可以用来验证消息类型是否正常工作：

```bash
# 构建项目
cd workspace/aerogripper_ws
colcon build --packages-select aerogripper_msgs

# 运行测试发布者
ros2 run aerogripper_msgs test_controller_state

# 在另一个终端查看消息
ros2 topic echo /aerogripper/controller_state

# 或者使用Python脚本查看（更友好的格式）
ros2 run aerogripper_msgs controller_state_subscriber

# 快速演示（运行演示脚本）
./scripts/demo_controller_state.sh
```

## 注意事项

1. 确保在 `CMakeLists.txt` 中包含了新的消息类型
2. 消息发布频率建议与控制循环频率保持一致
3. 如果某些字段不需要，可以设置为默认值或NaN
4. 时间戳使用 `this->now()` 确保准确性
