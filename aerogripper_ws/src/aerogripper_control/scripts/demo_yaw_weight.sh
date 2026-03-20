#!/bin/bash

echo "=========================================="
echo "Yaw控制权重参数演示"
echo "=========================================="

echo ""
echo "1. 构建项目..."
cd /home/nvidia/workspace/aerogripper_ws
colcon build --packages-select aerogripper_control

echo ""
echo "2. 运行Python测试脚本..."
echo "   在第一个终端中运行:"
echo "   cd /home/nvidia/workspace/aerogripper_ws/src/aerogripper_control/scripts"
echo "   python3 test_yaw_weight.py"
echo ""

echo "3. 启动控制器节点..."
echo "   在第二个终端中运行:"
echo "   cd /home/nvidia/workspace/aerogripper_ws"
echo "   source install/setup.bash"
echo "   ros2 run aerogripper_control pose_controller"
echo ""

echo "4. 查看参数..."
echo "   在第三个终端中运行:"
echo "   ros2 param list /aerogripper_control"
echo "   ros2 param get /aerogripper_control yaw_weight"
echo ""

echo "5. 动态调整yaw_weight参数..."
echo "   # 设置为0.1 (降低yaw敏感度)"
echo "   ros2 param set /aerogripper_control yaw_weight 0.1"
echo ""
echo "   # 设置为0.05 (进一步降低yaw敏感度)"
echo "   ros2 param set /aerogripper_control yaw_weight 0.05"
echo ""
echo "   # 设置为0.5 (中等yaw敏感度)"
echo "   ros2 param set /aerogripper_control yaw_weight 0.5"
echo ""

echo "6. 监控控制效果..."
echo "   # 查看控制器状态"
echo "   ros2 topic echo /aerogripper/controller_state"
echo ""

echo "=========================================="
echo "参数说明:"
echo "=========================================="
echo "yaw_weight: Yaw控制权重参数"
echo "- 值越小，yaw控制越不敏感 (推荐: 0.05-0.2)"
echo "- 值越大，yaw控制越敏感 (默认: 1.0)"
echo "- 对于torque_coeff=0.01的小型无人机，建议设置为0.1"
echo ""

echo "=========================================="
echo "演示完成！"
echo "=========================================="
