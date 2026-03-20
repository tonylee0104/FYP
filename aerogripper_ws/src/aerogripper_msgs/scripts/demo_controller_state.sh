#!/bin/bash

echo "=========================================="
echo "ControllerState 消息类型演示"
echo "=========================================="

echo ""
echo "1. 构建项目..."
cd /home/nvidia/workspace/aerogripper_ws
colcon build --packages-select aerogripper_msgs

echo ""
echo "2. 启动测试发布者..."
echo "   在第一个终端中运行:"
echo "   ros2 run aerogripper_msgs test_controller_state"
echo ""

echo "3. 查看消息内容..."
echo "   在第二个终端中运行以下命令之一:"
echo "   # 原始消息格式:"
echo "   ros2 topic echo /aerogripper/controller_state"
echo ""
echo "   # 友好格式 (Python脚本):"
echo "   ros2 run aerogripper_msgs controller_state_subscriber"
echo ""

echo "4. 查看消息信息..."
echo "   在第三个终端中运行:"
echo "   ros2 topic info /aerogripper/controller_state"
echo "   ros2 topic hz /aerogripper/controller_state"
echo ""

echo "5. 查看话题列表..."
echo "   ros2 topic list | grep controller"
echo ""

echo "=========================================="
echo "演示完成！"
echo "=========================================="
