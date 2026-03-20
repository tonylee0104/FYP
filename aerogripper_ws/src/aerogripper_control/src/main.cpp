#include "pose_controller.h"
#include <memory>
#include <rclcpp/rclcpp.hpp>
#include <iostream>
using namespace std;

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);

    // 创建PoseController实例，它会自己从yaml文件读取参数
    auto pose_controller = std::make_shared<PoseController>();

    // 运行PoseController节点
    rclcpp::spin(pose_controller);
    rclcpp::shutdown();
    return 0;
}