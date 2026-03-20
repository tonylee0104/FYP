#include "aerogripper_msgs/msg/controller_state.hpp"
#include "rclcpp/rclcpp.hpp"
#include <chrono>

using namespace std::chrono_literals;

class TestControllerStatePublisher : public rclcpp::Node
{
public:
    TestControllerStatePublisher() : Node("test_controller_state_publisher")
    {
        publisher_ = this->create_publisher<aerogripper_msgs::msg::ControllerState>(
            "/aerogripper/controller_state", 10);
        
        timer_ = this->create_wall_timer(
            100ms, std::bind(&TestControllerStatePublisher::timer_callback, this));
        
        RCLCPP_INFO(this->get_logger(), "Test ControllerState publisher started");
    }

private:
    void timer_callback()
    {
        auto msg = std::make_unique<aerogripper_msgs::msg::ControllerState>();
        
        // 设置时间戳
        msg->timestamp = this->now();
        
        // 设置测试数据
        msg->dt = 0.003; // 3ms
        
        // 期望加速度
        msg->desired_acceleration.x = 1.0;
        msg->desired_acceleration.y = 2.0;
        msg->desired_acceleration.z = 3.0;
        
        // 期望角速度
        msg->desired_angular_velocity.x = 0.1;
        msg->desired_angular_velocity.y = 0.2;
        msg->desired_angular_velocity.z = 0.3;
        
        // 期望推力
        msg->desired_thrust.x = 0.5;
        msg->desired_thrust.y = 0.6;
        msg->desired_thrust.z = 0.7;
        
        // 期望力矩
        msg->desired_torque.x = 0.01;
        msg->desired_torque.y = 0.02;
        msg->desired_torque.z = 0.03;
        
        // 当前状态
        msg->current_position.x = 0.0;
        msg->current_position.y = 0.0;
        msg->current_position.z = 0.0;
        
        msg->current_velocity.x = 0.0;
        msg->current_velocity.y = 0.0;
        msg->current_velocity.z = 0.0;
        
        msg->current_orientation.w = 1.0;
        msg->current_orientation.x = 0.0;
        msg->current_orientation.y = 0.0;
        msg->current_orientation.z = 0.0;
        
        // 当前姿态欧拉角 (roll, pitch, yaw)
        msg->current_euler_angles.x = 0.0; // roll
        msg->current_euler_angles.y = 0.0; // pitch
        msg->current_euler_angles.z = 0.0; // yaw
        
        msg->current_angular_velocity.x = 0.0;
        msg->current_angular_velocity.y = 0.0;
        msg->current_angular_velocity.z = 0.0;
        
        // 参考值
        msg->reference_position.x = 1.0;
        msg->reference_position.y = 2.0;
        msg->reference_position.z = 3.0;
        
        msg->reference_velocity.x = 0.1;
        msg->reference_velocity.y = 0.2;
        msg->reference_velocity.z = 0.3;
        
        msg->reference_orientation.w = 1.0;
        msg->reference_orientation.x = 0.0;
        msg->reference_orientation.y = 0.0;
        msg->reference_orientation.z = 0.0;
        
        // 参考姿态欧拉角 (roll, pitch, yaw)
        msg->reference_euler_angles.x = 0.1; // roll
        msg->reference_euler_angles.y = 0.2; // pitch
        msg->reference_euler_angles.z = 0.3; // yaw
        
        // 油门输出
        msg->throttle_output[0] = 0.5;
        msg->throttle_output[1] = 0.6;
        msg->throttle_output[2] = 0.7;
        msg->throttle_output[3] = 0.8;
        
        // 绳子拉力
        msg->tether_force_bodyframe.x = 0.1;
        msg->tether_force_bodyframe.y = 0.2;
        msg->tether_force_bodyframe.z = 0.3;
        
        // 发布消息
        publisher_->publish(*msg);
        
        RCLCPP_INFO(this->get_logger(), "Published ControllerState message");
    }
    
    rclcpp::Publisher<aerogripper_msgs::msg::ControllerState>::SharedPtr publisher_;
    rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<TestControllerStatePublisher>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
