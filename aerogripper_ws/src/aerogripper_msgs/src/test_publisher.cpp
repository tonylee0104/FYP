#include "rclcpp/rclcpp.hpp"
#include "aerogripper_msgs/msg/pose_reference.hpp"
#include <chrono>

using namespace std::chrono_literals;

class TestPublisher : public rclcpp::Node
{
public:
    TestPublisher() : Node("test_publisher")
    {
        publisher_ = this->create_publisher<aerogripper_msgs::msg::PoseReference>(
            "pose_reference", 10);
        
        timer_ = this->create_wall_timer(
            1000ms, std::bind(&TestPublisher::timer_callback, this));
        
        RCLCPP_INFO(this->get_logger(), "Test publisher started");
    }

private:
    void timer_callback()
    {
        auto msg = aerogripper_msgs::msg::PoseReference();
        
        // 设置位置
        msg.position.x = 1.0;
        msg.position.y = 2.0;
        msg.position.z = 3.0;
        
        // 设置速度
        msg.linear_velocity.x = 0.1;
        msg.linear_velocity.y = 0.2;
        msg.linear_velocity.z = 0.3;
        
        // 设置加速度
        msg.linear_acceleration.x = 0.01;
        msg.linear_acceleration.y = 0.02;
        msg.linear_acceleration.z = 0.03;
        
        // 设置姿态四元数
        msg.orientation.w = 1.0;
        msg.orientation.x = 0.0;
        msg.orientation.y = 0.0;
        msg.orientation.z = 0.0;
        
        // 设置角速度
        msg.angular_velocity.x = 0.1;
        msg.angular_velocity.y = 0.2;
        msg.angular_velocity.z = 0.3;
        
        publisher_->publish(msg);
        RCLCPP_INFO(this->get_logger(), "Published PoseReference message");
    }
    
    rclcpp::Publisher<aerogripper_msgs::msg::PoseReference>::SharedPtr publisher_;
    rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<TestPublisher>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
} 