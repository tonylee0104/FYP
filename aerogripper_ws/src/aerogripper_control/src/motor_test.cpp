#include <rclcpp/rclcpp.hpp>
#include "px4_msgs/msg/actuator_motors.hpp"
#include "px4_msgs/msg/offboard_control_mode.hpp"
#include "px4_msgs/msg/vehicle_command.hpp"
#include <chrono>
#include <cmath> // for std::nanf

using namespace std::chrono_literals;
using namespace px4_msgs::msg;

class MotorTestNode : public rclcpp::Node {
public:
    MotorTestNode() : Node("motor_test_node_class"), setpoint_counter_(0) {
        offboard_control_mode_pub_ = this->create_publisher<OffboardControlMode>("/fmu/in/offboard_control_mode", 10);
        actuator_motors_pub_ = this->create_publisher<ActuatorMotors>("/fmu/in/actuator_motors", 10);
        vehicle_command_pub_ = this->create_publisher<VehicleCommand>("/fmu/in/vehicle_command", 10);

        timer_ = this->create_wall_timer(100ms, std::bind(&MotorTestNode::timer_callback, this));
    }

private:
    rclcpp::TimerBase::SharedPtr timer_;
    rclcpp::Publisher<OffboardControlMode>::SharedPtr offboard_control_mode_pub_;
    rclcpp::Publisher<ActuatorMotors>::SharedPtr actuator_motors_pub_;
    rclcpp::Publisher<VehicleCommand>::SharedPtr vehicle_command_pub_;
    uint64_t setpoint_counter_;

    void timer_callback() {
        // 1. 持续发布 OffboardControlMode
        publish_offboard_control_mode();
        // 2. 持续发布 ActuatorMotors
        if (setpoint_counter_ > 220) {
            publish_actuator_motors(0.0f, 0.0f, 0.0f, 0.0f);
            RCLCPP_INFO(this->get_logger(), "已关闭所有电机，准备退出");
            rclcpp::shutdown();
        }

        for (int i = 0; i <= 10; i++) {
            if (setpoint_counter_ >= 2 * 10 * i && setpoint_counter_ < 2 * (10 * i + 10)) {
                publish_actuator_motors(i / 10.0f, i / 10.0f, i / 10.0f, i / 10.0f);
                RCLCPP_INFO(this->get_logger(), "motor throttle: %f", i / 10.0f);
            }
        }

        // 3. 到第10次时切换 Offboard 模式并解锁
        if (setpoint_counter_ >= 10) {
            publish_vehicle_command(VehicleCommand::VEHICLE_CMD_DO_SET_MODE, 1, 6); // 切换 Offboard
            RCLCPP_INFO(this->get_logger(), "已发送切换Offboard模式命令");
            publish_vehicle_command(VehicleCommand::VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0); // 解锁
            RCLCPP_INFO(this->get_logger(), "已发送解锁命令");
        }
        setpoint_counter_++;
    }

    void publish_offboard_control_mode() {
        OffboardControlMode msg;
        msg.position = false;
        msg.velocity = false;
        msg.acceleration = false;
        msg.attitude = false;
        msg.body_rate = false;
        msg.thrust_and_torque = false;
        msg.direct_actuator = true;
        msg.timestamp = this->now().nanoseconds() / 1000;
        offboard_control_mode_pub_->publish(msg);
    }

    void publish_actuator_motors(float motor1, float motor2, float motor3, float motor4) {
        ActuatorMotors msg;
        msg.control = {
            motor1, motor2, motor3, motor4,
            std::nanf("1"), std::nanf("1"), std::nanf("1"), std::nanf("1"),
            std::nanf("1"), std::nanf("1"), std::nanf("1"), std::nanf("1")
        };
        msg.reversible_flags = 0;
        msg.timestamp = this->now().nanoseconds() / 1000;
        msg.timestamp_sample = msg.timestamp;
        actuator_motors_pub_->publish(msg);
    }

    void publish_vehicle_command(uint16_t command, float param1 = 0.0, float param2 = 0.0) {
        VehicleCommand msg;
        msg.param1 = param1;
        msg.param2 = param2;
        msg.command = command;
        msg.target_system = 1;
        msg.target_component = 1;
        msg.source_system = 1;
        msg.source_component = 1;
        msg.from_external = true;
        msg.timestamp = this->now().nanoseconds() / 1000;
        vehicle_command_pub_->publish(msg);
        RCLCPP_INFO(this->get_logger(), "vehicle command send: %u", command);
    }
};

int main(int argc, char* argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<MotorTestNode>());
    rclcpp::shutdown();
    return 0;
} 




