// todo 离心力观测！！！！！！！！！！！

#pragma once

#include "mixer.h"

#include "px4_msgs/msg/actuator_motors.hpp"
#include "px4_msgs/msg/vehicle_attitude.hpp"
#include "px4_msgs/msg/sensor_combined.hpp"
#include "geometry_msgs/msg/vector3.hpp"
#include "geometry_msgs/msg/quaternion.hpp"
#include <rclcpp/rclcpp.hpp>
#include <Eigen/Dense>
#include <vector>
#include <cmath>
#include <iostream>
using namespace std;

class TetherObserver
{
public:
    TetherObserver(std::shared_ptr<Mixer> mixer);
    
    // 设置参数函数
    void set_params(double mass, double thrust_coeff);
    
    void update_tether_observation();
    
    // 获取观测结果
    Eigen::Matrix3d get_R_bt() const { return R_bt_; }
    Eigen::Vector3d get_ft_bodyframe() const { return ft_bodyframe_; }
    Eigen::Vector3d get_position() const { return position_; }
    Eigen::Vector3d get_velocity() const { return velocity_; }
    
    // 更新函数，供外部调用
    void update_attitude(const Eigen::Matrix3d& R_bw, const Eigen::Quaterniond& q);
    void update_imu(const Eigen::Vector3d& acc, const Eigen::Vector3d& omega);
    void update_throttle(const Eigen::Vector4d& throttle);

private:
    void motor_feedback_callback(const px4_msgs::msg::ActuatorMotors::SharedPtr msg);
    void attitude_callback(const px4_msgs::msg::VehicleAttitude::SharedPtr msg);
    void imu_callback(const px4_msgs::msg::SensorCombined::SharedPtr msg);
    void update_thrust();

    // 状态
    Eigen::Matrix3d R_bt_;
    Eigen::Matrix3d R_bw_;
    Eigen::Quaterniond q_;  // body系
    Eigen::Vector3d acc_;   // body系
    Eigen::Vector3d omega_; // body系
    Eigen::Vector3d Tb_;    // 推力 body系
    Eigen::Vector3d tau_b_; // 力矩 body系
    Eigen::Vector4d throttle_;
    Eigen::Vector3d ft_bodyframe_;
    Eigen::Vector3d position_; // 世界系
    Eigen::Vector3d last_position_; // 世界系
    Eigen::Vector3d velocity_;      // 世界系

    double mass_;
    double thrust_coeff_;
    double tether_length_;
    double last_time_;
    // mixer
    std::shared_ptr<Mixer> mixer_;

    bool param_set_flag_;
};