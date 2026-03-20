#pragma once

#include "mixer.h"

#include "px4_msgs/msg/actuator_motors.hpp"
#include "px4_msgs/msg/vehicle_attitude.hpp"
#include "px4_msgs/msg/sensor_combined.hpp"
#include "geometry_msgs/msg/vector3.hpp"
#include "geometry_msgs/msg/quaternion.hpp"
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
    
    void update_tether_force();
    
    // 获取观测结果
    Eigen::Matrix3d get_R_bt() const { return R_bt_; }
    Eigen::Vector3d get_ft_bodyframe() const { return ft_bodyframe_; }
    
    // 更新函数，供外部调用
    void update_attitude(const px4_msgs::msg::VehicleAttitude::SharedPtr msg);
    void update_imu(const px4_msgs::msg::SensorCombined::SharedPtr msg);
    void update_motor_feedback(const px4_msgs::msg::ActuatorMotors::SharedPtr msg);

private:
    void motor_feedback_callback(const px4_msgs::msg::ActuatorMotors::SharedPtr msg);
    void attitude_callback(const px4_msgs::msg::VehicleAttitude::SharedPtr msg);
    void imu_callback(const px4_msgs::msg::SensorCombined::SharedPtr msg);
    void update_thrust();

    // 状态
    Eigen::Matrix3d R_bt_;
    Eigen::Matrix3d R_bw_;
    Eigen::Vector3d acc_;
    Eigen::Vector3d acc_raw_;
    Eigen::Vector3d Tb_;    // 推力（机体系）
    Eigen::Vector3d tau_b_; // 力矩（机体系）
    Eigen::Vector4d throttle_ = Eigen::Vector4d::Zero();
    Eigen::Vector3d ft_bodyframe_;

    double mass_;
    double thrust_coeff_;
    
    // mixer
    std::shared_ptr<Mixer> mixer_;

    bool param_set_flag_;
};
