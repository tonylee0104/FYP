#pragma once

#include "geometry_msgs/msg/quaternion.hpp"
#include <Eigen/Dense>
#include <functional>
#include <iostream>
using namespace std;

class Mixer
{
public:
    Mixer();

    // 设置参数函数
    void set_params(double thrust_arm_length, double gravity_arm_length, 
                   double torque_coeff, double rpm_coeff, double thrust_coeff,
                   double yaw_weight, double attitude_weight, double thrust_weight);

    // mix
    Eigen::Matrix<double,4,1> mix(const Eigen::Matrix<double,6,1>& u) const;
    Eigen::Matrix<double,5,1> mix_enhanced(const Eigen::Matrix<double,6,1>& u) const;

    // 控制分配：输入4x1电机量，输出6x1推力和力矩
    Eigen::Matrix<double,6,1> control_allocation(const Eigen::Matrix<double,4,1>& motors_thrust) const;
    Eigen::Matrix<double,6,1> control_allocation_enhanced(const Eigen::Matrix<double,4,1>& motors_thrust, double w) const;

    // throttle -> thrust
    // Eigen::Matrix<double,4,1> throttle2rpm(const Eigen::Matrix<double,4,1>& throttle) const;
    // Eigen::Matrix<double,4,1> rpm2throttle(const Eigen::Matrix<double,4,1>& rpm) const;
    // Eigen::Matrix<double,4,1> rpm2thrust(const Eigen::Matrix<double,4,1>& rpm) const;
    // Eigen::Matrix<double,4,1> thrust2rpm(const Eigen::Matrix<double,4,1>& thrust) const;
    Eigen::Matrix<double,4,1> throttle2thrust(const Eigen::Matrix<double,4,1>& throttle) const;
    Eigen::Matrix<double,4,1> thrust2throttle(const Eigen::Matrix<double,4,1>& thrust) const;
    
    void update_parameters(const Eigen::Matrix3d& R_bt, const Eigen::Vector3d& ft_bodyframe);
    // Eigen::Matrix3d get_R_bt();
    // Eigen::Vector3d get_ft_bodyframe();

    
    Eigen::Matrix<double,4,6> mix_matrix_;//todo 挪回private

private:
    void update_enhanced_matrix(const Eigen::Matrix3d& R_bt);
    void update_ft_bodyframe(const Eigen::Vector3d& ft_bodyframe);

    Eigen::Matrix<double,6,4> allocation_matrix_;
    Eigen::Matrix<double,5,6> mix_matrix_enhanced_;
    Eigen::Matrix<double,6,5> allocation_matrix_enhanced_;
    Eigen::Matrix<double,6,1> allocation_matrix_w_;

    Eigen::Matrix3d R_bt_;
    Eigen::Vector3d ft_bodyframe_;
    
    // 配置参数
    double thrust_arm_length_;    // 推力臂长
    double gravity_arm_length_;   // 重力臂长
    double torque_coeff_;         // 力矩常数
    double yaw_weight_;           // yaw控制权重
    double attitude_weight_;      // 姿态控制权重
    double thrust_weight_;        // 推力控制权重
    
    // 系数
    double rpm_coeff_;
    double thrust_coeff_;

    bool param_set_flag_;
};