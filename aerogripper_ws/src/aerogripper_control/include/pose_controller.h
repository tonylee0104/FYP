#pragma once

#include "mixer.h"
#include "tether_observer.h"
#include "aerogripper_msgs/msg/pose_reference.hpp"
#include "aerogripper_msgs/msg/controller_state.hpp"

#include "px4_msgs/msg/vehicle_local_position.hpp"
#include "px4_msgs/msg/vehicle_attitude.hpp"
#include "px4_msgs/msg/vehicle_angular_velocity.hpp"
#include "px4_msgs/msg/actuator_motors.hpp"
#include "px4_msgs/msg/vehicle_command.hpp"
#include "px4_msgs/msg/offboard_control_mode.hpp"
#include "geometry_msgs/msg/pose.hpp"
#include "px4_msgs/msg/sensor_combined.hpp"
#include <vector>
#include <rclcpp/rclcpp.hpp>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <chrono>
#include <iomanip>  // 用于格式化输出
#include <iostream>
using namespace std;
using namespace px4_msgs::msg;

class PoseController : public rclcpp::Node
{
public:
    PoseController();
    
    // 设置参数函数
    void set_params(const std::vector<double>& kp_pos, 
                   const std::vector<double>& kv_pos,
                   const std::vector<double>& k_R,
                   const std::vector<double>& kp_att,
                   const std::vector<double>& ki_att,
                   const std::vector<double>& kd_att,
                   double mass,
                   const std::vector<double>& inertia,
                   double thrust_coeff,
                   double thrust_arm_length,
                   double gravity_arm_length,
                   double torque_coeff,
                   double rpm_coeff,
                   double yaw_weight,
                   double attitude_weight,
                   double thrust_weight,
                   double compensation_factor,
                   double filter_coefficient,
                   double max_throttle_delta,
                   double throttle_threshold);

private:
    enum class HoverPhase {
        WaitHome,
        Delay,
        GoToHover,
        HoldHover,
        ReturnHome,
        HoldHome
    };
    // 从yaml文件加载参数并设置
    void load_and_set_parameters();
    
    // 辅助函数：格式化输出向量
    void print_vector(const std::string& name, const Eigen::Vector3d& vec);
    void print_vector(const std::string& name, const Eigen::Vector4d& vec);
    void print_vector(const std::string& name, const Eigen::Matrix<double, 6, 1>& vec);
    
    // Offboard和解锁相关方法
    void enable_offboard_mode();
    void arm_vehicle();
    void offboard_control_loop();
    
    // 静态QoS配置方法
    static rclcpp::QoS get_px4_compatible_qos();
    
    // 回调
    void position_callback(const px4_msgs::msg::VehicleLocalPosition::SharedPtr msg);
    // void reference_callback(const geometry_msgs::msg::Pose::SharedPtr msg);
    void reference_callback_custom(const aerogripper_msgs::msg::PoseReference::SharedPtr msg);
    void attitude_callback(const px4_msgs::msg::VehicleAttitude::SharedPtr msg);
    void imu_callback(const px4_msgs::msg::SensorCombined::SharedPtr msg);
    void motors_throttle_pub(float motor1, float motor2, float motor3, float motor4);
    void control_loop();
    
    // 新增：发布控制器状态消息
    void publish_controller_state(double dt, const Eigen::Vector3d& a_des, 
                                const Eigen::Vector3d& omega_des, 
                                const Eigen::Vector3d& T_des, 
                                const Eigen::Vector3d& tau_des,
                                const Eigen::Vector3d& ref_pos,
                                const Eigen::Quaterniond& ref_q);

    // 订阅
    rclcpp::Subscription<aerogripper_msgs::msg::PoseReference>::SharedPtr reference_custom_sub_;
    rclcpp::Subscription<px4_msgs::msg::VehicleAttitude>::SharedPtr attitude_sub_;
    rclcpp::Subscription<px4_msgs::msg::SensorCombined>::SharedPtr imu_sub_;
    // 发布
    rclcpp::Publisher<px4_msgs::msg::ActuatorMotors>::SharedPtr motors_throttle_pub_;
    rclcpp::Publisher<px4_msgs::msg::VehicleCommand>::SharedPtr vehicle_command_pub_;
    rclcpp::Publisher<px4_msgs::msg::OffboardControlMode>::SharedPtr offboard_control_mode_pub_;
    
    // 新增：控制器状态发布者
    rclcpp::Publisher<aerogripper_msgs::msg::ControllerState>::SharedPtr controller_state_pub_;

    // 定时器
    rclcpp::TimerBase::SharedPtr timer_;
    rclcpp::TimerBase::SharedPtr offboard_timer_;

    // Offboard和解锁相关
    bool offboard_mode_enabled_;
    bool vehicle_armed_;
    int offboard_counter_;
    int arm_counter_;
    
    // 初始姿态相关
    Eigen::Quaterniond q_init_FRD_;      // 初始FRD四元数
    bool q_init_flag_;                   // 初始姿态标志

    // 参考与状态
    Eigen::Vector3d ref_pos_;
    Eigen::Vector3d ref_vel_;
    Eigen::Vector3d ref_acc_;
    Eigen::Quaterniond ref_q_;
    Eigen::Vector3d ref_omega_;
    Eigen::Vector3d cur_pos_;
    Eigen::Vector3d cur_vel_;
    Eigen::Vector3d acc_raw_;
    Eigen::Vector3d cur_acc_; // IMU原始加速度
    Eigen::Quaterniond cur_q_;
    Eigen::Vector3d cur_omega_;

    // PID参数
    Eigen::Vector3d kp_pos_, kv_pos_;
    Eigen::Vector3d k_R_, kp_att_, ki_att_, kd_att_;

    // PID内部值
    Eigen::Vector3d omega_err_int_;
    Eigen::Vector3d omega_last_err_;

    double last_time_;

    // 集成的组件
    std::shared_ptr<Mixer> mixer_;
    std::shared_ptr<TetherObserver> tether_observer_;

    Eigen::Vector4d throttle_;
    Eigen::VectorXd u_;
    Eigen::Matrix3d R_bw_;
    Eigen::Matrix3d R_bt_;
    Eigen::Vector3d ft_bodyframe_;

    // 物理参数
    double mass_;
    Eigen::Vector3d inertia_;
    double compensation_factor_;  // 补偿因子参数
    
    // 油门滤波和限制参数
    double filter_coefficient_;  // 低通滤波系数
    double max_throttle_delta_;  // 最大油门变化率
    double throttle_threshold_;  // 油门阈值，用于防止控制饱和
    Eigen::Vector4d throttle_filtered_;  // 滤波后的油门值
    Eigen::Vector4d throttle_last_;      // 上一时刻的油门值

    bool param_set_flag_;

    // Hover and return sequence
    bool home_set_;
    Eigen::Vector3d home_pos_;
    Eigen::Vector3d hover_target_;
    double start_delay_s_;
    double travel_time_s_;
    double hover_time_s_;
    double return_time_s_;
    double phase_start_time_s_;
    HoverPhase hover_phase_;
};