#include "pose_controller.h"
#include <algorithm>

// 静态QoS配置方法实现
rclcpp::QoS PoseController::get_px4_compatible_qos() {
    return rclcpp::QoS(1)
        .best_effort()
        .transient_local()
        .keep_last(1);
}

PoseController::PoseController() : 
    Node("aerogripper_control"),
    // 1. 参考量
    ref_pos_(Eigen::Vector3d::Zero()),
    ref_vel_(Eigen::Vector3d::Zero()),
    ref_acc_(Eigen::Vector3d::Zero()),
    ref_q_(1,0,0,0),
    ref_omega_(Eigen::Vector3d::Zero()),

    // 2. 当前状态量
    cur_pos_(Eigen::Vector3d::Zero()),
    cur_q_(1,0,0,0),
    R_bw_(Eigen::Matrix3d::Identity()),
    cur_vel_(Eigen::Vector3d::Zero()),
    cur_omega_(Eigen::Vector3d::Zero()),
    cur_acc_(Eigen::Vector3d::Zero()),

    // 3. PID相关变量
    omega_err_int_(Eigen::Vector3d::Zero()),
    omega_last_err_(Eigen::Vector3d::Zero()),

    // 4. 时间戳
    last_time_(this->now().seconds()),

    // 5. 控制量初始化
    u_(Eigen::VectorXd::Zero(5)), // 控制量初始化为零
    throttle_(Eigen::Vector4d::Zero()), // 控制量初始化为零
    throttle_filtered_(Eigen::Vector4d::Zero()), // 滤波后的油门值初始化为零
    throttle_last_(Eigen::Vector4d::Zero()),     // 上一时刻的油门值初始化为零
    
    // 6. Offboard和解锁状态初始化
    offboard_mode_enabled_(false),
    vehicle_armed_(false),
    offboard_counter_(0),
    arm_counter_(0),
    
    // 7. 初始姿态初始化
    q_init_FRD_(1, 0, 0, 0),  // 单位四元数
    q_init_flag_(false),

    // 8. Hover and return sequence
    home_set_(false),
    home_pos_(Eigen::Vector3d::Zero()),
    hover_target_(Eigen::Vector3d(0.0, 0.0, -0.25)),
    start_delay_s_(1.0),
    travel_time_s_(3.0),
    hover_time_s_(5.0),
    return_time_s_(3.0),
    phase_start_time_s_(0.0),
    hover_phase_(HoverPhase::WaitHome)
{
    // 声明参数（这些是默认值，会被yaml文件覆盖）
    this->declare_parameter("kp_pos", std::vector<double>{2.0, 2.0, 3.0});
    this->declare_parameter("kv_pos", std::vector<double>{0.6, 0.6, 0.8});
    this->declare_parameter("k_R", std::vector<double>{1.0, 1.0, 1.0});
    this->declare_parameter("kp_att", std::vector<double>{8.0, 8.0, 5.0});
    this->declare_parameter("ki_att", std::vector<double>{0.0, 0.0, 0.0});
    this->declare_parameter("kd_att", std::vector<double>{0.1, 0.1, 0.1});
    this->declare_parameter("mass", 0.110);
    this->declare_parameter("inertia", std::vector<double>{0.01, 0.01, 0.005});
    this->declare_parameter("thrust_coeff", 1.99);
    this->declare_parameter("thrust_arm_length", 0.055);
    this->declare_parameter("gravity_arm_length", 0.045);
    this->declare_parameter("torque_coeff", 0.01);
    this->declare_parameter("rpm_coeff", 1.0);
    this->declare_parameter("yaw_weight", 1.0);
    this->declare_parameter("attitude_weight", 1.0);
    this->declare_parameter("thrust_weight", 1.0);
    this->declare_parameter("throttle_threshold", 0.3);
    this->declare_parameter("compensation_factor", 1.5);
    this->declare_parameter("filter_coefficient", 0.3);
    this->declare_parameter("max_throttle_delta", 0.1);
    this->declare_parameter("hover_target", std::vector<double>{0.0, 0.0, -0.25});
    this->declare_parameter("start_delay_s", 1.0);
    this->declare_parameter("travel_time_s", 3.0);
    this->declare_parameter("hover_time_s", 5.0);
    this->declare_parameter("return_time_s", 3.0);

    // 等待一小段时间确保参数加载完成
    std::this_thread::sleep_for(std::chrono::milliseconds(500));

    // 创建集成的组件
    mixer_ = std::make_shared<Mixer>();
    tether_observer_ = std::make_shared<TetherObserver>(mixer_);

    // 从yaml文件加载参数并设置
    load_and_set_parameters();

    reference_custom_sub_ = this->create_subscription<aerogripper_msgs::msg::PoseReference>(
        "/reference_control_output", 1, std::bind(&PoseController::reference_callback_custom, this, std::placeholders::_1));
    
    attitude_sub_ = this->create_subscription<px4_msgs::msg::VehicleAttitude>(
        "/fmu/out/vehicle_attitude", get_px4_compatible_qos(), std::bind(&PoseController::attitude_callback, this, std::placeholders::_1));
    imu_sub_ = this->create_subscription<px4_msgs::msg::SensorCombined>(
        "/fmu/out/sensor_combined", get_px4_compatible_qos(), std::bind(&PoseController::imu_callback, this, std::placeholders::_1));

    motors_throttle_pub_ = this->create_publisher<px4_msgs::msg::ActuatorMotors>("/fmu/in/actuator_motors", 0);
    vehicle_command_pub_ = this->create_publisher<px4_msgs::msg::VehicleCommand>("/fmu/in/vehicle_command", 0);
    offboard_control_mode_pub_ = this->create_publisher<px4_msgs::msg::OffboardControlMode>("/fmu/in/offboard_control_mode", 0);
    
    // 新增：控制器状态发布者
    controller_state_pub_ = this->create_publisher<aerogripper_msgs::msg::ControllerState>("/aerogripper/controller_state", 10);
    
    timer_ = this->create_wall_timer(std::chrono::milliseconds(3), std::bind(&PoseController::control_loop, this));
    offboard_timer_ = this->create_wall_timer(std::chrono::milliseconds(100), std::bind(&PoseController::offboard_control_loop, this));
}

void PoseController::set_params(const std::vector<double>& kp_pos, 
                               const std::vector<double>& kv_pos,
                               const std::vector<double>& k_R,
                               const std::vector<double>& kp_att,
                               const std::vector<double>& ki_att,
                               const std::vector<double>& kd_att,
                               const double mass,
                               const std::vector<double>& inertia,
                               const double thrust_coeff,
                               const double thrust_arm_length,
                               const double gravity_arm_length,
                               const double torque_coeff,
                               const double rpm_coeff,
                               const double yaw_weight,
                               const double attitude_weight,
                               const double thrust_weight,
                               const double compensation_factor,
                               const double filter_coefficient,
                               const double max_throttle_delta,
                               const double throttle_threshold)
{
    // 设置PID参数
    kp_pos_ = Eigen::Vector3d(kp_pos[0], kp_pos[1], kp_pos[2]);
    kv_pos_ = Eigen::Vector3d(kv_pos[0], kv_pos[1], kv_pos[2]);
    k_R_ = Eigen::Vector3d(k_R[0], k_R[1], k_R[2]);
    kp_att_ = Eigen::Vector3d(kp_att[0], kp_att[1], kp_att[2]);
    ki_att_ = Eigen::Vector3d(ki_att[0], ki_att[1], ki_att[2]);
    kd_att_ = Eigen::Vector3d(kd_att[0], kd_att[1], kd_att[2]);
    
    // 设置物理参数
    mass_ = mass;
    inertia_ = Eigen::Vector3d(inertia[0], inertia[1], inertia[2]);
    compensation_factor_ = compensation_factor;
    
    // 设置油门滤波和限制参数
    filter_coefficient_ = filter_coefficient;
    max_throttle_delta_ = max_throttle_delta;
    throttle_threshold_ = throttle_threshold;

    // 设置Mixer和TetherObserver的参数
    mixer_->set_params(thrust_arm_length, gravity_arm_length, torque_coeff, rpm_coeff, thrust_coeff, 
                       yaw_weight, attitude_weight,thrust_weight);
    tether_observer_->set_params(mass, thrust_coeff);

    param_set_flag_ = true;    
    RCLCPP_INFO(this->get_logger(), "PoseController parameters set successfully");
}

void PoseController::load_and_set_parameters()
{
    auto kp_pos = this->get_parameter("kp_pos").as_double_array();
    auto kv_pos = this->get_parameter("kv_pos").as_double_array();
    auto k_R = this->get_parameter("k_R").as_double_array();
    auto kp_att = this->get_parameter("kp_att").as_double_array();
    auto ki_att = this->get_parameter("ki_att").as_double_array();
    auto kd_att = this->get_parameter("kd_att").as_double_array();
    auto mass = this->get_parameter("mass").as_double();
    auto inertia = this->get_parameter("inertia").as_double_array();
    auto thrust_coeff = this->get_parameter("thrust_coeff").as_double();
    auto thrust_arm_length = this->get_parameter("thrust_arm_length").as_double();
    auto gravity_arm_length = this->get_parameter("gravity_arm_length").as_double();
    auto torque_coeff = this->get_parameter("torque_coeff").as_double();
    auto rpm_coeff = this->get_parameter("rpm_coeff").as_double();
    auto yaw_weight = this->get_parameter("yaw_weight").as_double();
    auto attitude_weight = this->get_parameter("attitude_weight").as_double();
    auto thrust_weight = this->get_parameter("thrust_weight").as_double();
    auto throttle_threshold = this->get_parameter("throttle_threshold").as_double();
    auto compensation_factor = this->get_parameter("compensation_factor").as_double();
    auto filter_coefficient = this->get_parameter("filter_coefficient").as_double();
    auto max_throttle_delta = this->get_parameter("max_throttle_delta").as_double();
    auto hover_target = this->get_parameter("hover_target").as_double_array();
    auto start_delay_s = this->get_parameter("start_delay_s").as_double();
    auto travel_time_s = this->get_parameter("travel_time_s").as_double();
    auto hover_time_s = this->get_parameter("hover_time_s").as_double();
    auto return_time_s = this->get_parameter("return_time_s").as_double();
    
    RCLCPP_INFO(this->get_logger(), "Loaded params.mass: %f", mass);
    RCLCPP_INFO(this->get_logger(), "Loaded params.thrust_coeff: %f", thrust_coeff);
    
    set_params(kp_pos, kv_pos, k_R, kp_att, ki_att, kd_att,
               mass, inertia, thrust_coeff, thrust_arm_length,
               gravity_arm_length, torque_coeff, rpm_coeff, yaw_weight, attitude_weight, thrust_weight, compensation_factor,
               filter_coefficient, max_throttle_delta, throttle_threshold);

    if (hover_target.size() >= 3) {
        hover_target_ = Eigen::Vector3d(hover_target[0], hover_target[1], hover_target[2]);
    }
    start_delay_s_ = start_delay_s;
    travel_time_s_ = travel_time_s;
    hover_time_s_ = hover_time_s;
    return_time_s_ = return_time_s;
}

void PoseController::control_loop()
{
    static int log_decimation = 0;
    const bool should_log = (++log_decimation % 100 == 0); // ~3.3Hz at 3ms loop

    double now = this->now().seconds();
    double dt = now - last_time_;
    if (dt <= 0.0) dt = 0.001; // 防止时间倒流或除以零
    last_time_ = now;

    // --- 定义辅助限幅函数 (用于阻挡 NaN/Inf 和超量程) ---
    auto clamp_vector = [](Eigen::Vector3d& vec, double limit) {
        for (int i = 0; i < 3; ++i) {
            if (std::isnan(vec(i)) || std::isinf(vec(i))) {
                vec(i) = 0.0; // 非法值归零
            } else if (vec(i) > limit) {
                vec(i) = limit;
            } else if (vec(i) < -limit) {
                vec(i) = -limit;
            }
        }
    };

    tether_observer_->update_tether_observation(); // 更新观测结果
    cur_pos_ = tether_observer_->get_position();
    cur_vel_ = tether_observer_->get_velocity();

    // 1. 期望 在world系下
    auto is_finite_vec = [](const Eigen::Vector3d& vec) {
        return std::isfinite(vec(0)) && std::isfinite(vec(1)) && std::isfinite(vec(2));
    };

    if (!home_set_ && is_finite_vec(cur_pos_)) {
        home_pos_ = cur_pos_;
        home_set_ = true;
        hover_phase_ = HoverPhase::Delay;
        phase_start_time_s_ = now;
    }

    Eigen::Vector3d p_r = cur_pos_;
    const double travel_time = std::max(0.01, travel_time_s_);
    const double return_time = std::max(0.01, return_time_s_);

    switch (hover_phase_) {
        case HoverPhase::WaitHome:
            p_r = cur_pos_;
            break;
        case HoverPhase::Delay:
            p_r = home_pos_;
            if ((now - phase_start_time_s_) >= start_delay_s_) {
                hover_phase_ = HoverPhase::GoToHover;
                phase_start_time_s_ = now;
            }
            break;
        case HoverPhase::GoToHover: {
            double t = (now - phase_start_time_s_) / travel_time;
            if (t >= 1.0) {
                t = 1.0;
                hover_phase_ = HoverPhase::HoldHover;
                phase_start_time_s_ = now;
            }
            p_r = home_pos_ + t * (hover_target_ - home_pos_);
            break;
        }
        case HoverPhase::HoldHover:
            p_r = hover_target_;
            if ((now - phase_start_time_s_) >= hover_time_s_) {
                hover_phase_ = HoverPhase::ReturnHome;
                phase_start_time_s_ = now;
            }
            break;
        case HoverPhase::ReturnHome: {
            double t = (now - phase_start_time_s_) / return_time;
            if (t >= 1.0) {
                t = 1.0;
                hover_phase_ = HoverPhase::HoldHome;
                phase_start_time_s_ = now;
            }
            p_r = hover_target_ + t * (home_pos_ - hover_target_);
            break;
        }
        case HoverPhase::HoldHome:
        default:
            p_r = home_pos_;
            break;
    }
    // 任务状态机已经给出位置参考，速度/加速度参考置零可避免参考不一致引入抖动
    Eigen::Vector3d v_r = Eigen::Vector3d::Zero();
    Eigen::Vector3d a_r = Eigen::Vector3d::Zero();
    Eigen::Vector3d p = cur_pos_;
    Eigen::Vector3d v = cur_vel_;

    if (should_log) {
        print_vector("position", p);
        print_vector("position_r", p_r);
    }

    // 2.1 位置环（期望加速度）
    Eigen::Vector3d Kp = kp_pos_;
    Eigen::Vector3d Kv = kv_pos_;
    Eigen::Vector3d p_err = p_r - p;
    Eigen::Vector3d v_err = v_r - v;
    
    // 【修复】：放开位置误差限幅，允许达到 2.0 米，确保能产生足够的矫正加速度
    for ( int i = 0; i < 3; i++) {
        p_err(i) = p_err(i) > 2.0 ? 2.0 : (p_err(i) < -2.0 ? -2.0 : p_err(i)); 
    }
    
    Eigen::Vector3d a_des_world = a_r + Kv.cwiseProduct(v_err + Kp.cwiseProduct(p_err));
    clamp_vector(a_des_world, 10.0); // 限制最大期望加速度，防止激增
    if (should_log) {
        print_vector("a_des_world", a_des_world);
    }

    // 2.2 T_des 和 R_des 计算 (微分平坦映射)
    Eigen::Vector3d g(0, 0, -9.81);
    ft_bodyframe_ = tether_observer_->get_ft_bodyframe(); // 获取拉力观测
    if (should_log) {
        std::cout << "ft_bodyframe: " << ft_bodyframe_.transpose() << std::endl;
    }

    R_bt_ = tether_observer_->get_R_bt(); // 获取旋转矩阵
    mixer_->update_parameters(R_bt_, ft_bodyframe_);

    // 计算世界系下的总期望力 (包含重力补偿)
    Eigen::Vector3d F_des_world = mass_ * a_des_world - mass_ * g;
    
    // 获取期望推力大小 (标量)
    double thrust_scalar = F_des_world.norm();

    // 对于四旋翼，所有推力只能由机体Z轴提供。
    // 所以传给Mixer的期望推力矢量，应该完全在Z轴上 (FLU坐标系下向上为正)
    Eigen::Vector3d T_des(0.0, 0.0, thrust_scalar); 
    clamp_vector(T_des, 50.0); // 限幅保护
    
    // --- 姿态生成：根据期望力矩方向计算无人机应有的倾斜姿态 ---
    Eigen::Matrix3d R_des = Eigen::Matrix3d::Identity();
    if (thrust_scalar > 1e-4) { // 防止自由落体时的除零错误
        // 1. 期望的机体Z轴方向 (就是总力的方向)
        Eigen::Vector3d z_b_des = F_des_world / thrust_scalar;

        // 2. 设定期望航向角 (Yaw)，假设保持 Yaw = 0 (可根据需要改成 ref_q_ 的 yaw)
        double yaw_des = 0.0; 
        Eigen::Vector3d x_c(cos(yaw_des), sin(yaw_des), 0.0);

        // 3. 计算期望机体Y轴 (Z叉乘X)
        Eigen::Vector3d y_b_des = z_b_des.cross(x_c);
        if (y_b_des.norm() < 1e-4) { // 极小概率的奇异点保护
            x_c = Eigen::Vector3d(1.0, 0.0, 0.0); 
            y_b_des = z_b_des.cross(x_c);
        }
        y_b_des.normalize();

        // 4. 计算期望机体X轴 (Y叉乘Z)
        Eigen::Vector3d x_b_des = y_b_des.cross(z_b_des);
        x_b_des.normalize();

        // 5. 组合出期望旋转矩阵
        R_des.col(0) = x_b_des;
        R_des.col(1) = y_b_des;
        R_des.col(2) = z_b_des;
    }

    // 供上层发布和调试使用的参考四元数
    Eigen::Quaterniond q_r(R_des); 
    
    // 3. 姿态环 tau_des计算 
    Eigen::Matrix3d R = cur_q_.toRotationMatrix();

    // R_err 李群误差 
    Eigen::Matrix3d R_err = 0.5 * (R_des.transpose() * R - R.transpose() * R_des);
    // Vee算子：从反对称矩阵提取向量 [0, -z, y; z, 0, -x; -y, x, 0] -> [x, y, z]
    Eigen::Vector3d Re_vee(R_err(2,1), R_err(0,2), R_err(1,0));
    
    // 角速度期望计算
    Eigen::Vector3d omega_des = k_R_.cwiseProduct(Re_vee);
    clamp_vector(omega_des, 20.0); // 限制最大期望角速度
    
    // 角速度误差计算
    Eigen::Vector3d omega_err = omega_des - cur_omega_;
    clamp_vector(omega_err, 30.0); // 限制角速度误差

    if (should_log) {
        print_vector("k_R", k_R_);
    }
    
    // 积分项计算及抗饱和
    omega_err_int_ += omega_err * dt;
    clamp_vector(omega_err_int_, 10.0); // 积分限幅
    
    // 微分项计算及抗扰动
    Eigen::Vector3d omega_err_diff = (omega_err - omega_last_err_) / dt;
    clamp_vector(omega_err_diff, 500.0); // 微分限幅
    
    omega_last_err_ = omega_err;
    
    // 【修复】：力矩期望计算 (PID 输出的角加速度 * 物理惯量矩阵)
    Eigen::Vector3d raw_pid_torque = kp_att_.cwiseProduct(omega_err)
                                   + ki_att_.cwiseProduct(omega_err_int_)
                                   + kd_att_.cwiseProduct(omega_err_diff);
                                   
    Eigen::Vector3d tau_des = inertia_.cwiseProduct(raw_pid_torque);
    clamp_vector(tau_des, 5.0); // 最终力矩限幅

    // 4. mix
    // Mixer uses FRD body convention. Convert FLU command to FRD: x unchanged, y/z flipped.
    Eigen::Vector3d T_des_frd(T_des(0), -T_des(1), -T_des(2));
    Eigen::Vector3d tau_des_frd(tau_des(0), -tau_des(1), -tau_des(2));

    Eigen::Matrix<double, 6, 1> control_des;
    control_des << T_des_frd, tau_des_frd;
    
    if (should_log) {
        print_vector("T_des", T_des);
        print_vector("tau_des", tau_des);
        print_vector("control_des", control_des);
    }

    Eigen::Matrix<double,4,1> motor_thrust_des;
    motor_thrust_des = mixer_->mix(control_des);
    if (should_log) {
        print_vector("motor_thrust_des", motor_thrust_des);
    }
    
    throttle_ = mixer_->thrust2throttle(motor_thrust_des);

    // 低通滤波和速率限制
    for(int i = 0; i < 4; i++) {
        // 防止 NaN 污染油门
        if(std::isnan(throttle_[i]) || std::isinf(throttle_[i])) {
            throttle_[i] = 0.0;
        }

        // 低通滤波
        throttle_filtered_[i] = filter_coefficient_ * throttle_[i] + (1.0 - filter_coefficient_) * throttle_last_[i];
        
        // 速率限制
        double delta = throttle_filtered_[i] - throttle_last_[i];
        if(std::abs(delta) > max_throttle_delta_) {
            delta = (delta > 0) ? max_throttle_delta_ : -max_throttle_delta_;
        }
        throttle_filtered_[i] = throttle_last_[i] + delta;
        
        // 更新上一时刻值
        throttle_last_[i] = throttle_filtered_[i];
    }

    if (should_log) {
        print_vector("throttle_filtered", throttle_filtered_);
    }

    tether_observer_->update_throttle(throttle_filtered_); // 更新TetherObserver使用的滤波后的油门数据
    motors_throttle_pub(throttle_filtered_[0], throttle_filtered_[1], throttle_filtered_[2], throttle_filtered_[3]);
    
    // 发布控制器状态消息
    publish_controller_state(dt, a_des_world, omega_des, T_des, tau_des, p_r, q_r);
    
    // 输出分隔线
    if (should_log) {
        std::cout << std::string(80, '-') << std::endl;
    }
}

// 实现print_vector函数
void PoseController::print_vector(const std::string& name, const Eigen::Vector3d& vec) {
    std::cout << std::setw(12) << std::left << name << ": [";
    for(int i = 0; i < vec.size(); i++) {
        if(i > 0) std::cout << ", ";
        std::cout << std::setw(8) << std::right << std::fixed << std::setprecision(4) << vec(i);
    }
    std::cout << "]" << std::endl;
}

void PoseController::print_vector(const std::string& name, const Eigen::Vector4d& vec) {
    std::cout << std::setw(12) << std::left << name << ": [";
    for(int i = 0; i < vec.size(); i++) {
        if(i > 0) std::cout << ", ";
        std::cout << std::setw(8) << std::right << std::fixed << std::setprecision(4) << vec(i);
    }
    std::cout << "]" << std::endl;
}

void PoseController::print_vector(const std::string& name, const Eigen::Matrix<double, 6, 1>& vec) {
    std::cout << std::setw(12) << std::left << name << ": [";
    for(int i = 0; i < vec.size(); i++) {
        if(i > 0) std::cout << ", ";
        std::cout << std::setw(8) << std::right << std::fixed << std::setprecision(4) << vec(i);
    }
    std::cout << "]" << std::endl;
}

void PoseController::reference_callback_custom(const aerogripper_msgs::msg::PoseReference::SharedPtr msg)
{
    // 从自定义消息中提取所有五个类型的信息
    ref_pos_ = Eigen::Vector3d(msg->position.x, msg->position.y, msg->position.z);
    ref_vel_ = Eigen::Vector3d(msg->linear_velocity.x, msg->linear_velocity.y, msg->linear_velocity.z);
    ref_acc_ = Eigen::Vector3d(msg->linear_acceleration.x, msg->linear_acceleration.y, msg->linear_acceleration.z);
    ref_q_ = Eigen::Quaterniond(msg->orientation.w, msg->orientation.x, msg->orientation.y, msg->orientation.z);
    ref_omega_ = Eigen::Vector3d(msg->angular_velocity.x, msg->angular_velocity.y, msg->angular_velocity.z);
}

void PoseController::position_callback(const px4_msgs::msg::VehicleLocalPosition::SharedPtr msg)
{
    // cur_pos_ = Eigen::Vector3d(msg->x, msg->y, msg->z);
    // cur_vel_ = Eigen::Vector3d(msg->vx, msg->vy, msg->vz);
}

void PoseController::imu_callback(const px4_msgs::msg::SensorCombined::SharedPtr msg)
{
    // 强制限幅过滤函数，防范 NaN/Inf 和编译器的 fast-math 优化失效
    auto sanitize_value = [](float val, float limit) {
        if (std::isnan(val) || std::isinf(val) || val > limit || val < -limit) {
            return 0.0f; // 出现非法值或超量程，强制归零
        }
        return val;
    };

    // 加速度限幅 (如 200 m/s^2)
    Eigen::Vector3d acc_raw_FRD(
        sanitize_value(msg->accelerometer_m_s2[0], 200.0f), 
        sanitize_value(msg->accelerometer_m_s2[1], 200.0f), 
        sanitize_value(msg->accelerometer_m_s2[2], 200.0f)
    );
    acc_raw_ = Eigen::Vector3d(acc_raw_FRD(0), -acc_raw_FRD(1), -acc_raw_FRD(2));
    Eigen::Vector3d g(0, 0, -9.81);
    cur_acc_ = acc_raw_ + R_bw_ * g;

    // 角速度限幅 (如 50 rad/s)
    Eigen::Vector3d omega_raw_FRD(
        sanitize_value(msg->gyro_rad[0], 50.0f), 
        sanitize_value(msg->gyro_rad[1], 50.0f), 
        sanitize_value(msg->gyro_rad[2], 50.0f)
    );
    cur_omega_ = Eigen::Vector3d(omega_raw_FRD(0), -omega_raw_FRD(1), -omega_raw_FRD(2));

    // 更新TetherObserver的IMU数据
    tether_observer_->update_imu(cur_acc_, cur_omega_);
}

void PoseController::attitude_callback(const px4_msgs::msg::VehicleAttitude::SharedPtr msg)
{
    // 获取当前FRD四元数并强制归一化
    Eigen::Quaterniond q_FRD_raw(msg->q[0], msg->q[1], msg->q[2], msg->q[3]);
    q_FRD_raw.normalize(); 
    
    // 设置初始姿态作为offset（只在第一次调用时）
    if (!q_init_flag_) {
        q_init_FRD_ = q_FRD_raw;
        q_init_flag_ = true;
        RCLCPP_INFO(this->get_logger(), "Initial attitude set as offset: w=%.3f, x=%.3f, y=%.3f, z=%.3f", 
                    q_init_FRD_.w(), q_init_FRD_.x(), q_init_FRD_.y(), q_init_FRD_.z());
    }
    
    // 计算相对于启动时刻坐标系的当前姿态
    Eigen::Quaterniond q_FRD = q_init_FRD_.inverse() * q_FRD_raw;
    
    // 坐标系转换：FRD -> FLU
    static const Eigen::Quaterniond FRD2FLU(0, 1, 0, 0); // (w, x, y, z) 表示绕x轴旋转180度
    
    Eigen::Quaterniond q_FLU = FRD2FLU * q_FRD * FRD2FLU.inverse();
    q_FLU.normalize(); // 坐标系变换后再次归一化防偏转
    
    // 转换为旋转矩阵和四元数
    cur_q_ = q_FLU;
    R_bw_ = q_FLU.toRotationMatrix().transpose();
    
    // 更新TetherObserver的姿态数据
    tether_observer_->update_attitude(R_bw_, cur_q_);
}

void PoseController::motors_throttle_pub(float motor1, float motor2, float motor3, float motor4) {
    px4_msgs::msg::ActuatorMotors msg;
    msg.control = {
        motor1, motor2, motor3, motor4,
        std::nanf("1"), std::nanf("1"), std::nanf("1"), std::nanf("1"),
        std::nanf("1"), std::nanf("1"), std::nanf("1"), std::nanf("1")
    };
    msg.reversible_flags = 0;
    msg.timestamp = this->now().nanoseconds() / 1000;
    msg.timestamp_sample = msg.timestamp;
    motors_throttle_pub_->publish(msg);
}

void PoseController::enable_offboard_mode()
{
    px4_msgs::msg::VehicleCommand mode_msg;
    mode_msg.timestamp = this->now().nanoseconds() / 1000;
    mode_msg.command = 176;  // MAV_CMD_DO_SET_MODE
    mode_msg.param1 = 1.0;   // 基础模式
    mode_msg.param2 = 6.0;   // 自定义模式 (Offboard)
    mode_msg.param3 = 0.0;   // 自定义子模式
    mode_msg.target_system = 1;
    mode_msg.target_component = 1;
    mode_msg.source_system = 1;
    mode_msg.source_component = 1;
    mode_msg.from_external = true;
    
    vehicle_command_pub_->publish(mode_msg);
    RCLCPP_INFO(this->get_logger(), "Sent mode switch command to Offboard");
    
    px4_msgs::msg::OffboardControlMode offboard_msg;
    offboard_msg.timestamp = this->now().nanoseconds() / 1000;
    offboard_msg.position = false;
    offboard_msg.velocity = false;
    offboard_msg.acceleration = false;
    offboard_msg.attitude = false;
    offboard_msg.body_rate = false;
    offboard_msg.thrust_and_torque = false;
    offboard_msg.direct_actuator = true;  // 与motor_test保持一致
    offboard_control_mode_pub_->publish(offboard_msg);
    offboard_counter_++;
    
    if (offboard_counter_ >= 10) { 
        offboard_mode_enabled_ = true;
        RCLCPP_INFO(this->get_logger(), "Offboard mode enabled!");
    }
}

void PoseController::arm_vehicle()
{
    if (!offboard_mode_enabled_) {
        RCLCPP_WARN(this->get_logger(), "Cannot arm vehicle: offboard mode not enabled yet");
        return;
    }
    
    px4_msgs::msg::VehicleCommand msg;
    msg.timestamp = this->now().nanoseconds() / 1000; 
    msg.param1 = 1.0; // 1 = arm, 0 = disarm
    msg.param2 = 0.0; // 强制解锁
    msg.param3 = 0.0; 
    msg.param4 = 0.0; 
    msg.param5 = 0.0; 
    msg.param6 = 0.0; 
    msg.param7 = 0.0; 
    msg.command = 400; // MAV_CMD_COMPONENT_ARM_DISARM
    msg.target_system = 1;
    msg.target_component = 1;
    msg.source_system = 1;
    msg.source_component = 1;
    msg.from_external = true;
    
    vehicle_command_pub_->publish(msg);
    arm_counter_++;
    
    if (arm_counter_ >= 5) { 
        vehicle_armed_ = true;
        RCLCPP_INFO(this->get_logger(), "Vehicle armed successfully!");
    }
}

void PoseController::offboard_control_loop()
{
    if (!offboard_mode_enabled_) {
        enable_offboard_mode();
        return;
    }
    
    if (!vehicle_armed_) {
        arm_vehicle();
        return;
    }
    
    if (offboard_mode_enabled_ && vehicle_armed_) {
        px4_msgs::msg::OffboardControlMode msg;
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
}

void PoseController::publish_controller_state(double dt, const Eigen::Vector3d& a_des_world, 
                                            const Eigen::Vector3d& omega_des, 
                                            const Eigen::Vector3d& T_des, 
                                            const Eigen::Vector3d& tau_des,
                                            const Eigen::Vector3d& ref_pos_,
                                            const Eigen::Quaterniond& ref_q_)
{
    auto msg = std::make_unique<aerogripper_msgs::msg::ControllerState>();
    
    msg->timestamp = this->now();
    msg->dt = dt;
    
    msg->desired_acceleration.x = a_des_world(0);
    msg->desired_acceleration.y = a_des_world(1);
    msg->desired_acceleration.z = a_des_world(2);
    
    msg->desired_angular_velocity.x = omega_des(0);
    msg->desired_angular_velocity.y = omega_des(1);
    msg->desired_angular_velocity.z = omega_des(2);
    
    msg->desired_thrust.x = T_des(0);
    msg->desired_thrust.y = T_des(1);
    msg->desired_thrust.z = T_des(2);
    
    msg->desired_torque.x = tau_des(0);
    msg->desired_torque.y = tau_des(1);
    msg->desired_torque.z = tau_des(2);
    
    msg->current_position.x = cur_pos_(0);
    msg->current_position.y = cur_pos_(1);
    msg->current_position.z = cur_pos_(2);
    
    msg->current_velocity.x = cur_vel_(0);
    msg->current_velocity.y = cur_vel_(1);
    msg->current_velocity.z = cur_vel_(2);
    
    msg->current_orientation.w = cur_q_.w();
    msg->current_orientation.x = cur_q_.x();
    msg->current_orientation.y = cur_q_.y();
    msg->current_orientation.z = cur_q_.z();
    
    Eigen::Vector3d euler_angles = cur_q_.toRotationMatrix().eulerAngles(0, 1, 2); 
    msg->current_euler_angles.x = euler_angles(0); 
    msg->current_euler_angles.y = euler_angles(1); 
    msg->current_euler_angles.z = euler_angles(2); 
    
    msg->current_angular_velocity.x = cur_omega_(0);
    msg->current_angular_velocity.y = cur_omega_(1);
    msg->current_angular_velocity.z = cur_omega_(2);
    
    msg->reference_position.x = ref_pos_(0);
    msg->reference_position.y = ref_pos_(1);
    msg->reference_position.z = ref_pos_(2);
    
    msg->reference_velocity.x = ref_vel_(0);
    msg->reference_velocity.y = ref_vel_(1);
    msg->reference_velocity.z = ref_vel_(2);
    
    msg->reference_orientation.w = ref_q_.w();
    msg->reference_orientation.x = ref_q_.x();
    msg->reference_orientation.y = ref_q_.y();
    msg->reference_orientation.z = ref_q_.z();
    
    Eigen::Vector3d ref_euler_angles = ref_q_.toRotationMatrix().eulerAngles(0, 1, 2); 
    msg->reference_euler_angles.x = ref_euler_angles(0); 
    msg->reference_euler_angles.y = ref_euler_angles(1); 
    msg->reference_euler_angles.z = ref_euler_angles(2); 
    
    msg->throttle_output[0] = throttle_filtered_[0];
    msg->throttle_output[1] = throttle_filtered_[1];
    msg->throttle_output[2] = throttle_filtered_[2];
    msg->throttle_output[3] = throttle_filtered_[3];
    
    msg->tether_force_bodyframe.x = ft_bodyframe_(0);
    msg->tether_force_bodyframe.y = ft_bodyframe_(1);
    msg->tether_force_bodyframe.z = ft_bodyframe_(2);
    
    controller_state_pub_->publish(*msg);
}