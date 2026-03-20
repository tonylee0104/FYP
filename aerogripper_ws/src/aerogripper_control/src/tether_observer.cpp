// todo 离心力观测！！！！！！！！！！！


#include "tether_observer.h"

TetherObserver::TetherObserver(std::shared_ptr<Mixer> mixer) : 
    mixer_(mixer), // 初始化mixer，传递this指针以读取参数
    R_bt_(Eigen::Matrix3d::Identity()),
    R_bw_(Eigen::Matrix3d::Identity()),
    acc_(Eigen::Vector3d::Zero()),
    Tb_(Eigen::Vector3d::Zero()),
    throttle_(Eigen::Vector4d::Zero()),
    ft_bodyframe_(Eigen::Vector3d::Zero()),
    position_(Eigen::Vector3d::Zero()),
    velocity_(Eigen::Vector3d::Zero())
{
    // 初始化参数为默认值
    param_set_flag_ = false;
}

void TetherObserver::set_params(double mass, double thrust_coeff)
{
    mass_ = mass;
    thrust_coeff_ = thrust_coeff;

    Eigen::Vector3d g(0, 0, -9.81);
    ft_bodyframe_ = -mass_ * g;

    param_set_flag_ = true;
    std::cout << "TetherObserver parameters set successfully" << std::endl;
}

// 公共接口方法，供外部调用
void TetherObserver::update_imu(const Eigen::Vector3d& acc, const Eigen::Vector3d& omega) {
    acc_ = acc;
    omega_ = omega;
}

void TetherObserver::update_attitude(const Eigen::Matrix3d& R_bw, const Eigen::Quaterniond& q) {
    R_bw_ = R_bw;
    q_ = q;
}

void TetherObserver::update_throttle(const Eigen::Vector4d& throttle) {
    throttle_ = throttle;
    update_thrust();
}

// 控制分配：输入为4个电机的转速，输出为机体系三轴推力和三轴力矩，用于观测机体系推力矢量
void TetherObserver::update_thrust() {    
    Eigen::Vector4d motors_thrust;
    motors_thrust = mixer_->throttle2thrust(throttle_);
    // cout<<"motor_thrust:"<<endl<<motors_thrust<<endl;
    Eigen::Matrix<double,6,1> control_output = mixer_->control_allocation(motors_thrust);
    Tb_ = control_output.head<3>(); // 推力（机体系）

    if(std::isnan(Tb_(0)) || std::isnan(Tb_(1)) || std::isnan(Tb_(2))) { 
        cout<<"Tb is nan"<<endl
        << "throttle: "<<throttle_.transpose()<<endl
        <<"Tb: "<<Tb_.transpose()<<endl;
        exit(0); //debug
    }

    tau_b_ = control_output.tail<3>(); // 力矩（机体系）
}

// 观测机体系下的绳子拉力矢量
void TetherObserver::update_tether_observation()
{

    Eigen::Vector3d g(0, 0, -9.81);
    
    // f_t = R_bt^T [m*pddot - R_bw*m*g - Tb] body系下
    ft_bodyframe_ = mass_ * acc_ - R_bw_ * mass_ * g - Tb_;
    // ft_bodyframe_ = mass_ * acc_ - R_bw_ * mass_ * g - mass_ * g;

    // R_bt_ 满足 R_bt_^(-1) * ft_bodyframe_ = n * (0,0,1)
    // 即 n * R_bt_ * (0,0,1) = ft_bodyframe_
    // 构造一个旋转矩阵，使机体系z轴对准绳子拉力方向
    Eigen::Vector3d z_b = ft_bodyframe_.normalized();
    Eigen::Vector3d x_cand(1, 0, 0);
    if (std::abs(z_b.dot(x_cand)) > 0.99) x_cand = Eigen::Vector3d(0, 1, 0); // 防止共线
    Eigen::Vector3d y_b = z_b.cross(x_cand).normalized();
    Eigen::Vector3d x_b = y_b.cross(z_b).normalized();
    R_bt_.col(0) = x_b;
    R_bt_.col(1) = y_b;
    R_bt_.col(2) = z_b;
    
    last_position_ = position_;
    tether_length_ = 0.2;
    Eigen::Vector3d ft_worldframe_ = R_bw_.transpose() * ft_bodyframe_;
    cout<<"ftworld "<<endl<<ft_worldframe_<<endl;
    position_ = -ft_worldframe_.normalized() * tether_length_; // 更新绳子末端在世界坐标系下位置
    
    // double now = this->now().seconds(); /////////todo/////////////
    // double dt = now - last_time_;
    // if (dt <= 0.0) dt = 0.001;
    // last_time_ = now;

    // velocity_ = (position_ - last_position_) / dt; // 更新绳子末端在世界坐标系下速度
}
