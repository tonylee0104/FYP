#include "mixer.h"

Mixer::Mixer()
{
 // 初始化参数为默认值
 param_set_flag_ = false;
 thrust_arm_length_ = 0.2;
 gravity_arm_length_ = 0.2;
 torque_coeff_ = 0.01;
 rpm_coeff_ = 0.01;
 thrust_coeff_ = 0.01;
 yaw_weight_ = 1.0; // 默认yaw权重为1.0
 attitude_weight_ = 1.0; // 默认姿态权重为1.0
 thrust_weight_ = 1.0; // 默认推力权重为1.0


 // 使用默认参数构控制矩阵
 double L_T = thrust_arm_length_;
 double L_G = gravity_arm_length_;
 double kappa_t = torque_coeff_;
 double sqrt2_2 = std::sqrt(2.0) / 2.0;
 double theta = 135 * M_PI / 180.0; // 135度转弯角，单位为弧度
 
 allocation_matrix_ = Eigen::Matrix<double,6,4>::Zero();
 allocation_matrix_ <<
  -sqrt2_2*cos(theta),  sqrt2_2*cos(theta), -sqrt2_2*cos(theta),  sqrt2_2*cos(theta),
   sqrt2_2*cos(theta), -sqrt2_2*cos(theta), -sqrt2_2*cos(theta),  sqrt2_2*cos(theta),
  -sin(theta),         -sin(theta),         -sin(theta),         -sin(theta),
   sqrt2_2*cos(theta)*L_T - sqrt2_2*cos(theta)*kappa_t, -sqrt2_2*cos(theta)*L_T + sqrt2_2*cos(theta)*kappa_t, -sqrt2_2*cos(theta)*L_T + sqrt2_2*cos(theta)*kappa_t,  sqrt2_2*cos(theta)*L_T - sqrt2_2*cos(theta)*kappa_t,
   sqrt2_2*cos(theta)*L_T + sqrt2_2*cos(theta)*kappa_t, -sqrt2_2*cos(theta)*L_T - sqrt2_2*cos(theta)*kappa_t,  sqrt2_2*cos(theta)*L_T + sqrt2_2*cos(theta)*kappa_t, -sqrt2_2*cos(theta)*L_T - sqrt2_2*cos(theta)*kappa_t,
  -sin(theta)*kappa_t, -sin(theta)*kappa_t,  sin(theta)*kappa_t,  sin(theta)*kappa_t;

 // 初始化增强矩阵
 allocation_matrix_w_ << 0, 0, 1, 0, 0, 0; // 初始化假定垂直悬吊，绳子只给一个向上的力
 allocation_matrix_enhanced_.block<6, 4> (0, 0) = allocation_matrix_;
 allocation_matrix_enhanced_.block<6, 1> (0, 4) = allocation_matrix_w_;

 // // 使用加权伪逆，给yaw控制和姿态控制适当的权重
 // // 方法1: 直接修改控制分配矩阵的相应行
 // Eigen::Matrix<double,6,4> allocation_matrix_weighted = allocation_matrix_;
 
 // allocation_matrix_weighted.row(0) *= thrust_weight_;
 // allocation_matrix_weighted.row(1) *= thrust_weight_;
 // allocation_matrix_weighted.row(2) *= thrust_weight_;
 // // 姿态控制行（第4、5、6行：roll moment, pitch moment, yaw moment）
 // allocation_matrix_weighted.row(3) *= attitude_weight_; // roll moment
 // allocation_matrix_weighted.row(4) *= attitude_weight_; // pitch moment
 // allocation_matrix_weighted.row(5) *= attitude_weight_; // yaw moment
 // allocation_matrix_weighted.row(5) *= yaw_weight_; // yaw moment
 // // 计算加权后的伪逆
 // mix_matrix_ = allocation_matrix_weighted.completeOrthogonalDecomposition().pseudoInverse();

 mix_matrix_ = allocation_matrix_.completeOrthogonalDecomposition().pseudoInverse();
 mix_matrix_.col(3) *= attitude_weight_;
 mix_matrix_.col(4) *= attitude_weight_;
 mix_matrix_.col(5) *= attitude_weight_;
 mix_matrix_.col(5) *= yaw_weight_;
}

void Mixer::set_params(double thrust_arm_length, double gravity_arm_length, 
       double torque_coeff, double rpm_coeff, double thrust_coeff,
       double yaw_weight, double attitude_weight, double thrust_weight)
{
 thrust_arm_length_ = thrust_arm_length;
 gravity_arm_length_ = gravity_arm_length;
 torque_coeff_ = torque_coeff;
 rpm_coeff_ = rpm_coeff;
 thrust_coeff_ = thrust_coeff;
 yaw_weight_ = yaw_weight;
 attitude_weight_ = attitude_weight;
 thrust_weight_ = thrust_weight;

 // 重新计算控制矩阵
 double L_T = thrust_arm_length_;
 double L_G = gravity_arm_length_;
 double kappa_t = torque_coeff_;
 double sqrt2_2 = std::sqrt(2.0) / 2.0;
 double theta = 135.0 * M_PI / 180.0; // 135度转弯角，单位为弧度
 
 allocation_matrix_ <<
  -sqrt2_2*cos(theta),  sqrt2_2*cos(theta), -sqrt2_2*cos(theta),  sqrt2_2*cos(theta),
   sqrt2_2*cos(theta), -sqrt2_2*cos(theta), -sqrt2_2*cos(theta),  sqrt2_2*cos(theta),
  -sin(theta),         -sin(theta),         -sin(theta),         -sin(theta),
   sqrt2_2*cos(theta)*L_T - sqrt2_2*cos(theta)*kappa_t, -sqrt2_2*cos(theta)*L_T + sqrt2_2*cos(theta)*kappa_t, -sqrt2_2*cos(theta)*L_T + sqrt2_2*cos(theta)*kappa_t,  sqrt2_2*cos(theta)*L_T - sqrt2_2*cos(theta)*kappa_t,
   sqrt2_2*cos(theta)*L_T + sqrt2_2*cos(theta)*kappa_t, -sqrt2_2*cos(theta)*L_T - sqrt2_2*cos(theta)*kappa_t,  sqrt2_2*cos(theta)*L_T + sqrt2_2*cos(theta)*kappa_t, -sqrt2_2*cos(theta)*L_T - sqrt2_2*cos(theta)*kappa_t,
  -sin(theta)*kappa_t, -sin(theta)*kappa_t,  sin(theta)*kappa_t,  sin(theta)*kappa_t;

 // 更新增强矩阵
 allocation_matrix_w_ << 0, 0, 1, 0, 0, 0; // 初始化假定垂直悬吊，绳子只给一个向上的力
 allocation_matrix_enhanced_.block<6, 4> (0, 0) = allocation_matrix_;
 allocation_matrix_enhanced_.block<6, 1> (0, 4) = allocation_matrix_w_;
 

 // // 使用加权伪逆，给yaw控制和姿态控制适当的权重
 // // 方法1: 直接修改控制分配矩阵的相应行
 // Eigen::Matrix<double,6,4> allocation_matrix_weighted = allocation_matrix_;
 
 // allocation_matrix_weighted.row(0) *= thrust_weight_;
 // allocation_matrix_weighted.row(1) *= thrust_weight_;
 // allocation_matrix_weighted.row(2) *= thrust_weight_;
 // // 姿态控制行（第4、5、6行：roll moment, pitch moment, yaw moment）
 // allocation_matrix_weighted.row(3) *= attitude_weight_; // roll moment
 // allocation_matrix_weighted.row(4) *= attitude_weight_; // pitch moment
 // allocation_matrix_weighted.row(5) *= attitude_weight_; // yaw moment
 // allocation_matrix_weighted.row(5) *= yaw_weight_; // yaw moment
 
 cout<<"yaw_weight_:"<<yaw_weight_<<endl;
 cout<<"attitude_weight_:"<<attitude_weight_<<endl;
 cout<<"allocation_matrix_:"<<endl<<allocation_matrix_<<endl;
 
 // 计算加权后的伪逆
 mix_matrix_ = allocation_matrix_.completeOrthogonalDecomposition().pseudoInverse();
 cout<<"mix_matrix_:"<<endl<<mix_matrix_<<endl;
 mix_matrix_.col(3) *= attitude_weight_;
 mix_matrix_.col(4) *= attitude_weight_;
 mix_matrix_.col(5) *= attitude_weight_;
 mix_matrix_.col(5) *= yaw_weight_;
 cout<<"weighted mix_matrix_:"<<endl<<mix_matrix_<<endl;
 
 mix_matrix_enhanced_ = allocation_matrix_enhanced_.completeOrthogonalDecomposition().pseudoInverse();
 
 param_set_flag_ = true;
 std::cout << "Mixer parameters set successfully" << std::endl;
}

void Mixer::update_parameters(const Eigen::Matrix3d& R_bt, const Eigen::Vector3d& ft_bodyframe)
{
 update_enhanced_matrix(R_bt);
 update_ft_bodyframe(ft_bodyframe);
}

//读取到R_bt后，计算 allocation_matrix_w_ 并更新 mix_matrix_enhanced_
void Mixer::update_enhanced_matrix(const Eigen::Matrix3d& R_bt)
{
 R_bt_ = R_bt;

 // 计算 R_bt * z
 Eigen::Vector3d z_axis(0, 0, 1);
 Eigen::Vector3d Rbt_z = R_bt * z_axis;

 // 计算 -l_G x (R_bt * z)
 Eigen::Vector3d L_G_vec(0, 0, gravity_arm_length_); // 重力臂长度向量

 // 构造 allocation_matrix_w_
 allocation_matrix_w_.block<3, 1> (0, 0) = Rbt_z;
 allocation_matrix_w_.block<3, 1> (3, 0) = L_G_vec.cross(Rbt_z);

 allocation_matrix_enhanced_.block<6, 4> (0, 0) = allocation_matrix_;
 allocation_matrix_enhanced_.block<6, 1> (0, 4) = allocation_matrix_w_;

 mix_matrix_enhanced_ = allocation_matrix_enhanced_.completeOrthogonalDecomposition().pseudoInverse();

 // cout<<"allocation_matrix_enhanced_"<<endl<<allocation_matrix_enhanced_<<endl;
 // cout<<"mix_matrix_enhanced_"<<endl<<mix_matrix_enhanced_<<endl;
}

void Mixer::update_ft_bodyframe(const Eigen::Vector3d& ft_bodyframe)
{
 ft_bodyframe_ = ft_bodyframe;
}

Eigen::Matrix<double,4,1> Mixer::mix(const Eigen::Matrix<double,6,1>& u) const
{
 return mix_matrix_ * u;
}

Eigen::Matrix<double,5,1> Mixer::mix_enhanced(const Eigen::Matrix<double,6,1>& u) const
{
 cout<<"mix_matrix_enhanced_:"<<endl<<mix_matrix_enhanced_<<endl;
 cout<<"allocation_matrix_w_:"<<endl<<allocation_matrix_w_.transpose()<<endl;    
 return mix_matrix_enhanced_ * u;
}

Eigen::Matrix<double,6,1> Mixer::control_allocation(const Eigen::Matrix<double,4,1>& motors_thrust) const
{
 return allocation_matrix_ * motors_thrust;
}

Eigen::Matrix<double,6,1> Mixer::control_allocation_enhanced(const Eigen::Matrix<double,4,1>& motors_thrust, double w) const
{
 Eigen::Matrix<double,5,1> extended_input;
 extended_input << motors_thrust, w;
 return allocation_matrix_enhanced_ * extended_input;
}

Eigen::Matrix<double,4,1> Mixer::throttle2thrust(const Eigen::Matrix<double,4,1>& throttle) const
{
 Eigen::Matrix<double,4,1> thrust;
 for (int i = 0; i < 4; ++i) {
  thrust(i) = thrust_coeff_ * throttle(i) * throttle(i);
 }
 return thrust;
}

Eigen::Matrix<double,4,1> Mixer::thrust2throttle(const Eigen::Matrix<double,4,1>& thrust) const
{
 Eigen::Matrix<double,4,1> throttle;
 double threshold = 0.3;
 for (int i = 0; i < 4; ++i) {
  if (thrust(i) > 0) {
   throttle(i) = sqrt(thrust(i) / thrust_coeff_);
  } else {
   throttle(i) = 0;
  }

  // cout<<"oringinal throttle: "<<throttle(i)<<endl;

  if (throttle(i) > threshold) {
   throttle(i) = threshold;
   // std::cout << "throttle is too large" << std::endl;
  }
 }
 return throttle;
}

