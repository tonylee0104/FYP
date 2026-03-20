// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from aerogripper_msgs:msg/ControllerState.idl
// generated code does not contain a copyright notice

#ifndef AEROGRIPPER_MSGS__MSG__DETAIL__CONTROLLER_STATE__BUILDER_HPP_
#define AEROGRIPPER_MSGS__MSG__DETAIL__CONTROLLER_STATE__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "aerogripper_msgs/msg/detail/controller_state__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace aerogripper_msgs
{

namespace msg
{

namespace builder
{

class Init_ControllerState_tether_force_bodyframe
{
public:
  explicit Init_ControllerState_tether_force_bodyframe(::aerogripper_msgs::msg::ControllerState & msg)
  : msg_(msg)
  {}
  ::aerogripper_msgs::msg::ControllerState tether_force_bodyframe(::aerogripper_msgs::msg::ControllerState::_tether_force_bodyframe_type arg)
  {
    msg_.tether_force_bodyframe = std::move(arg);
    return std::move(msg_);
  }

private:
  ::aerogripper_msgs::msg::ControllerState msg_;
};

class Init_ControllerState_throttle_output
{
public:
  explicit Init_ControllerState_throttle_output(::aerogripper_msgs::msg::ControllerState & msg)
  : msg_(msg)
  {}
  Init_ControllerState_tether_force_bodyframe throttle_output(::aerogripper_msgs::msg::ControllerState::_throttle_output_type arg)
  {
    msg_.throttle_output = std::move(arg);
    return Init_ControllerState_tether_force_bodyframe(msg_);
  }

private:
  ::aerogripper_msgs::msg::ControllerState msg_;
};

class Init_ControllerState_reference_euler_angles
{
public:
  explicit Init_ControllerState_reference_euler_angles(::aerogripper_msgs::msg::ControllerState & msg)
  : msg_(msg)
  {}
  Init_ControllerState_throttle_output reference_euler_angles(::aerogripper_msgs::msg::ControllerState::_reference_euler_angles_type arg)
  {
    msg_.reference_euler_angles = std::move(arg);
    return Init_ControllerState_throttle_output(msg_);
  }

private:
  ::aerogripper_msgs::msg::ControllerState msg_;
};

class Init_ControllerState_reference_orientation
{
public:
  explicit Init_ControllerState_reference_orientation(::aerogripper_msgs::msg::ControllerState & msg)
  : msg_(msg)
  {}
  Init_ControllerState_reference_euler_angles reference_orientation(::aerogripper_msgs::msg::ControllerState::_reference_orientation_type arg)
  {
    msg_.reference_orientation = std::move(arg);
    return Init_ControllerState_reference_euler_angles(msg_);
  }

private:
  ::aerogripper_msgs::msg::ControllerState msg_;
};

class Init_ControllerState_reference_velocity
{
public:
  explicit Init_ControllerState_reference_velocity(::aerogripper_msgs::msg::ControllerState & msg)
  : msg_(msg)
  {}
  Init_ControllerState_reference_orientation reference_velocity(::aerogripper_msgs::msg::ControllerState::_reference_velocity_type arg)
  {
    msg_.reference_velocity = std::move(arg);
    return Init_ControllerState_reference_orientation(msg_);
  }

private:
  ::aerogripper_msgs::msg::ControllerState msg_;
};

class Init_ControllerState_reference_position
{
public:
  explicit Init_ControllerState_reference_position(::aerogripper_msgs::msg::ControllerState & msg)
  : msg_(msg)
  {}
  Init_ControllerState_reference_velocity reference_position(::aerogripper_msgs::msg::ControllerState::_reference_position_type arg)
  {
    msg_.reference_position = std::move(arg);
    return Init_ControllerState_reference_velocity(msg_);
  }

private:
  ::aerogripper_msgs::msg::ControllerState msg_;
};

class Init_ControllerState_current_angular_velocity
{
public:
  explicit Init_ControllerState_current_angular_velocity(::aerogripper_msgs::msg::ControllerState & msg)
  : msg_(msg)
  {}
  Init_ControllerState_reference_position current_angular_velocity(::aerogripper_msgs::msg::ControllerState::_current_angular_velocity_type arg)
  {
    msg_.current_angular_velocity = std::move(arg);
    return Init_ControllerState_reference_position(msg_);
  }

private:
  ::aerogripper_msgs::msg::ControllerState msg_;
};

class Init_ControllerState_current_euler_angles
{
public:
  explicit Init_ControllerState_current_euler_angles(::aerogripper_msgs::msg::ControllerState & msg)
  : msg_(msg)
  {}
  Init_ControllerState_current_angular_velocity current_euler_angles(::aerogripper_msgs::msg::ControllerState::_current_euler_angles_type arg)
  {
    msg_.current_euler_angles = std::move(arg);
    return Init_ControllerState_current_angular_velocity(msg_);
  }

private:
  ::aerogripper_msgs::msg::ControllerState msg_;
};

class Init_ControllerState_current_orientation
{
public:
  explicit Init_ControllerState_current_orientation(::aerogripper_msgs::msg::ControllerState & msg)
  : msg_(msg)
  {}
  Init_ControllerState_current_euler_angles current_orientation(::aerogripper_msgs::msg::ControllerState::_current_orientation_type arg)
  {
    msg_.current_orientation = std::move(arg);
    return Init_ControllerState_current_euler_angles(msg_);
  }

private:
  ::aerogripper_msgs::msg::ControllerState msg_;
};

class Init_ControllerState_current_velocity
{
public:
  explicit Init_ControllerState_current_velocity(::aerogripper_msgs::msg::ControllerState & msg)
  : msg_(msg)
  {}
  Init_ControllerState_current_orientation current_velocity(::aerogripper_msgs::msg::ControllerState::_current_velocity_type arg)
  {
    msg_.current_velocity = std::move(arg);
    return Init_ControllerState_current_orientation(msg_);
  }

private:
  ::aerogripper_msgs::msg::ControllerState msg_;
};

class Init_ControllerState_current_position
{
public:
  explicit Init_ControllerState_current_position(::aerogripper_msgs::msg::ControllerState & msg)
  : msg_(msg)
  {}
  Init_ControllerState_current_velocity current_position(::aerogripper_msgs::msg::ControllerState::_current_position_type arg)
  {
    msg_.current_position = std::move(arg);
    return Init_ControllerState_current_velocity(msg_);
  }

private:
  ::aerogripper_msgs::msg::ControllerState msg_;
};

class Init_ControllerState_desired_torque
{
public:
  explicit Init_ControllerState_desired_torque(::aerogripper_msgs::msg::ControllerState & msg)
  : msg_(msg)
  {}
  Init_ControllerState_current_position desired_torque(::aerogripper_msgs::msg::ControllerState::_desired_torque_type arg)
  {
    msg_.desired_torque = std::move(arg);
    return Init_ControllerState_current_position(msg_);
  }

private:
  ::aerogripper_msgs::msg::ControllerState msg_;
};

class Init_ControllerState_desired_thrust
{
public:
  explicit Init_ControllerState_desired_thrust(::aerogripper_msgs::msg::ControllerState & msg)
  : msg_(msg)
  {}
  Init_ControllerState_desired_torque desired_thrust(::aerogripper_msgs::msg::ControllerState::_desired_thrust_type arg)
  {
    msg_.desired_thrust = std::move(arg);
    return Init_ControllerState_desired_torque(msg_);
  }

private:
  ::aerogripper_msgs::msg::ControllerState msg_;
};

class Init_ControllerState_desired_angular_velocity
{
public:
  explicit Init_ControllerState_desired_angular_velocity(::aerogripper_msgs::msg::ControllerState & msg)
  : msg_(msg)
  {}
  Init_ControllerState_desired_thrust desired_angular_velocity(::aerogripper_msgs::msg::ControllerState::_desired_angular_velocity_type arg)
  {
    msg_.desired_angular_velocity = std::move(arg);
    return Init_ControllerState_desired_thrust(msg_);
  }

private:
  ::aerogripper_msgs::msg::ControllerState msg_;
};

class Init_ControllerState_desired_acceleration
{
public:
  explicit Init_ControllerState_desired_acceleration(::aerogripper_msgs::msg::ControllerState & msg)
  : msg_(msg)
  {}
  Init_ControllerState_desired_angular_velocity desired_acceleration(::aerogripper_msgs::msg::ControllerState::_desired_acceleration_type arg)
  {
    msg_.desired_acceleration = std::move(arg);
    return Init_ControllerState_desired_angular_velocity(msg_);
  }

private:
  ::aerogripper_msgs::msg::ControllerState msg_;
};

class Init_ControllerState_dt
{
public:
  explicit Init_ControllerState_dt(::aerogripper_msgs::msg::ControllerState & msg)
  : msg_(msg)
  {}
  Init_ControllerState_desired_acceleration dt(::aerogripper_msgs::msg::ControllerState::_dt_type arg)
  {
    msg_.dt = std::move(arg);
    return Init_ControllerState_desired_acceleration(msg_);
  }

private:
  ::aerogripper_msgs::msg::ControllerState msg_;
};

class Init_ControllerState_timestamp
{
public:
  Init_ControllerState_timestamp()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_ControllerState_dt timestamp(::aerogripper_msgs::msg::ControllerState::_timestamp_type arg)
  {
    msg_.timestamp = std::move(arg);
    return Init_ControllerState_dt(msg_);
  }

private:
  ::aerogripper_msgs::msg::ControllerState msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::aerogripper_msgs::msg::ControllerState>()
{
  return aerogripper_msgs::msg::builder::Init_ControllerState_timestamp();
}

}  // namespace aerogripper_msgs

#endif  // AEROGRIPPER_MSGS__MSG__DETAIL__CONTROLLER_STATE__BUILDER_HPP_
