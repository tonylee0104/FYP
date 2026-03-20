// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from aerogripper_msgs:msg/PoseReference.idl
// generated code does not contain a copyright notice

#ifndef AEROGRIPPER_MSGS__MSG__DETAIL__POSE_REFERENCE__BUILDER_HPP_
#define AEROGRIPPER_MSGS__MSG__DETAIL__POSE_REFERENCE__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "aerogripper_msgs/msg/detail/pose_reference__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace aerogripper_msgs
{

namespace msg
{

namespace builder
{

class Init_PoseReference_angular_velocity
{
public:
  explicit Init_PoseReference_angular_velocity(::aerogripper_msgs::msg::PoseReference & msg)
  : msg_(msg)
  {}
  ::aerogripper_msgs::msg::PoseReference angular_velocity(::aerogripper_msgs::msg::PoseReference::_angular_velocity_type arg)
  {
    msg_.angular_velocity = std::move(arg);
    return std::move(msg_);
  }

private:
  ::aerogripper_msgs::msg::PoseReference msg_;
};

class Init_PoseReference_orientation
{
public:
  explicit Init_PoseReference_orientation(::aerogripper_msgs::msg::PoseReference & msg)
  : msg_(msg)
  {}
  Init_PoseReference_angular_velocity orientation(::aerogripper_msgs::msg::PoseReference::_orientation_type arg)
  {
    msg_.orientation = std::move(arg);
    return Init_PoseReference_angular_velocity(msg_);
  }

private:
  ::aerogripper_msgs::msg::PoseReference msg_;
};

class Init_PoseReference_linear_acceleration
{
public:
  explicit Init_PoseReference_linear_acceleration(::aerogripper_msgs::msg::PoseReference & msg)
  : msg_(msg)
  {}
  Init_PoseReference_orientation linear_acceleration(::aerogripper_msgs::msg::PoseReference::_linear_acceleration_type arg)
  {
    msg_.linear_acceleration = std::move(arg);
    return Init_PoseReference_orientation(msg_);
  }

private:
  ::aerogripper_msgs::msg::PoseReference msg_;
};

class Init_PoseReference_linear_velocity
{
public:
  explicit Init_PoseReference_linear_velocity(::aerogripper_msgs::msg::PoseReference & msg)
  : msg_(msg)
  {}
  Init_PoseReference_linear_acceleration linear_velocity(::aerogripper_msgs::msg::PoseReference::_linear_velocity_type arg)
  {
    msg_.linear_velocity = std::move(arg);
    return Init_PoseReference_linear_acceleration(msg_);
  }

private:
  ::aerogripper_msgs::msg::PoseReference msg_;
};

class Init_PoseReference_position
{
public:
  Init_PoseReference_position()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_PoseReference_linear_velocity position(::aerogripper_msgs::msg::PoseReference::_position_type arg)
  {
    msg_.position = std::move(arg);
    return Init_PoseReference_linear_velocity(msg_);
  }

private:
  ::aerogripper_msgs::msg::PoseReference msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::aerogripper_msgs::msg::PoseReference>()
{
  return aerogripper_msgs::msg::builder::Init_PoseReference_position();
}

}  // namespace aerogripper_msgs

#endif  // AEROGRIPPER_MSGS__MSG__DETAIL__POSE_REFERENCE__BUILDER_HPP_
