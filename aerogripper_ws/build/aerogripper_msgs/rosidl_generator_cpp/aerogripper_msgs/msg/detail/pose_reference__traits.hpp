// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from aerogripper_msgs:msg/PoseReference.idl
// generated code does not contain a copyright notice

#ifndef AEROGRIPPER_MSGS__MSG__DETAIL__POSE_REFERENCE__TRAITS_HPP_
#define AEROGRIPPER_MSGS__MSG__DETAIL__POSE_REFERENCE__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "aerogripper_msgs/msg/detail/pose_reference__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

// Include directives for member types
// Member 'position'
#include "geometry_msgs/msg/detail/point__traits.hpp"
// Member 'linear_velocity'
// Member 'linear_acceleration'
// Member 'angular_velocity'
#include "geometry_msgs/msg/detail/vector3__traits.hpp"
// Member 'orientation'
#include "geometry_msgs/msg/detail/quaternion__traits.hpp"

namespace aerogripper_msgs
{

namespace msg
{

inline void to_flow_style_yaml(
  const PoseReference & msg,
  std::ostream & out)
{
  out << "{";
  // member: position
  {
    out << "position: ";
    to_flow_style_yaml(msg.position, out);
    out << ", ";
  }

  // member: linear_velocity
  {
    out << "linear_velocity: ";
    to_flow_style_yaml(msg.linear_velocity, out);
    out << ", ";
  }

  // member: linear_acceleration
  {
    out << "linear_acceleration: ";
    to_flow_style_yaml(msg.linear_acceleration, out);
    out << ", ";
  }

  // member: orientation
  {
    out << "orientation: ";
    to_flow_style_yaml(msg.orientation, out);
    out << ", ";
  }

  // member: angular_velocity
  {
    out << "angular_velocity: ";
    to_flow_style_yaml(msg.angular_velocity, out);
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const PoseReference & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: position
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "position:\n";
    to_block_style_yaml(msg.position, out, indentation + 2);
  }

  // member: linear_velocity
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "linear_velocity:\n";
    to_block_style_yaml(msg.linear_velocity, out, indentation + 2);
  }

  // member: linear_acceleration
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "linear_acceleration:\n";
    to_block_style_yaml(msg.linear_acceleration, out, indentation + 2);
  }

  // member: orientation
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "orientation:\n";
    to_block_style_yaml(msg.orientation, out, indentation + 2);
  }

  // member: angular_velocity
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "angular_velocity:\n";
    to_block_style_yaml(msg.angular_velocity, out, indentation + 2);
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const PoseReference & msg, bool use_flow_style = false)
{
  std::ostringstream out;
  if (use_flow_style) {
    to_flow_style_yaml(msg, out);
  } else {
    to_block_style_yaml(msg, out);
  }
  return out.str();
}

}  // namespace msg

}  // namespace aerogripper_msgs

namespace rosidl_generator_traits
{

[[deprecated("use aerogripper_msgs::msg::to_block_style_yaml() instead")]]
inline void to_yaml(
  const aerogripper_msgs::msg::PoseReference & msg,
  std::ostream & out, size_t indentation = 0)
{
  aerogripper_msgs::msg::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use aerogripper_msgs::msg::to_yaml() instead")]]
inline std::string to_yaml(const aerogripper_msgs::msg::PoseReference & msg)
{
  return aerogripper_msgs::msg::to_yaml(msg);
}

template<>
inline const char * data_type<aerogripper_msgs::msg::PoseReference>()
{
  return "aerogripper_msgs::msg::PoseReference";
}

template<>
inline const char * name<aerogripper_msgs::msg::PoseReference>()
{
  return "aerogripper_msgs/msg/PoseReference";
}

template<>
struct has_fixed_size<aerogripper_msgs::msg::PoseReference>
  : std::integral_constant<bool, has_fixed_size<geometry_msgs::msg::Point>::value && has_fixed_size<geometry_msgs::msg::Quaternion>::value && has_fixed_size<geometry_msgs::msg::Vector3>::value> {};

template<>
struct has_bounded_size<aerogripper_msgs::msg::PoseReference>
  : std::integral_constant<bool, has_bounded_size<geometry_msgs::msg::Point>::value && has_bounded_size<geometry_msgs::msg::Quaternion>::value && has_bounded_size<geometry_msgs::msg::Vector3>::value> {};

template<>
struct is_message<aerogripper_msgs::msg::PoseReference>
  : std::true_type {};

}  // namespace rosidl_generator_traits

#endif  // AEROGRIPPER_MSGS__MSG__DETAIL__POSE_REFERENCE__TRAITS_HPP_
