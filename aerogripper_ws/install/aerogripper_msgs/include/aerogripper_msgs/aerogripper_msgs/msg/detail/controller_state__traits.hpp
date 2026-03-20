// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from aerogripper_msgs:msg/ControllerState.idl
// generated code does not contain a copyright notice

#ifndef AEROGRIPPER_MSGS__MSG__DETAIL__CONTROLLER_STATE__TRAITS_HPP_
#define AEROGRIPPER_MSGS__MSG__DETAIL__CONTROLLER_STATE__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "aerogripper_msgs/msg/detail/controller_state__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

// Include directives for member types
// Member 'timestamp'
#include "builtin_interfaces/msg/detail/time__traits.hpp"
// Member 'desired_acceleration'
// Member 'desired_angular_velocity'
// Member 'desired_thrust'
// Member 'desired_torque'
// Member 'current_velocity'
// Member 'current_euler_angles'
// Member 'current_angular_velocity'
// Member 'reference_velocity'
// Member 'reference_euler_angles'
// Member 'tether_force_bodyframe'
#include "geometry_msgs/msg/detail/vector3__traits.hpp"
// Member 'current_position'
// Member 'reference_position'
#include "geometry_msgs/msg/detail/point__traits.hpp"
// Member 'current_orientation'
// Member 'reference_orientation'
#include "geometry_msgs/msg/detail/quaternion__traits.hpp"

namespace aerogripper_msgs
{

namespace msg
{

inline void to_flow_style_yaml(
  const ControllerState & msg,
  std::ostream & out)
{
  out << "{";
  // member: timestamp
  {
    out << "timestamp: ";
    to_flow_style_yaml(msg.timestamp, out);
    out << ", ";
  }

  // member: dt
  {
    out << "dt: ";
    rosidl_generator_traits::value_to_yaml(msg.dt, out);
    out << ", ";
  }

  // member: desired_acceleration
  {
    out << "desired_acceleration: ";
    to_flow_style_yaml(msg.desired_acceleration, out);
    out << ", ";
  }

  // member: desired_angular_velocity
  {
    out << "desired_angular_velocity: ";
    to_flow_style_yaml(msg.desired_angular_velocity, out);
    out << ", ";
  }

  // member: desired_thrust
  {
    out << "desired_thrust: ";
    to_flow_style_yaml(msg.desired_thrust, out);
    out << ", ";
  }

  // member: desired_torque
  {
    out << "desired_torque: ";
    to_flow_style_yaml(msg.desired_torque, out);
    out << ", ";
  }

  // member: current_position
  {
    out << "current_position: ";
    to_flow_style_yaml(msg.current_position, out);
    out << ", ";
  }

  // member: current_velocity
  {
    out << "current_velocity: ";
    to_flow_style_yaml(msg.current_velocity, out);
    out << ", ";
  }

  // member: current_orientation
  {
    out << "current_orientation: ";
    to_flow_style_yaml(msg.current_orientation, out);
    out << ", ";
  }

  // member: current_euler_angles
  {
    out << "current_euler_angles: ";
    to_flow_style_yaml(msg.current_euler_angles, out);
    out << ", ";
  }

  // member: current_angular_velocity
  {
    out << "current_angular_velocity: ";
    to_flow_style_yaml(msg.current_angular_velocity, out);
    out << ", ";
  }

  // member: reference_position
  {
    out << "reference_position: ";
    to_flow_style_yaml(msg.reference_position, out);
    out << ", ";
  }

  // member: reference_velocity
  {
    out << "reference_velocity: ";
    to_flow_style_yaml(msg.reference_velocity, out);
    out << ", ";
  }

  // member: reference_orientation
  {
    out << "reference_orientation: ";
    to_flow_style_yaml(msg.reference_orientation, out);
    out << ", ";
  }

  // member: reference_euler_angles
  {
    out << "reference_euler_angles: ";
    to_flow_style_yaml(msg.reference_euler_angles, out);
    out << ", ";
  }

  // member: throttle_output
  {
    if (msg.throttle_output.size() == 0) {
      out << "throttle_output: []";
    } else {
      out << "throttle_output: [";
      size_t pending_items = msg.throttle_output.size();
      for (auto item : msg.throttle_output) {
        rosidl_generator_traits::value_to_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
    out << ", ";
  }

  // member: tether_force_bodyframe
  {
    out << "tether_force_bodyframe: ";
    to_flow_style_yaml(msg.tether_force_bodyframe, out);
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const ControllerState & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: timestamp
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "timestamp:\n";
    to_block_style_yaml(msg.timestamp, out, indentation + 2);
  }

  // member: dt
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "dt: ";
    rosidl_generator_traits::value_to_yaml(msg.dt, out);
    out << "\n";
  }

  // member: desired_acceleration
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "desired_acceleration:\n";
    to_block_style_yaml(msg.desired_acceleration, out, indentation + 2);
  }

  // member: desired_angular_velocity
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "desired_angular_velocity:\n";
    to_block_style_yaml(msg.desired_angular_velocity, out, indentation + 2);
  }

  // member: desired_thrust
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "desired_thrust:\n";
    to_block_style_yaml(msg.desired_thrust, out, indentation + 2);
  }

  // member: desired_torque
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "desired_torque:\n";
    to_block_style_yaml(msg.desired_torque, out, indentation + 2);
  }

  // member: current_position
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "current_position:\n";
    to_block_style_yaml(msg.current_position, out, indentation + 2);
  }

  // member: current_velocity
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "current_velocity:\n";
    to_block_style_yaml(msg.current_velocity, out, indentation + 2);
  }

  // member: current_orientation
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "current_orientation:\n";
    to_block_style_yaml(msg.current_orientation, out, indentation + 2);
  }

  // member: current_euler_angles
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "current_euler_angles:\n";
    to_block_style_yaml(msg.current_euler_angles, out, indentation + 2);
  }

  // member: current_angular_velocity
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "current_angular_velocity:\n";
    to_block_style_yaml(msg.current_angular_velocity, out, indentation + 2);
  }

  // member: reference_position
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "reference_position:\n";
    to_block_style_yaml(msg.reference_position, out, indentation + 2);
  }

  // member: reference_velocity
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "reference_velocity:\n";
    to_block_style_yaml(msg.reference_velocity, out, indentation + 2);
  }

  // member: reference_orientation
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "reference_orientation:\n";
    to_block_style_yaml(msg.reference_orientation, out, indentation + 2);
  }

  // member: reference_euler_angles
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "reference_euler_angles:\n";
    to_block_style_yaml(msg.reference_euler_angles, out, indentation + 2);
  }

  // member: throttle_output
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.throttle_output.size() == 0) {
      out << "throttle_output: []\n";
    } else {
      out << "throttle_output:\n";
      for (auto item : msg.throttle_output) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "- ";
        rosidl_generator_traits::value_to_yaml(item, out);
        out << "\n";
      }
    }
  }

  // member: tether_force_bodyframe
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "tether_force_bodyframe:\n";
    to_block_style_yaml(msg.tether_force_bodyframe, out, indentation + 2);
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const ControllerState & msg, bool use_flow_style = false)
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
  const aerogripper_msgs::msg::ControllerState & msg,
  std::ostream & out, size_t indentation = 0)
{
  aerogripper_msgs::msg::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use aerogripper_msgs::msg::to_yaml() instead")]]
inline std::string to_yaml(const aerogripper_msgs::msg::ControllerState & msg)
{
  return aerogripper_msgs::msg::to_yaml(msg);
}

template<>
inline const char * data_type<aerogripper_msgs::msg::ControllerState>()
{
  return "aerogripper_msgs::msg::ControllerState";
}

template<>
inline const char * name<aerogripper_msgs::msg::ControllerState>()
{
  return "aerogripper_msgs/msg/ControllerState";
}

template<>
struct has_fixed_size<aerogripper_msgs::msg::ControllerState>
  : std::integral_constant<bool, has_fixed_size<builtin_interfaces::msg::Time>::value && has_fixed_size<geometry_msgs::msg::Point>::value && has_fixed_size<geometry_msgs::msg::Quaternion>::value && has_fixed_size<geometry_msgs::msg::Vector3>::value> {};

template<>
struct has_bounded_size<aerogripper_msgs::msg::ControllerState>
  : std::integral_constant<bool, has_bounded_size<builtin_interfaces::msg::Time>::value && has_bounded_size<geometry_msgs::msg::Point>::value && has_bounded_size<geometry_msgs::msg::Quaternion>::value && has_bounded_size<geometry_msgs::msg::Vector3>::value> {};

template<>
struct is_message<aerogripper_msgs::msg::ControllerState>
  : std::true_type {};

}  // namespace rosidl_generator_traits

#endif  // AEROGRIPPER_MSGS__MSG__DETAIL__CONTROLLER_STATE__TRAITS_HPP_
