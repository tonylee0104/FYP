// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from aerogripper_msgs:msg/ControllerState.idl
// generated code does not contain a copyright notice

#ifndef AEROGRIPPER_MSGS__MSG__DETAIL__CONTROLLER_STATE__STRUCT_HPP_
#define AEROGRIPPER_MSGS__MSG__DETAIL__CONTROLLER_STATE__STRUCT_HPP_

#include <algorithm>
#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "rosidl_runtime_cpp/bounded_vector.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


// Include directives for member types
// Member 'timestamp'
#include "builtin_interfaces/msg/detail/time__struct.hpp"
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
#include "geometry_msgs/msg/detail/vector3__struct.hpp"
// Member 'current_position'
// Member 'reference_position'
#include "geometry_msgs/msg/detail/point__struct.hpp"
// Member 'current_orientation'
// Member 'reference_orientation'
#include "geometry_msgs/msg/detail/quaternion__struct.hpp"

#ifndef _WIN32
# define DEPRECATED__aerogripper_msgs__msg__ControllerState __attribute__((deprecated))
#else
# define DEPRECATED__aerogripper_msgs__msg__ControllerState __declspec(deprecated)
#endif

namespace aerogripper_msgs
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct ControllerState_
{
  using Type = ControllerState_<ContainerAllocator>;

  explicit ControllerState_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : timestamp(_init),
    desired_acceleration(_init),
    desired_angular_velocity(_init),
    desired_thrust(_init),
    desired_torque(_init),
    current_position(_init),
    current_velocity(_init),
    current_orientation(_init),
    current_euler_angles(_init),
    current_angular_velocity(_init),
    reference_position(_init),
    reference_velocity(_init),
    reference_orientation(_init),
    reference_euler_angles(_init),
    tether_force_bodyframe(_init)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->dt = 0.0;
      std::fill<typename std::array<double, 4>::iterator, double>(this->throttle_output.begin(), this->throttle_output.end(), 0.0);
    }
  }

  explicit ControllerState_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : timestamp(_alloc, _init),
    desired_acceleration(_alloc, _init),
    desired_angular_velocity(_alloc, _init),
    desired_thrust(_alloc, _init),
    desired_torque(_alloc, _init),
    current_position(_alloc, _init),
    current_velocity(_alloc, _init),
    current_orientation(_alloc, _init),
    current_euler_angles(_alloc, _init),
    current_angular_velocity(_alloc, _init),
    reference_position(_alloc, _init),
    reference_velocity(_alloc, _init),
    reference_orientation(_alloc, _init),
    reference_euler_angles(_alloc, _init),
    throttle_output(_alloc),
    tether_force_bodyframe(_alloc, _init)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->dt = 0.0;
      std::fill<typename std::array<double, 4>::iterator, double>(this->throttle_output.begin(), this->throttle_output.end(), 0.0);
    }
  }

  // field types and members
  using _timestamp_type =
    builtin_interfaces::msg::Time_<ContainerAllocator>;
  _timestamp_type timestamp;
  using _dt_type =
    double;
  _dt_type dt;
  using _desired_acceleration_type =
    geometry_msgs::msg::Vector3_<ContainerAllocator>;
  _desired_acceleration_type desired_acceleration;
  using _desired_angular_velocity_type =
    geometry_msgs::msg::Vector3_<ContainerAllocator>;
  _desired_angular_velocity_type desired_angular_velocity;
  using _desired_thrust_type =
    geometry_msgs::msg::Vector3_<ContainerAllocator>;
  _desired_thrust_type desired_thrust;
  using _desired_torque_type =
    geometry_msgs::msg::Vector3_<ContainerAllocator>;
  _desired_torque_type desired_torque;
  using _current_position_type =
    geometry_msgs::msg::Point_<ContainerAllocator>;
  _current_position_type current_position;
  using _current_velocity_type =
    geometry_msgs::msg::Vector3_<ContainerAllocator>;
  _current_velocity_type current_velocity;
  using _current_orientation_type =
    geometry_msgs::msg::Quaternion_<ContainerAllocator>;
  _current_orientation_type current_orientation;
  using _current_euler_angles_type =
    geometry_msgs::msg::Vector3_<ContainerAllocator>;
  _current_euler_angles_type current_euler_angles;
  using _current_angular_velocity_type =
    geometry_msgs::msg::Vector3_<ContainerAllocator>;
  _current_angular_velocity_type current_angular_velocity;
  using _reference_position_type =
    geometry_msgs::msg::Point_<ContainerAllocator>;
  _reference_position_type reference_position;
  using _reference_velocity_type =
    geometry_msgs::msg::Vector3_<ContainerAllocator>;
  _reference_velocity_type reference_velocity;
  using _reference_orientation_type =
    geometry_msgs::msg::Quaternion_<ContainerAllocator>;
  _reference_orientation_type reference_orientation;
  using _reference_euler_angles_type =
    geometry_msgs::msg::Vector3_<ContainerAllocator>;
  _reference_euler_angles_type reference_euler_angles;
  using _throttle_output_type =
    std::array<double, 4>;
  _throttle_output_type throttle_output;
  using _tether_force_bodyframe_type =
    geometry_msgs::msg::Vector3_<ContainerAllocator>;
  _tether_force_bodyframe_type tether_force_bodyframe;

  // setters for named parameter idiom
  Type & set__timestamp(
    const builtin_interfaces::msg::Time_<ContainerAllocator> & _arg)
  {
    this->timestamp = _arg;
    return *this;
  }
  Type & set__dt(
    const double & _arg)
  {
    this->dt = _arg;
    return *this;
  }
  Type & set__desired_acceleration(
    const geometry_msgs::msg::Vector3_<ContainerAllocator> & _arg)
  {
    this->desired_acceleration = _arg;
    return *this;
  }
  Type & set__desired_angular_velocity(
    const geometry_msgs::msg::Vector3_<ContainerAllocator> & _arg)
  {
    this->desired_angular_velocity = _arg;
    return *this;
  }
  Type & set__desired_thrust(
    const geometry_msgs::msg::Vector3_<ContainerAllocator> & _arg)
  {
    this->desired_thrust = _arg;
    return *this;
  }
  Type & set__desired_torque(
    const geometry_msgs::msg::Vector3_<ContainerAllocator> & _arg)
  {
    this->desired_torque = _arg;
    return *this;
  }
  Type & set__current_position(
    const geometry_msgs::msg::Point_<ContainerAllocator> & _arg)
  {
    this->current_position = _arg;
    return *this;
  }
  Type & set__current_velocity(
    const geometry_msgs::msg::Vector3_<ContainerAllocator> & _arg)
  {
    this->current_velocity = _arg;
    return *this;
  }
  Type & set__current_orientation(
    const geometry_msgs::msg::Quaternion_<ContainerAllocator> & _arg)
  {
    this->current_orientation = _arg;
    return *this;
  }
  Type & set__current_euler_angles(
    const geometry_msgs::msg::Vector3_<ContainerAllocator> & _arg)
  {
    this->current_euler_angles = _arg;
    return *this;
  }
  Type & set__current_angular_velocity(
    const geometry_msgs::msg::Vector3_<ContainerAllocator> & _arg)
  {
    this->current_angular_velocity = _arg;
    return *this;
  }
  Type & set__reference_position(
    const geometry_msgs::msg::Point_<ContainerAllocator> & _arg)
  {
    this->reference_position = _arg;
    return *this;
  }
  Type & set__reference_velocity(
    const geometry_msgs::msg::Vector3_<ContainerAllocator> & _arg)
  {
    this->reference_velocity = _arg;
    return *this;
  }
  Type & set__reference_orientation(
    const geometry_msgs::msg::Quaternion_<ContainerAllocator> & _arg)
  {
    this->reference_orientation = _arg;
    return *this;
  }
  Type & set__reference_euler_angles(
    const geometry_msgs::msg::Vector3_<ContainerAllocator> & _arg)
  {
    this->reference_euler_angles = _arg;
    return *this;
  }
  Type & set__throttle_output(
    const std::array<double, 4> & _arg)
  {
    this->throttle_output = _arg;
    return *this;
  }
  Type & set__tether_force_bodyframe(
    const geometry_msgs::msg::Vector3_<ContainerAllocator> & _arg)
  {
    this->tether_force_bodyframe = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    aerogripper_msgs::msg::ControllerState_<ContainerAllocator> *;
  using ConstRawPtr =
    const aerogripper_msgs::msg::ControllerState_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<aerogripper_msgs::msg::ControllerState_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<aerogripper_msgs::msg::ControllerState_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      aerogripper_msgs::msg::ControllerState_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<aerogripper_msgs::msg::ControllerState_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      aerogripper_msgs::msg::ControllerState_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<aerogripper_msgs::msg::ControllerState_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<aerogripper_msgs::msg::ControllerState_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<aerogripper_msgs::msg::ControllerState_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__aerogripper_msgs__msg__ControllerState
    std::shared_ptr<aerogripper_msgs::msg::ControllerState_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__aerogripper_msgs__msg__ControllerState
    std::shared_ptr<aerogripper_msgs::msg::ControllerState_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const ControllerState_ & other) const
  {
    if (this->timestamp != other.timestamp) {
      return false;
    }
    if (this->dt != other.dt) {
      return false;
    }
    if (this->desired_acceleration != other.desired_acceleration) {
      return false;
    }
    if (this->desired_angular_velocity != other.desired_angular_velocity) {
      return false;
    }
    if (this->desired_thrust != other.desired_thrust) {
      return false;
    }
    if (this->desired_torque != other.desired_torque) {
      return false;
    }
    if (this->current_position != other.current_position) {
      return false;
    }
    if (this->current_velocity != other.current_velocity) {
      return false;
    }
    if (this->current_orientation != other.current_orientation) {
      return false;
    }
    if (this->current_euler_angles != other.current_euler_angles) {
      return false;
    }
    if (this->current_angular_velocity != other.current_angular_velocity) {
      return false;
    }
    if (this->reference_position != other.reference_position) {
      return false;
    }
    if (this->reference_velocity != other.reference_velocity) {
      return false;
    }
    if (this->reference_orientation != other.reference_orientation) {
      return false;
    }
    if (this->reference_euler_angles != other.reference_euler_angles) {
      return false;
    }
    if (this->throttle_output != other.throttle_output) {
      return false;
    }
    if (this->tether_force_bodyframe != other.tether_force_bodyframe) {
      return false;
    }
    return true;
  }
  bool operator!=(const ControllerState_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct ControllerState_

// alias to use template instance with default allocator
using ControllerState =
  aerogripper_msgs::msg::ControllerState_<std::allocator<void>>;

// constant definitions

}  // namespace msg

}  // namespace aerogripper_msgs

#endif  // AEROGRIPPER_MSGS__MSG__DETAIL__CONTROLLER_STATE__STRUCT_HPP_
