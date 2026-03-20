// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from aerogripper_msgs:msg/PoseReference.idl
// generated code does not contain a copyright notice

#ifndef AEROGRIPPER_MSGS__MSG__DETAIL__POSE_REFERENCE__STRUCT_HPP_
#define AEROGRIPPER_MSGS__MSG__DETAIL__POSE_REFERENCE__STRUCT_HPP_

#include <algorithm>
#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "rosidl_runtime_cpp/bounded_vector.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


// Include directives for member types
// Member 'position'
#include "geometry_msgs/msg/detail/point__struct.hpp"
// Member 'linear_velocity'
// Member 'linear_acceleration'
// Member 'angular_velocity'
#include "geometry_msgs/msg/detail/vector3__struct.hpp"
// Member 'orientation'
#include "geometry_msgs/msg/detail/quaternion__struct.hpp"

#ifndef _WIN32
# define DEPRECATED__aerogripper_msgs__msg__PoseReference __attribute__((deprecated))
#else
# define DEPRECATED__aerogripper_msgs__msg__PoseReference __declspec(deprecated)
#endif

namespace aerogripper_msgs
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct PoseReference_
{
  using Type = PoseReference_<ContainerAllocator>;

  explicit PoseReference_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : position(_init),
    linear_velocity(_init),
    linear_acceleration(_init),
    orientation(_init),
    angular_velocity(_init)
  {
    (void)_init;
  }

  explicit PoseReference_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : position(_alloc, _init),
    linear_velocity(_alloc, _init),
    linear_acceleration(_alloc, _init),
    orientation(_alloc, _init),
    angular_velocity(_alloc, _init)
  {
    (void)_init;
  }

  // field types and members
  using _position_type =
    geometry_msgs::msg::Point_<ContainerAllocator>;
  _position_type position;
  using _linear_velocity_type =
    geometry_msgs::msg::Vector3_<ContainerAllocator>;
  _linear_velocity_type linear_velocity;
  using _linear_acceleration_type =
    geometry_msgs::msg::Vector3_<ContainerAllocator>;
  _linear_acceleration_type linear_acceleration;
  using _orientation_type =
    geometry_msgs::msg::Quaternion_<ContainerAllocator>;
  _orientation_type orientation;
  using _angular_velocity_type =
    geometry_msgs::msg::Vector3_<ContainerAllocator>;
  _angular_velocity_type angular_velocity;

  // setters for named parameter idiom
  Type & set__position(
    const geometry_msgs::msg::Point_<ContainerAllocator> & _arg)
  {
    this->position = _arg;
    return *this;
  }
  Type & set__linear_velocity(
    const geometry_msgs::msg::Vector3_<ContainerAllocator> & _arg)
  {
    this->linear_velocity = _arg;
    return *this;
  }
  Type & set__linear_acceleration(
    const geometry_msgs::msg::Vector3_<ContainerAllocator> & _arg)
  {
    this->linear_acceleration = _arg;
    return *this;
  }
  Type & set__orientation(
    const geometry_msgs::msg::Quaternion_<ContainerAllocator> & _arg)
  {
    this->orientation = _arg;
    return *this;
  }
  Type & set__angular_velocity(
    const geometry_msgs::msg::Vector3_<ContainerAllocator> & _arg)
  {
    this->angular_velocity = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    aerogripper_msgs::msg::PoseReference_<ContainerAllocator> *;
  using ConstRawPtr =
    const aerogripper_msgs::msg::PoseReference_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<aerogripper_msgs::msg::PoseReference_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<aerogripper_msgs::msg::PoseReference_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      aerogripper_msgs::msg::PoseReference_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<aerogripper_msgs::msg::PoseReference_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      aerogripper_msgs::msg::PoseReference_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<aerogripper_msgs::msg::PoseReference_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<aerogripper_msgs::msg::PoseReference_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<aerogripper_msgs::msg::PoseReference_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__aerogripper_msgs__msg__PoseReference
    std::shared_ptr<aerogripper_msgs::msg::PoseReference_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__aerogripper_msgs__msg__PoseReference
    std::shared_ptr<aerogripper_msgs::msg::PoseReference_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const PoseReference_ & other) const
  {
    if (this->position != other.position) {
      return false;
    }
    if (this->linear_velocity != other.linear_velocity) {
      return false;
    }
    if (this->linear_acceleration != other.linear_acceleration) {
      return false;
    }
    if (this->orientation != other.orientation) {
      return false;
    }
    if (this->angular_velocity != other.angular_velocity) {
      return false;
    }
    return true;
  }
  bool operator!=(const PoseReference_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct PoseReference_

// alias to use template instance with default allocator
using PoseReference =
  aerogripper_msgs::msg::PoseReference_<std::allocator<void>>;

// constant definitions

}  // namespace msg

}  // namespace aerogripper_msgs

#endif  // AEROGRIPPER_MSGS__MSG__DETAIL__POSE_REFERENCE__STRUCT_HPP_
