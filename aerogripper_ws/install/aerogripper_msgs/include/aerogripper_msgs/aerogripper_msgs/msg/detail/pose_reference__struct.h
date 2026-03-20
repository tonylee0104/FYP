// NOLINT: This file starts with a BOM since it contain non-ASCII characters
// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from aerogripper_msgs:msg/PoseReference.idl
// generated code does not contain a copyright notice

#ifndef AEROGRIPPER_MSGS__MSG__DETAIL__POSE_REFERENCE__STRUCT_H_
#define AEROGRIPPER_MSGS__MSG__DETAIL__POSE_REFERENCE__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

// Include directives for member types
// Member 'position'
#include "geometry_msgs/msg/detail/point__struct.h"
// Member 'linear_velocity'
// Member 'linear_acceleration'
// Member 'angular_velocity'
#include "geometry_msgs/msg/detail/vector3__struct.h"
// Member 'orientation'
#include "geometry_msgs/msg/detail/quaternion__struct.h"

/// Struct defined in msg/PoseReference in the package aerogripper_msgs.
/**
  * 位置参考 (x, y, z)
 */
typedef struct aerogripper_msgs__msg__PoseReference
{
  geometry_msgs__msg__Point position;
  /// 线性速度参考 (vx, vy, vz)
  geometry_msgs__msg__Vector3 linear_velocity;
  /// 线性加速度参考 (ax, ay, az)
  geometry_msgs__msg__Vector3 linear_acceleration;
  /// 姿态四元数参考 (w, x, y, z)
  geometry_msgs__msg__Quaternion orientation;
  /// 角速度参考 (wx, wy, wz)
  geometry_msgs__msg__Vector3 angular_velocity;
} aerogripper_msgs__msg__PoseReference;

// Struct for a sequence of aerogripper_msgs__msg__PoseReference.
typedef struct aerogripper_msgs__msg__PoseReference__Sequence
{
  aerogripper_msgs__msg__PoseReference * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} aerogripper_msgs__msg__PoseReference__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // AEROGRIPPER_MSGS__MSG__DETAIL__POSE_REFERENCE__STRUCT_H_
