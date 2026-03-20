// NOLINT: This file starts with a BOM since it contain non-ASCII characters
// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from aerogripper_msgs:msg/ControllerState.idl
// generated code does not contain a copyright notice

#ifndef AEROGRIPPER_MSGS__MSG__DETAIL__CONTROLLER_STATE__STRUCT_H_
#define AEROGRIPPER_MSGS__MSG__DETAIL__CONTROLLER_STATE__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

// Include directives for member types
// Member 'timestamp'
#include "builtin_interfaces/msg/detail/time__struct.h"
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
#include "geometry_msgs/msg/detail/vector3__struct.h"
// Member 'current_position'
// Member 'reference_position'
#include "geometry_msgs/msg/detail/point__struct.h"
// Member 'current_orientation'
// Member 'reference_orientation'
#include "geometry_msgs/msg/detail/quaternion__struct.h"

/// Struct defined in msg/ControllerState in the package aerogripper_msgs.
/**
  * 控制器状态消息
  * 用于发布控制器的内部状态数据，便于调试和监控
 */
typedef struct aerogripper_msgs__msg__ControllerState
{
  /// 时间戳
  builtin_interfaces__msg__Time timestamp;
  /// 时间步长 (dt)
  double dt;
  /// 期望加速度 (a_des)
  geometry_msgs__msg__Vector3 desired_acceleration;
  /// 期望角速度 (omega_des)
  geometry_msgs__msg__Vector3 desired_angular_velocity;
  /// 期望推力 (T_des)
  geometry_msgs__msg__Vector3 desired_thrust;
  /// 期望力矩 (tau_des)
  geometry_msgs__msg__Vector3 desired_torque;
  /// 当前位置
  geometry_msgs__msg__Point current_position;
  /// 当前速度
  geometry_msgs__msg__Vector3 current_velocity;
  /// 当前姿态四元数
  geometry_msgs__msg__Quaternion current_orientation;
  /// 当前姿态欧拉角 (roll, pitch, yaw)
  geometry_msgs__msg__Vector3 current_euler_angles;
  /// 当前角速度
  geometry_msgs__msg__Vector3 current_angular_velocity;
  /// 参考位置
  geometry_msgs__msg__Point reference_position;
  /// 参考速度
  geometry_msgs__msg__Vector3 reference_velocity;
  /// 参考姿态四元数
  geometry_msgs__msg__Quaternion reference_orientation;
  /// 参考姿态欧拉角 (roll, pitch, yaw)
  geometry_msgs__msg__Vector3 reference_euler_angles;
  /// 油门输出值
  double throttle_output[4];
  /// 绳子拉力观测值 (在机体坐标系下)
  geometry_msgs__msg__Vector3 tether_force_bodyframe;
} aerogripper_msgs__msg__ControllerState;

// Struct for a sequence of aerogripper_msgs__msg__ControllerState.
typedef struct aerogripper_msgs__msg__ControllerState__Sequence
{
  aerogripper_msgs__msg__ControllerState * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} aerogripper_msgs__msg__ControllerState__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // AEROGRIPPER_MSGS__MSG__DETAIL__CONTROLLER_STATE__STRUCT_H_
