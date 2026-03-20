// generated from rosidl_typesupport_introspection_c/resource/idl__type_support.c.em
// with input from aerogripper_msgs:msg/PoseReference.idl
// generated code does not contain a copyright notice

#include <stddef.h>
#include "aerogripper_msgs/msg/detail/pose_reference__rosidl_typesupport_introspection_c.h"
#include "aerogripper_msgs/msg/rosidl_typesupport_introspection_c__visibility_control.h"
#include "rosidl_typesupport_introspection_c/field_types.h"
#include "rosidl_typesupport_introspection_c/identifier.h"
#include "rosidl_typesupport_introspection_c/message_introspection.h"
#include "aerogripper_msgs/msg/detail/pose_reference__functions.h"
#include "aerogripper_msgs/msg/detail/pose_reference__struct.h"


// Include directives for member types
// Member `position`
#include "geometry_msgs/msg/point.h"
// Member `position`
#include "geometry_msgs/msg/detail/point__rosidl_typesupport_introspection_c.h"
// Member `linear_velocity`
// Member `linear_acceleration`
// Member `angular_velocity`
#include "geometry_msgs/msg/vector3.h"
// Member `linear_velocity`
// Member `linear_acceleration`
// Member `angular_velocity`
#include "geometry_msgs/msg/detail/vector3__rosidl_typesupport_introspection_c.h"
// Member `orientation`
#include "geometry_msgs/msg/quaternion.h"
// Member `orientation`
#include "geometry_msgs/msg/detail/quaternion__rosidl_typesupport_introspection_c.h"

#ifdef __cplusplus
extern "C"
{
#endif

void aerogripper_msgs__msg__PoseReference__rosidl_typesupport_introspection_c__PoseReference_init_function(
  void * message_memory, enum rosidl_runtime_c__message_initialization _init)
{
  // TODO(karsten1987): initializers are not yet implemented for typesupport c
  // see https://github.com/ros2/ros2/issues/397
  (void) _init;
  aerogripper_msgs__msg__PoseReference__init(message_memory);
}

void aerogripper_msgs__msg__PoseReference__rosidl_typesupport_introspection_c__PoseReference_fini_function(void * message_memory)
{
  aerogripper_msgs__msg__PoseReference__fini(message_memory);
}

static rosidl_typesupport_introspection_c__MessageMember aerogripper_msgs__msg__PoseReference__rosidl_typesupport_introspection_c__PoseReference_message_member_array[5] = {
  {
    "position",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(aerogripper_msgs__msg__PoseReference, position),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "linear_velocity",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(aerogripper_msgs__msg__PoseReference, linear_velocity),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "linear_acceleration",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(aerogripper_msgs__msg__PoseReference, linear_acceleration),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "orientation",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(aerogripper_msgs__msg__PoseReference, orientation),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "angular_velocity",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(aerogripper_msgs__msg__PoseReference, angular_velocity),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  }
};

static const rosidl_typesupport_introspection_c__MessageMembers aerogripper_msgs__msg__PoseReference__rosidl_typesupport_introspection_c__PoseReference_message_members = {
  "aerogripper_msgs__msg",  // message namespace
  "PoseReference",  // message name
  5,  // number of fields
  sizeof(aerogripper_msgs__msg__PoseReference),
  aerogripper_msgs__msg__PoseReference__rosidl_typesupport_introspection_c__PoseReference_message_member_array,  // message members
  aerogripper_msgs__msg__PoseReference__rosidl_typesupport_introspection_c__PoseReference_init_function,  // function to initialize message memory (memory has to be allocated)
  aerogripper_msgs__msg__PoseReference__rosidl_typesupport_introspection_c__PoseReference_fini_function  // function to terminate message instance (will not free memory)
};

// this is not const since it must be initialized on first access
// since C does not allow non-integral compile-time constants
static rosidl_message_type_support_t aerogripper_msgs__msg__PoseReference__rosidl_typesupport_introspection_c__PoseReference_message_type_support_handle = {
  0,
  &aerogripper_msgs__msg__PoseReference__rosidl_typesupport_introspection_c__PoseReference_message_members,
  get_message_typesupport_handle_function,
};

ROSIDL_TYPESUPPORT_INTROSPECTION_C_EXPORT_aerogripper_msgs
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, aerogripper_msgs, msg, PoseReference)() {
  aerogripper_msgs__msg__PoseReference__rosidl_typesupport_introspection_c__PoseReference_message_member_array[0].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, geometry_msgs, msg, Point)();
  aerogripper_msgs__msg__PoseReference__rosidl_typesupport_introspection_c__PoseReference_message_member_array[1].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, geometry_msgs, msg, Vector3)();
  aerogripper_msgs__msg__PoseReference__rosidl_typesupport_introspection_c__PoseReference_message_member_array[2].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, geometry_msgs, msg, Vector3)();
  aerogripper_msgs__msg__PoseReference__rosidl_typesupport_introspection_c__PoseReference_message_member_array[3].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, geometry_msgs, msg, Quaternion)();
  aerogripper_msgs__msg__PoseReference__rosidl_typesupport_introspection_c__PoseReference_message_member_array[4].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, geometry_msgs, msg, Vector3)();
  if (!aerogripper_msgs__msg__PoseReference__rosidl_typesupport_introspection_c__PoseReference_message_type_support_handle.typesupport_identifier) {
    aerogripper_msgs__msg__PoseReference__rosidl_typesupport_introspection_c__PoseReference_message_type_support_handle.typesupport_identifier =
      rosidl_typesupport_introspection_c__identifier;
  }
  return &aerogripper_msgs__msg__PoseReference__rosidl_typesupport_introspection_c__PoseReference_message_type_support_handle;
}
#ifdef __cplusplus
}
#endif
