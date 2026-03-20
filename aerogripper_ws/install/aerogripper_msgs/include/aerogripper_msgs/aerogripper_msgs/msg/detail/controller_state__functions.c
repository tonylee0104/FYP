// generated from rosidl_generator_c/resource/idl__functions.c.em
// with input from aerogripper_msgs:msg/ControllerState.idl
// generated code does not contain a copyright notice
#include "aerogripper_msgs/msg/detail/controller_state__functions.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "rcutils/allocator.h"


// Include directives for member types
// Member `timestamp`
#include "builtin_interfaces/msg/detail/time__functions.h"
// Member `desired_acceleration`
// Member `desired_angular_velocity`
// Member `desired_thrust`
// Member `desired_torque`
// Member `current_velocity`
// Member `current_euler_angles`
// Member `current_angular_velocity`
// Member `reference_velocity`
// Member `reference_euler_angles`
// Member `tether_force_bodyframe`
#include "geometry_msgs/msg/detail/vector3__functions.h"
// Member `current_position`
// Member `reference_position`
#include "geometry_msgs/msg/detail/point__functions.h"
// Member `current_orientation`
// Member `reference_orientation`
#include "geometry_msgs/msg/detail/quaternion__functions.h"

bool
aerogripper_msgs__msg__ControllerState__init(aerogripper_msgs__msg__ControllerState * msg)
{
  if (!msg) {
    return false;
  }
  // timestamp
  if (!builtin_interfaces__msg__Time__init(&msg->timestamp)) {
    aerogripper_msgs__msg__ControllerState__fini(msg);
    return false;
  }
  // dt
  // desired_acceleration
  if (!geometry_msgs__msg__Vector3__init(&msg->desired_acceleration)) {
    aerogripper_msgs__msg__ControllerState__fini(msg);
    return false;
  }
  // desired_angular_velocity
  if (!geometry_msgs__msg__Vector3__init(&msg->desired_angular_velocity)) {
    aerogripper_msgs__msg__ControllerState__fini(msg);
    return false;
  }
  // desired_thrust
  if (!geometry_msgs__msg__Vector3__init(&msg->desired_thrust)) {
    aerogripper_msgs__msg__ControllerState__fini(msg);
    return false;
  }
  // desired_torque
  if (!geometry_msgs__msg__Vector3__init(&msg->desired_torque)) {
    aerogripper_msgs__msg__ControllerState__fini(msg);
    return false;
  }
  // current_position
  if (!geometry_msgs__msg__Point__init(&msg->current_position)) {
    aerogripper_msgs__msg__ControllerState__fini(msg);
    return false;
  }
  // current_velocity
  if (!geometry_msgs__msg__Vector3__init(&msg->current_velocity)) {
    aerogripper_msgs__msg__ControllerState__fini(msg);
    return false;
  }
  // current_orientation
  if (!geometry_msgs__msg__Quaternion__init(&msg->current_orientation)) {
    aerogripper_msgs__msg__ControllerState__fini(msg);
    return false;
  }
  // current_euler_angles
  if (!geometry_msgs__msg__Vector3__init(&msg->current_euler_angles)) {
    aerogripper_msgs__msg__ControllerState__fini(msg);
    return false;
  }
  // current_angular_velocity
  if (!geometry_msgs__msg__Vector3__init(&msg->current_angular_velocity)) {
    aerogripper_msgs__msg__ControllerState__fini(msg);
    return false;
  }
  // reference_position
  if (!geometry_msgs__msg__Point__init(&msg->reference_position)) {
    aerogripper_msgs__msg__ControllerState__fini(msg);
    return false;
  }
  // reference_velocity
  if (!geometry_msgs__msg__Vector3__init(&msg->reference_velocity)) {
    aerogripper_msgs__msg__ControllerState__fini(msg);
    return false;
  }
  // reference_orientation
  if (!geometry_msgs__msg__Quaternion__init(&msg->reference_orientation)) {
    aerogripper_msgs__msg__ControllerState__fini(msg);
    return false;
  }
  // reference_euler_angles
  if (!geometry_msgs__msg__Vector3__init(&msg->reference_euler_angles)) {
    aerogripper_msgs__msg__ControllerState__fini(msg);
    return false;
  }
  // throttle_output
  // tether_force_bodyframe
  if (!geometry_msgs__msg__Vector3__init(&msg->tether_force_bodyframe)) {
    aerogripper_msgs__msg__ControllerState__fini(msg);
    return false;
  }
  return true;
}

void
aerogripper_msgs__msg__ControllerState__fini(aerogripper_msgs__msg__ControllerState * msg)
{
  if (!msg) {
    return;
  }
  // timestamp
  builtin_interfaces__msg__Time__fini(&msg->timestamp);
  // dt
  // desired_acceleration
  geometry_msgs__msg__Vector3__fini(&msg->desired_acceleration);
  // desired_angular_velocity
  geometry_msgs__msg__Vector3__fini(&msg->desired_angular_velocity);
  // desired_thrust
  geometry_msgs__msg__Vector3__fini(&msg->desired_thrust);
  // desired_torque
  geometry_msgs__msg__Vector3__fini(&msg->desired_torque);
  // current_position
  geometry_msgs__msg__Point__fini(&msg->current_position);
  // current_velocity
  geometry_msgs__msg__Vector3__fini(&msg->current_velocity);
  // current_orientation
  geometry_msgs__msg__Quaternion__fini(&msg->current_orientation);
  // current_euler_angles
  geometry_msgs__msg__Vector3__fini(&msg->current_euler_angles);
  // current_angular_velocity
  geometry_msgs__msg__Vector3__fini(&msg->current_angular_velocity);
  // reference_position
  geometry_msgs__msg__Point__fini(&msg->reference_position);
  // reference_velocity
  geometry_msgs__msg__Vector3__fini(&msg->reference_velocity);
  // reference_orientation
  geometry_msgs__msg__Quaternion__fini(&msg->reference_orientation);
  // reference_euler_angles
  geometry_msgs__msg__Vector3__fini(&msg->reference_euler_angles);
  // throttle_output
  // tether_force_bodyframe
  geometry_msgs__msg__Vector3__fini(&msg->tether_force_bodyframe);
}

bool
aerogripper_msgs__msg__ControllerState__are_equal(const aerogripper_msgs__msg__ControllerState * lhs, const aerogripper_msgs__msg__ControllerState * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // timestamp
  if (!builtin_interfaces__msg__Time__are_equal(
      &(lhs->timestamp), &(rhs->timestamp)))
  {
    return false;
  }
  // dt
  if (lhs->dt != rhs->dt) {
    return false;
  }
  // desired_acceleration
  if (!geometry_msgs__msg__Vector3__are_equal(
      &(lhs->desired_acceleration), &(rhs->desired_acceleration)))
  {
    return false;
  }
  // desired_angular_velocity
  if (!geometry_msgs__msg__Vector3__are_equal(
      &(lhs->desired_angular_velocity), &(rhs->desired_angular_velocity)))
  {
    return false;
  }
  // desired_thrust
  if (!geometry_msgs__msg__Vector3__are_equal(
      &(lhs->desired_thrust), &(rhs->desired_thrust)))
  {
    return false;
  }
  // desired_torque
  if (!geometry_msgs__msg__Vector3__are_equal(
      &(lhs->desired_torque), &(rhs->desired_torque)))
  {
    return false;
  }
  // current_position
  if (!geometry_msgs__msg__Point__are_equal(
      &(lhs->current_position), &(rhs->current_position)))
  {
    return false;
  }
  // current_velocity
  if (!geometry_msgs__msg__Vector3__are_equal(
      &(lhs->current_velocity), &(rhs->current_velocity)))
  {
    return false;
  }
  // current_orientation
  if (!geometry_msgs__msg__Quaternion__are_equal(
      &(lhs->current_orientation), &(rhs->current_orientation)))
  {
    return false;
  }
  // current_euler_angles
  if (!geometry_msgs__msg__Vector3__are_equal(
      &(lhs->current_euler_angles), &(rhs->current_euler_angles)))
  {
    return false;
  }
  // current_angular_velocity
  if (!geometry_msgs__msg__Vector3__are_equal(
      &(lhs->current_angular_velocity), &(rhs->current_angular_velocity)))
  {
    return false;
  }
  // reference_position
  if (!geometry_msgs__msg__Point__are_equal(
      &(lhs->reference_position), &(rhs->reference_position)))
  {
    return false;
  }
  // reference_velocity
  if (!geometry_msgs__msg__Vector3__are_equal(
      &(lhs->reference_velocity), &(rhs->reference_velocity)))
  {
    return false;
  }
  // reference_orientation
  if (!geometry_msgs__msg__Quaternion__are_equal(
      &(lhs->reference_orientation), &(rhs->reference_orientation)))
  {
    return false;
  }
  // reference_euler_angles
  if (!geometry_msgs__msg__Vector3__are_equal(
      &(lhs->reference_euler_angles), &(rhs->reference_euler_angles)))
  {
    return false;
  }
  // throttle_output
  for (size_t i = 0; i < 4; ++i) {
    if (lhs->throttle_output[i] != rhs->throttle_output[i]) {
      return false;
    }
  }
  // tether_force_bodyframe
  if (!geometry_msgs__msg__Vector3__are_equal(
      &(lhs->tether_force_bodyframe), &(rhs->tether_force_bodyframe)))
  {
    return false;
  }
  return true;
}

bool
aerogripper_msgs__msg__ControllerState__copy(
  const aerogripper_msgs__msg__ControllerState * input,
  aerogripper_msgs__msg__ControllerState * output)
{
  if (!input || !output) {
    return false;
  }
  // timestamp
  if (!builtin_interfaces__msg__Time__copy(
      &(input->timestamp), &(output->timestamp)))
  {
    return false;
  }
  // dt
  output->dt = input->dt;
  // desired_acceleration
  if (!geometry_msgs__msg__Vector3__copy(
      &(input->desired_acceleration), &(output->desired_acceleration)))
  {
    return false;
  }
  // desired_angular_velocity
  if (!geometry_msgs__msg__Vector3__copy(
      &(input->desired_angular_velocity), &(output->desired_angular_velocity)))
  {
    return false;
  }
  // desired_thrust
  if (!geometry_msgs__msg__Vector3__copy(
      &(input->desired_thrust), &(output->desired_thrust)))
  {
    return false;
  }
  // desired_torque
  if (!geometry_msgs__msg__Vector3__copy(
      &(input->desired_torque), &(output->desired_torque)))
  {
    return false;
  }
  // current_position
  if (!geometry_msgs__msg__Point__copy(
      &(input->current_position), &(output->current_position)))
  {
    return false;
  }
  // current_velocity
  if (!geometry_msgs__msg__Vector3__copy(
      &(input->current_velocity), &(output->current_velocity)))
  {
    return false;
  }
  // current_orientation
  if (!geometry_msgs__msg__Quaternion__copy(
      &(input->current_orientation), &(output->current_orientation)))
  {
    return false;
  }
  // current_euler_angles
  if (!geometry_msgs__msg__Vector3__copy(
      &(input->current_euler_angles), &(output->current_euler_angles)))
  {
    return false;
  }
  // current_angular_velocity
  if (!geometry_msgs__msg__Vector3__copy(
      &(input->current_angular_velocity), &(output->current_angular_velocity)))
  {
    return false;
  }
  // reference_position
  if (!geometry_msgs__msg__Point__copy(
      &(input->reference_position), &(output->reference_position)))
  {
    return false;
  }
  // reference_velocity
  if (!geometry_msgs__msg__Vector3__copy(
      &(input->reference_velocity), &(output->reference_velocity)))
  {
    return false;
  }
  // reference_orientation
  if (!geometry_msgs__msg__Quaternion__copy(
      &(input->reference_orientation), &(output->reference_orientation)))
  {
    return false;
  }
  // reference_euler_angles
  if (!geometry_msgs__msg__Vector3__copy(
      &(input->reference_euler_angles), &(output->reference_euler_angles)))
  {
    return false;
  }
  // throttle_output
  for (size_t i = 0; i < 4; ++i) {
    output->throttle_output[i] = input->throttle_output[i];
  }
  // tether_force_bodyframe
  if (!geometry_msgs__msg__Vector3__copy(
      &(input->tether_force_bodyframe), &(output->tether_force_bodyframe)))
  {
    return false;
  }
  return true;
}

aerogripper_msgs__msg__ControllerState *
aerogripper_msgs__msg__ControllerState__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  aerogripper_msgs__msg__ControllerState * msg = (aerogripper_msgs__msg__ControllerState *)allocator.allocate(sizeof(aerogripper_msgs__msg__ControllerState), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(aerogripper_msgs__msg__ControllerState));
  bool success = aerogripper_msgs__msg__ControllerState__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
aerogripper_msgs__msg__ControllerState__destroy(aerogripper_msgs__msg__ControllerState * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    aerogripper_msgs__msg__ControllerState__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
aerogripper_msgs__msg__ControllerState__Sequence__init(aerogripper_msgs__msg__ControllerState__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  aerogripper_msgs__msg__ControllerState * data = NULL;

  if (size) {
    data = (aerogripper_msgs__msg__ControllerState *)allocator.zero_allocate(size, sizeof(aerogripper_msgs__msg__ControllerState), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = aerogripper_msgs__msg__ControllerState__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        aerogripper_msgs__msg__ControllerState__fini(&data[i - 1]);
      }
      allocator.deallocate(data, allocator.state);
      return false;
    }
  }
  array->data = data;
  array->size = size;
  array->capacity = size;
  return true;
}

void
aerogripper_msgs__msg__ControllerState__Sequence__fini(aerogripper_msgs__msg__ControllerState__Sequence * array)
{
  if (!array) {
    return;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();

  if (array->data) {
    // ensure that data and capacity values are consistent
    assert(array->capacity > 0);
    // finalize all array elements
    for (size_t i = 0; i < array->capacity; ++i) {
      aerogripper_msgs__msg__ControllerState__fini(&array->data[i]);
    }
    allocator.deallocate(array->data, allocator.state);
    array->data = NULL;
    array->size = 0;
    array->capacity = 0;
  } else {
    // ensure that data, size, and capacity values are consistent
    assert(0 == array->size);
    assert(0 == array->capacity);
  }
}

aerogripper_msgs__msg__ControllerState__Sequence *
aerogripper_msgs__msg__ControllerState__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  aerogripper_msgs__msg__ControllerState__Sequence * array = (aerogripper_msgs__msg__ControllerState__Sequence *)allocator.allocate(sizeof(aerogripper_msgs__msg__ControllerState__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = aerogripper_msgs__msg__ControllerState__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
aerogripper_msgs__msg__ControllerState__Sequence__destroy(aerogripper_msgs__msg__ControllerState__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    aerogripper_msgs__msg__ControllerState__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
aerogripper_msgs__msg__ControllerState__Sequence__are_equal(const aerogripper_msgs__msg__ControllerState__Sequence * lhs, const aerogripper_msgs__msg__ControllerState__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!aerogripper_msgs__msg__ControllerState__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
aerogripper_msgs__msg__ControllerState__Sequence__copy(
  const aerogripper_msgs__msg__ControllerState__Sequence * input,
  aerogripper_msgs__msg__ControllerState__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(aerogripper_msgs__msg__ControllerState);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    aerogripper_msgs__msg__ControllerState * data =
      (aerogripper_msgs__msg__ControllerState *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!aerogripper_msgs__msg__ControllerState__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          aerogripper_msgs__msg__ControllerState__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!aerogripper_msgs__msg__ControllerState__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}
