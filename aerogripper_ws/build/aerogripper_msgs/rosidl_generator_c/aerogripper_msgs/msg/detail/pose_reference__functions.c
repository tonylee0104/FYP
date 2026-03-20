// generated from rosidl_generator_c/resource/idl__functions.c.em
// with input from aerogripper_msgs:msg/PoseReference.idl
// generated code does not contain a copyright notice
#include "aerogripper_msgs/msg/detail/pose_reference__functions.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "rcutils/allocator.h"


// Include directives for member types
// Member `position`
#include "geometry_msgs/msg/detail/point__functions.h"
// Member `linear_velocity`
// Member `linear_acceleration`
// Member `angular_velocity`
#include "geometry_msgs/msg/detail/vector3__functions.h"
// Member `orientation`
#include "geometry_msgs/msg/detail/quaternion__functions.h"

bool
aerogripper_msgs__msg__PoseReference__init(aerogripper_msgs__msg__PoseReference * msg)
{
  if (!msg) {
    return false;
  }
  // position
  if (!geometry_msgs__msg__Point__init(&msg->position)) {
    aerogripper_msgs__msg__PoseReference__fini(msg);
    return false;
  }
  // linear_velocity
  if (!geometry_msgs__msg__Vector3__init(&msg->linear_velocity)) {
    aerogripper_msgs__msg__PoseReference__fini(msg);
    return false;
  }
  // linear_acceleration
  if (!geometry_msgs__msg__Vector3__init(&msg->linear_acceleration)) {
    aerogripper_msgs__msg__PoseReference__fini(msg);
    return false;
  }
  // orientation
  if (!geometry_msgs__msg__Quaternion__init(&msg->orientation)) {
    aerogripper_msgs__msg__PoseReference__fini(msg);
    return false;
  }
  // angular_velocity
  if (!geometry_msgs__msg__Vector3__init(&msg->angular_velocity)) {
    aerogripper_msgs__msg__PoseReference__fini(msg);
    return false;
  }
  return true;
}

void
aerogripper_msgs__msg__PoseReference__fini(aerogripper_msgs__msg__PoseReference * msg)
{
  if (!msg) {
    return;
  }
  // position
  geometry_msgs__msg__Point__fini(&msg->position);
  // linear_velocity
  geometry_msgs__msg__Vector3__fini(&msg->linear_velocity);
  // linear_acceleration
  geometry_msgs__msg__Vector3__fini(&msg->linear_acceleration);
  // orientation
  geometry_msgs__msg__Quaternion__fini(&msg->orientation);
  // angular_velocity
  geometry_msgs__msg__Vector3__fini(&msg->angular_velocity);
}

bool
aerogripper_msgs__msg__PoseReference__are_equal(const aerogripper_msgs__msg__PoseReference * lhs, const aerogripper_msgs__msg__PoseReference * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // position
  if (!geometry_msgs__msg__Point__are_equal(
      &(lhs->position), &(rhs->position)))
  {
    return false;
  }
  // linear_velocity
  if (!geometry_msgs__msg__Vector3__are_equal(
      &(lhs->linear_velocity), &(rhs->linear_velocity)))
  {
    return false;
  }
  // linear_acceleration
  if (!geometry_msgs__msg__Vector3__are_equal(
      &(lhs->linear_acceleration), &(rhs->linear_acceleration)))
  {
    return false;
  }
  // orientation
  if (!geometry_msgs__msg__Quaternion__are_equal(
      &(lhs->orientation), &(rhs->orientation)))
  {
    return false;
  }
  // angular_velocity
  if (!geometry_msgs__msg__Vector3__are_equal(
      &(lhs->angular_velocity), &(rhs->angular_velocity)))
  {
    return false;
  }
  return true;
}

bool
aerogripper_msgs__msg__PoseReference__copy(
  const aerogripper_msgs__msg__PoseReference * input,
  aerogripper_msgs__msg__PoseReference * output)
{
  if (!input || !output) {
    return false;
  }
  // position
  if (!geometry_msgs__msg__Point__copy(
      &(input->position), &(output->position)))
  {
    return false;
  }
  // linear_velocity
  if (!geometry_msgs__msg__Vector3__copy(
      &(input->linear_velocity), &(output->linear_velocity)))
  {
    return false;
  }
  // linear_acceleration
  if (!geometry_msgs__msg__Vector3__copy(
      &(input->linear_acceleration), &(output->linear_acceleration)))
  {
    return false;
  }
  // orientation
  if (!geometry_msgs__msg__Quaternion__copy(
      &(input->orientation), &(output->orientation)))
  {
    return false;
  }
  // angular_velocity
  if (!geometry_msgs__msg__Vector3__copy(
      &(input->angular_velocity), &(output->angular_velocity)))
  {
    return false;
  }
  return true;
}

aerogripper_msgs__msg__PoseReference *
aerogripper_msgs__msg__PoseReference__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  aerogripper_msgs__msg__PoseReference * msg = (aerogripper_msgs__msg__PoseReference *)allocator.allocate(sizeof(aerogripper_msgs__msg__PoseReference), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(aerogripper_msgs__msg__PoseReference));
  bool success = aerogripper_msgs__msg__PoseReference__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
aerogripper_msgs__msg__PoseReference__destroy(aerogripper_msgs__msg__PoseReference * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    aerogripper_msgs__msg__PoseReference__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
aerogripper_msgs__msg__PoseReference__Sequence__init(aerogripper_msgs__msg__PoseReference__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  aerogripper_msgs__msg__PoseReference * data = NULL;

  if (size) {
    data = (aerogripper_msgs__msg__PoseReference *)allocator.zero_allocate(size, sizeof(aerogripper_msgs__msg__PoseReference), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = aerogripper_msgs__msg__PoseReference__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        aerogripper_msgs__msg__PoseReference__fini(&data[i - 1]);
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
aerogripper_msgs__msg__PoseReference__Sequence__fini(aerogripper_msgs__msg__PoseReference__Sequence * array)
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
      aerogripper_msgs__msg__PoseReference__fini(&array->data[i]);
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

aerogripper_msgs__msg__PoseReference__Sequence *
aerogripper_msgs__msg__PoseReference__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  aerogripper_msgs__msg__PoseReference__Sequence * array = (aerogripper_msgs__msg__PoseReference__Sequence *)allocator.allocate(sizeof(aerogripper_msgs__msg__PoseReference__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = aerogripper_msgs__msg__PoseReference__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
aerogripper_msgs__msg__PoseReference__Sequence__destroy(aerogripper_msgs__msg__PoseReference__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    aerogripper_msgs__msg__PoseReference__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
aerogripper_msgs__msg__PoseReference__Sequence__are_equal(const aerogripper_msgs__msg__PoseReference__Sequence * lhs, const aerogripper_msgs__msg__PoseReference__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!aerogripper_msgs__msg__PoseReference__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
aerogripper_msgs__msg__PoseReference__Sequence__copy(
  const aerogripper_msgs__msg__PoseReference__Sequence * input,
  aerogripper_msgs__msg__PoseReference__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(aerogripper_msgs__msg__PoseReference);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    aerogripper_msgs__msg__PoseReference * data =
      (aerogripper_msgs__msg__PoseReference *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!aerogripper_msgs__msg__PoseReference__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          aerogripper_msgs__msg__PoseReference__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!aerogripper_msgs__msg__PoseReference__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}
