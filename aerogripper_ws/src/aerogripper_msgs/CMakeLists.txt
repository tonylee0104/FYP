cmake_minimum_required(VERSION 3.8)
project(aerogripper_msgs)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(rclcpp REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/PoseReference.msg"
  "msg/ControllerState.msg"
  DEPENDENCIES geometry_msgs
)

# 添加测试发布者节点
add_executable(test_publisher src/test_publisher.cpp)
ament_target_dependencies(test_publisher rclcpp)
rosidl_target_interfaces(test_publisher ${PROJECT_NAME} "rosidl_typesupport_cpp")

# 添加ControllerState测试发布者节点
add_executable(test_controller_state src/test_controller_state.cpp)
ament_target_dependencies(test_controller_state rclcpp)
rosidl_target_interfaces(test_controller_state ${PROJECT_NAME} "rosidl_typesupport_cpp")

install(TARGETS test_publisher test_controller_state
  DESTINATION lib/${PROJECT_NAME}
)

# 安装Python脚本
install(PROGRAMS
  scripts/controller_state_subscriber.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
