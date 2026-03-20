cmake_minimum_required(VERSION 3.5)
project(aerogripper_control)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(px4_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(eigen3_cmake_module REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(aerogripper_msgs REQUIRED)

# 创建库文件，包含Mixer和TetherObserver类
add_library(aerogripper_control_lib
  src/mixer.cpp
  src/tether_observer.cpp
)

# 设置库的包含目录
target_include_directories(aerogripper_control_lib PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${EIGEN3_INCLUDE_DIRS}
)

# 链接库的依赖
ament_target_dependencies(aerogripper_control_lib
  rclcpp
  px4_msgs
  geometry_msgs
  Eigen3
  aerogripper_msgs
)

# 主控制节点
add_executable(aerogripper_control_node
  src/main.cpp
  src/pose_controller.cpp
)

# 链接库到主节点
target_link_libraries(aerogripper_control_node
  aerogripper_control_lib
)

# 设置主节点的包含目录
target_include_directories(aerogripper_control_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${EIGEN3_INCLUDE_DIRS}
)

ament_target_dependencies(aerogripper_control_node
  rclcpp
  px4_msgs
  geometry_msgs
  Eigen3
  aerogripper_msgs
)

install(TARGETS aerogripper_control_node
  DESTINATION lib/${PROJECT_NAME}
)

# 集成测试可执行文件
add_executable(test_integration
  test_integration.cpp
)

# 链接库到测试程序
target_link_libraries(test_integration
  aerogripper_control_lib
)

# 设置测试程序的包含目录
target_include_directories(test_integration PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${EIGEN3_INCLUDE_DIRS}
)

ament_target_dependencies(test_integration
  rclcpp
  px4_msgs
  geometry_msgs
  Eigen3
  aerogripper_msgs
)

install(TARGETS test_integration
  DESTINATION lib/${PROJECT_NAME}
)

# 安装库文件
install(TARGETS aerogripper_control_lib
  DESTINATION lib/${PROJECT_NAME}
)

add_executable(motor_test_node src/motor_test.cpp)

# 为motor_test_node也设置包含目录
target_include_directories(motor_test_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${EIGEN3_INCLUDE_DIRS}
)

ament_target_dependencies(motor_test_node rclcpp px4_msgs geometry_msgs Eigen3)
install(TARGETS motor_test_node DESTINATION lib/${PROJECT_NAME})

# 安装launch文件
install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}
)

# 安装配置文件
install(DIRECTORY
  config
  DESTINATION share/${PROJECT_NAME}
)

ament_package()