cmake_minimum_required(VERSION 3.8)
project(aerogripper_simulator)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclpy REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)

# Install Python package
ament_python_install_package(${PROJECT_NAME})

# Install executable script
install(
  PROGRAMS ${PROJECT_NAME}/simulator.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME simulator
)

install(
  PROGRAMS ${PROJECT_NAME}/px4_bridge.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME px4_bridge
)

# Install other Python modules
install(
  FILES ${PROJECT_NAME}/utils.py
  DESTINATION lib/${PROJECT_NAME}/${PROJECT_NAME}
)

# Install resource files
install(
  DIRECTORY config launch resource
  DESTINATION share/${PROJECT_NAME}
)

ament_package()