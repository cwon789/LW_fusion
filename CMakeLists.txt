cmake_minimum_required(VERSION 3.5)
project(ekf_lidar_wheel)

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(Eigen3 REQUIRED)

include_directories(
  include
  ${EIGEN3_INCLUDE_DIR}
)

add_executable(ekf_node src/ekf_node.cpp src/EKF.cpp src/ICP.cpp)

ament_target_dependencies(ekf_node rclcpp sensor_msgs nav_msgs geometry_msgs)

install(TARGETS
  ekf_node
  DESTINATION lib/${PROJECT_NAME})

ament_package()
