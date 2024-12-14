# cmake_minimum_required(VERSION 3.8)
# project(aws_audibot_controller)

# # Default to C++14
# if(NOT CMAKE_CXX_STANDARD)
#   set(CMAKE_CXX_STANDARD 14)
# endif()

# find_package(ament_cmake REQUIRED)

# ament_package()
cmake_minimum_required(VERSION 3.8)
project(aws_audibot_controller)

# Default to C++17
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

# Ensure dependencies are found
find_package(ament_cmake REQUIRED)
find_package(rclpy REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(std_msgs REQUIRED)

# Install Python scripts
install(
  DIRECTORY scripts/
  DESTINATION lib/${PROJECT_NAME}
  USE_SOURCE_PERMISSIONS
)

# Install launch files
install(
  DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch
)

# Install configuration files, if any
install(
  DIRECTORY config/
  DESTINATION share/${PROJECT_NAME}/config
)

# Add package.xml metadata
ament_package()
