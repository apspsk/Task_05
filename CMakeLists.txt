cmake_minimum_required(VERSION 3.8)

project(armor_detector)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# 查找依赖包
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(image_transport REQUIRED)
find_package(OpenCV REQUIRED)

# 设置包含目录
include_directories(
  include
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# 创建相机驱动可执行文件
add_executable(hik_camera_node 
  src/hik_camera_node.cpp
)

# 设置编译标准
target_compile_features(hik_camera_node PUBLIC c_std_99 cxx_std_17)

# 添加依赖
ament_target_dependencies(
  hik_camera_node
  rclcpp
  sensor_msgs
  cv_bridge
  image_transport
)

target_link_libraries(hik_camera_node
  ${OpenCV_LIBS}
)

# 创建装甲板检测可执行文件
add_executable(armor_detector_node 
  src/armor_detector_node.cpp
  src/armor_detector.cpp
)

# 设置编译标准
target_compile_features(armor_detector_node PUBLIC c_std_99 cxx_std_17)

# 添加依赖
ament_target_dependencies(
  armor_detector_node
  rclcpp
  sensor_msgs
  geometry_msgs
  std_msgs
  visualization_msgs
  cv_bridge
  image_transport
)

target_link_libraries(armor_detector_node
  ${OpenCV_LIBS}
)

# 海康SDK配置
set(HIK_SDK_DIR "/opt/MVS")
set(HIK_SDK_INCLUDE_DIR "${HIK_SDK_DIR}/include")
set(HIK_SDK_LIB_DIR "${HIK_SDK_DIR}/lib/64")

if(EXISTS ${HIK_SDK_INCLUDE_DIR} AND EXISTS ${HIK_SDK_LIB_DIR})
  message(STATUS "Found Hikvision SDK at ${HIK_SDK_DIR}")
  target_include_directories(hik_camera_node PUBLIC ${HIK_SDK_INCLUDE_DIR})
  target_link_libraries(hik_camera_node 
    ${HIK_SDK_LIB_DIR}/libMvCameraControl.so
    ${HIK_SDK_LIB_DIR}/libMvCameraControlWrapper.so
  )
  # 设置运行时库路径
  set_target_properties(hik_camera_node PROPERTIES 
    BUILD_RPATH "${HIK_SDK_LIB_DIR}" 
    INSTALL_RPATH "${HIK_SDK_LIB_DIR}"
  )
else()
  message(WARNING "Hikvision SDK not found at ${HIK_SDK_DIR}")
endif()

# 安装目标
install(TARGETS 
  hik_camera_node
  armor_detector_node
  DESTINATION lib/${PROJECT_NAME}
)

# 安装启动文件
install(DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch
  FILES_MATCHING PATTERN "*.py"
)

# 安装配置文件
install(DIRECTORY config/
  DESTINATION share/${PROJECT_NAME}/config
  FILES_MATCHING PATTERN "*.yaml"
)

# 安装RViz配置文件
install(
  FILES config/armor_detector.rviz
  DESTINATION share/${PROJECT_NAME}/config
)

# 测试配置
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()

