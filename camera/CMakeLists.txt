find_package(freenect2 REQUIRED)

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

find_package(Eigen3 REQUIRED)

add_definitions(-DEPTH_REG_OPENCL)
add_definitions(-DREG_OPENCL_FILE="$${CMAKE_CURRENT_SOURCE_DIR}/depth_registration.cl")

add_library(camera
  "camera.hpp"
  "camera.cpp"
  "kinect-sensor.cpp"
  "kinect-camera.hpp"
  "kinect-camera.cpp"
)

target_link_libraries (camera ${freenect2_LIBRARIES})
target_include_directories(camera PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
target_compile_features(camera PRIVATE "cxx_auto_type")
