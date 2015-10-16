cmake_minimum_required(VERSION 3.1.0 FATAL_ERROR)
project (ps-eye)

find_package (OpenCV REQUIRED)
find_package(libusb-1.0 REQUIRED)

include(ExternalProject)
set(PS3EYEDriver_INSTALL_DIR "${CMAKE_CURRENT_BINARY_DIR}/PS3EYEDriver")
ExternalProject_Add(PS3EYEDriver
  GIT_REPOSITORY https://github.com/inspirit/PS3EYEDriver
  PREFIX ${PS3EYEDriver_INSTALL_DIR}
  INSTALL_COMMAND ""
  BUILD_COMMAND ""
  CONFIGURE_COMMAND "")

set(SOURCES
  main.cpp
  ${PS3EYEDriver_INSTALL_DIR}/src/PS3EYEDriver/src/ps3eye.cpp
  ${PS3EYEDriver_INSTALL_DIR}/src/PS3EYEDriver/src/ps3eye.h)
include_directories(${PS3EYEDriver_INSTALL_DIR}/src/PS3EYEDriver/src)
add_executable(ps-eye ${SOURCES})

target_link_libraries (ps-eye ${OpenCV_LIBS})
target_link_libraries(ps-eye ${LIBUSB_1_LIBRARIES})

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# compiler features / version
set(needed_features cxx_strong_enums cxx_constexpr cxx_auto_type)
target_compile_features(ps-eye PRIVATE ${needed_features})
