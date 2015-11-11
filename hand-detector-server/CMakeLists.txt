set (HAND-DETECTOR-SERVER_ROOT ${CMAKE_CURRENT_SOURCE_DIR})
set (HAND-DETECTOR-SERVER_INCLUDE ${HAND-DETECTOR-SERVER_ROOT})
set (HAND-DETECTOR-SERVER_BUILD_ROOT ${CMAKE_CURRENT_BINARY_DIR})
set (HAND-DETECTOR-SERVER_BIN ${HAND-DETECTOR-SERVER_BUILD_ROOT}/bin)
set (HAND-DETECTOR-SERVER_LIB ${HAND-DETECTOR-SERVER_BUILD_ROOT}/lib)

find_package (Jsoncpp REQUIRED)
find_package(Boost COMPONENTS system REQUIRED)

include(ExternalProject)

externalproject_add(
    l2l-cpp
    PREFIX "${HAND-DETECTOR-SERVER_BUILD_ROOT}"
    GIT_REPOSITORY https://github.com/rksm/l2l-cpp.git
    BINARY_DIR "${HAND-DETECTOR-SERVER_BUILD_ROOT}"
    INSTALL_DIR "${HAND-DETECTOR-SERVER_LIB}")

add_executable (hand-detector-server
  "main.cpp"
  "services.cpp"
)

add_dependencies(hand-detector-server l2l-cpp)

target_include_directories(hand-detector-server PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${L2L-CPP_INCLUDE_DIR})

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# linking

target_link_libraries (hand-detector-server
  hand-detector
  "${HAND-DETECTOR-SERVER_LIB}/${CMAKE_STATIC_LIBRARY_PREFIX}l2l-cpp${CMAKE_STATIC_LIBRARY_SUFFIX}"
  ${Jsoncpp_LIBRARY}
  ${Boost_SYSTEM_LIBRARY}
)

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
target_compile_features(hand-detector-server PRIVATE "cxx_auto_type")
