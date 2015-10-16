cmake_minimum_required(VERSION 3.1.0 FATAL_ERROR)
project (hand-detector)
## include(${CMAKE_CURRENT_SOURCE_DIR}/test.cmake)

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# libs
find_package (OpenCV REQUIRED)
find_package (Jsoncpp REQUIRED)

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# sources
file(GLOB_RECURSE hand-detector_SOURCES "*.cpp")
file(GLOB_RECURSE hand-detector_HEADERS "*.hpp")

set (hand-detector_INCLUDE_DIRS "")
foreach (_headerFile ${hand-detector_HEADERS})
    get_filename_component(_dir ${_headerFile} PATH)
    list (APPEND hand-detector_INCLUDE_DIRS ${_dir})
endforeach()
list(REMOVE_DUPLICATES hand-detector_INCLUDE_DIRS)

include_directories(${hand-detector_INCLUDE_DIRS})

add_executable (hand-detector ${hand-detector_SOURCES})

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# linking
target_link_libraries (hand-detector ${OpenCV_LIBS})
target_link_libraries (hand-detector ${Jsoncpp_LIBRARY})

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# compiler features / version
set(needed_features cxx_strong_enums cxx_constexpr cxx_auto_type)
target_compile_features(hand-detector PRIVATE ${needed_features})
