cmake_minimum_required(VERSION 3.1.0 FATAL_ERROR)
project (HandDetection)
## include(${CMAKE_CURRENT_SOURCE_DIR}/test.cmake)

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# libs
find_package (OpenCV REQUIRED)

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# sources
file(GLOB_RECURSE HandDetection_SOURCES "src/*.cpp")
file(GLOB_RECURSE HandDetection_HEADERS "src/*.hpp")

set (HandDetection_INCLUDE_DIRS "")
foreach (_headerFile ${HandDetection_HEADERS})
    get_filename_component(_dir ${_headerFile} PATH)
    list (APPEND HandDetection_INCLUDE_DIRS ${_dir})
endforeach()
list(REMOVE_DUPLICATES HandDetection_INCLUDE_DIRS)

include_directories(${HandDetection_INCLUDE_DIRS})
add_executable (HandDetection ${HandDetection_SOURCES})

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# linking
target_link_libraries (HandDetection ${OpenCV_LIBS})

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# compiler features / version
set(needed_features
    cxx_strong_enums
    cxx_constexpr
    cxx_auto_type)
target_compile_features(HandDetection PRIVATE ${needed_features})
