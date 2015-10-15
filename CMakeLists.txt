cmake_minimum_required(VERSION 3.1.0 FATAL_ERROR)
project (LiveTable)
## include(${CMAKE_CURRENT_SOURCE_DIR}/test.cmake)

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# libs
find_package (OpenCV REQUIRED)

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# sources
file(GLOB_RECURSE LiveTable_SOURCES "src/*.cpp")
file(GLOB_RECURSE LiveTable_HEADERS "src/*.hpp")

set (LiveTable_INCLUDE_DIRS "")
foreach (_headerFile ${LiveTable_HEADERS})
    get_filename_component(_dir ${_headerFile} PATH)
    list (APPEND LiveTable_INCLUDE_DIRS ${_dir})
endforeach()
list(REMOVE_DUPLICATES LiveTable_INCLUDE_DIRS)

include_directories(${LiveTable_INCLUDE_DIRS})
add_executable (LiveTable ${LiveTable_SOURCES})

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# linking
target_link_libraries (LiveTable ${OpenCV_LIBS})

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# compiler features / version
set(needed_features
    cxx_strong_enums
    cxx_constexpr
    cxx_auto_type)
target_compile_features(LiveTable PRIVATE ${needed_features})
