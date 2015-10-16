cmake_minimum_required(VERSION 3.1.0 FATAL_ERROR)
project (LiveTable)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")
# set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")

add_subdirectory(hand-detector)
add_subdirectory(ps-eye)
