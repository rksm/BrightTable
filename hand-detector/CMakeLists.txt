# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# libs
find_package (OpenCV REQUIRED)
find_package (Jsoncpp REQUIRED)

add_library(hand-detector
  "vision/cv-helper.cpp"
  "vision/hand-detection-json.cpp"
  "vision/hand-detection.cpp"
  "vision/quad-transform.cpp"
  "vision/screen-detection.cpp")

target_include_directories(hand-detector PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# linking
target_link_libraries (hand-detector ${OpenCV_LIBS})
target_link_libraries (hand-detector ${Jsoncpp_LIBRARY})

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
target_compile_features(hand-detector PRIVATE "cxx_auto_type")
