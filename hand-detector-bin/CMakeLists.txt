# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# libs
# find_package (OpenCV REQUIRED)
# find_package (Jsoncpp REQUIRED)


add_executable (hand-detector-bin
  "main.cpp"
  "test-files.cpp"
)

target_include_directories(hand-detector-bin PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# linking
target_link_libraries (hand-detector-bin hand-detector)

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
target_compile_features(hand-detector-bin PRIVATE "cxx_auto_type")
