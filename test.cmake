# function (add_executable name ....)
#   # do stuff to remember this target
#   message (STATUS "FOOOO")
#   _add_executable (name ....)
# endfunction (add_executable)
# 

function(add_executable name)
  message(STATUS "enter add_executable override1: name='${name}'")
  _add_executable(${name} ${ARGN})
  get_target_property (out_dir ${name} RUNTIME_OUTPUT_DIRECTORY)
  get_property (suffix TARGET ${name} PROPERTY SUFFIX)
  message(STATUS " exit add_executable override1 ....  ${out_dir}/${name} ${suffix} ???")
endfunction()
