add_executable(ninfer
  cli/main.cpp
  cli/options.cpp)
target_include_directories(ninfer PRIVATE
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_SOURCE_DIR}/src
  ${PROJECT_SOURCE_DIR}/third_party)
target_link_libraries(ninfer PRIVATE
  ninfer_engine
  ninfer_product_load_progress
  ninfer_product_prompt_input)

add_executable(ninfer-serve serve/main.cpp)
target_include_directories(ninfer-serve PRIVATE
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_SOURCE_DIR}/src
  ${PROJECT_SOURCE_DIR}/third_party
  ${PROJECT_SOURCE_DIR}/third_party/cpp-httplib)
target_link_libraries(ninfer-serve PRIVATE ninfer_serve ninfer_product_load_progress)

if(WIN32 AND DSTORAGE_DLLS)
  add_custom_command(TARGET ninfer POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${DSTORAGE_DLLS} $<TARGET_FILE_DIR:ninfer>
  )
  add_custom_command(TARGET ninfer-serve POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${DSTORAGE_DLLS} $<TARGET_FILE_DIR:ninfer-serve>
  )
endif()
