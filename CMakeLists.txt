if(${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR})
	message(FATAL_ERROR "DO NOT BUILD in-tree.")
endif()

cmake_minimum_required(VERSION 3.20)

project(neural-study)

add_subdirectory(src)

install(TARGETS neural-study RUNTIME DESTINATION bin)

target_include_directories(${PROJECT_NAME} PUBLIC include)
