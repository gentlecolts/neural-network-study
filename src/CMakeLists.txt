#find_package(nlohmann_json REQUIRED)

add_executable(${PROJECT_NAME}
	main.cpp
	perceptron.cpp multlayerperceptron.cpp
)
target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_17)

#target_link_libraries(${PROJECT_NAME} bots)
#target_link_libraries(${PROJECT_NAME} nlohmann_json)

