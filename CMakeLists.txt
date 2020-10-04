# CMakeList.txt : CMake project for gtasa_taxi_sim, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.8)

project ("gtasa_taxi_sim")

# Add source to this project's executable.
add_executable (gtasa_taxi_sim "src/gtasa_taxi_sim.cpp")

set_property(TARGET gtasa_taxi_sim PROPERTY CXX_STANDARD 17)
set_property(TARGET gtasa_taxi_sim PROPERTY CXX_STANDARD_REQUIRED ON)

target_include_directories(gtasa_taxi_sim PRIVATE "src")

# TODO: Add tests and install targets if needed.
