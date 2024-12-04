cmake_minimum_required(VERSION 3.5.0)
project(Neural-Networks VERSION 0.1.0 LANGUAGES C CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)

# Compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Werror -pedantic -g")

# Add source files
file(GLOB SRC
    "src/*.cpp"
    "src/matrix/*.cpp"
    "src/parser/*.cpp"
    "src/data/*.cpp"
    "src/tests/*.cpp"
    "src/network/*.cpp"
)

add_executable(net ${SRC})

set(RELEASE_FLAGS "-O3 -funroll-loops -march=native")
# Uncomment the following line to add additional optimization flags if needed
# set(RELEASE_FLAGS "${RELEASE_FLAGS} -flto -fprefetch-loop-arrays -fno-rtti -ffast-math")

# Apply release flags if in release mode
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    set_target_properties(net PROPERTIES COMPILE_FLAGS "${RELEASE_FLAGS}")
endif()

# Print a message after building
add_custom_command(TARGET net POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E echo "compilation done")