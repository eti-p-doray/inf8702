cmake_minimum_required (VERSION 3.6)
project (inf8702)

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/libs")

find_package( OpenCV REQUIRED core imgcodecs)
find_package( Boost REQUIRED system filesystem iostreams )
find_package( TBB REQUIRED )
find_package( OpenCl REQUIRED )

message(${OpenCV_LIBRARIES})
message(${OpenCV_INCLUDE_DIRS})

add_executable(inf8702
  poisson_serial.cpp
  main.cpp
  poisson_tbb.cpp
  cl/command_queue.cpp
  cl/event.cpp
  cl/kernel.cpp
  cl/memory.cpp
  cl/platform.cpp
  cl/program.cpp)
set_property(TARGET inf8702 PROPERTY CXX_STANDARD 14)

include_directories(inf8702 ${CMAKE_CURRENT_SOURCE_DIR})
include_directories(inf8702 ${Boost_INCLUDE_DIRS})
include_directories(inf8702 ${OpenCV_INCLUDE_DIRS})
include_directories(inf8702 ${TBB_INCLUDE_DIRS})
include_directories(inf8702 ${OpenCL_INCLUDE_DIRS})

target_link_libraries( inf8702 ${OpenCV_LIBRARIES} )
target_link_libraries( inf8702 ${Boost_LIBRARIES} )
target_link_libraries( inf8702 ${TBB_LIBRARIES})
target_link_libraries( inf8702 ${OpenCL_LIBRARIES})
