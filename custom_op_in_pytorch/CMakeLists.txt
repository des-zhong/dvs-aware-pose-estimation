# make custom_ops.so command(depends on libtorch):
#   cmake -DCMAKE_PREFIX_PATH=your/libtorch/path ..
#   make -j

cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
project(custom_ops)

find_package(Torch REQUIRED)

add_library(custom_ops SHARED 
    custom_op_myLif.cpp
    custom_op_Rand.cpp)
set_property(TARGET custom_ops PROPERTY CXX_STANDARD 14)

target_compile_features(custom_ops PUBLIC cxx_range_for)
target_link_libraries(custom_ops "${TORCH_LIBRARIES}")


target_compile_definitions(custom_ops PRIVATE custom_ops_EXPORTS)