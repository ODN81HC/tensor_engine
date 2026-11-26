cmake_minimum_required(VERSION 3.22)
project(tensor_engine LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON) # Create build/compile_commands.json for VSCode so that F12 works

# Cuda find package
if(NOT DEFINED CMAKE_CUDA_COMPILER)
    find_program(CMAKE_CUDA_COMPILER nvcc HINTS /usr/local/cuda/bin /usr/local/cuda-13.0/bin)
endif()
enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(Eigen3 3.4 REQUIRED)
find_package(OpenCV REQUIRED)

# Dual camera with .engine
find_library(NVINFER_LIB nvinfer REQUIRED)

add_library(tensor_engine_core STATIC
    src/matrix.cpp
    src/thread_pool.cpp
    src/camera_stream_pipeline.cpp
    src/trt_engine.cpp
    src/cuda/activations.cu
    src/cuda/gemm.cu
    src/cuda/cuda_stream.cu
    src/cuda/image_processing.cu
    src/cuda/vision_kernels.cu
)

target_include_directories(tensor_engine_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/include/cuda
)
target_include_directories(tensor_engine_core SYSTEM PUBLIC
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

target_link_libraries(tensor_engine_core PUBLIC
    Eigen3::Eigen
    ${OpenCV_LIBS}
    ${NVINFER_LIB}
)

include(cmake/CompilerWarnings.cmake)
myproject_set_project_warnings(tensor_engine_core FALSE "" "" "" "")
target_compile_options(tensor_engine_core PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:${PROJECT_WARNINGS_CXX}>
    $<$<COMPILE_LANGUAGE:CUDA>:${PROJECT_WARNINGS_CUDA}>
)

enable_testing()
find_package(GTest REQUIRED)
# Create executable target from test .cpp file
add_executable(test_tensor_engine_core
    test/test_gemm.cpp
    test/test_thread_pool.cpp
    test/test_cuda.cpp
    test/test_tensor.cpp
)
# Tell CMake: "This test needs symbols and headers from tensor_engine_core and GoogleTest"
target_link_libraries(test_tensor_engine_core PRIVATE
    tensor_engine_core
    GTest::gtest
    GTest::gtest_main
)
target_compile_options(test_tensor_engine_core PRIVATE ${PROJECT_WARNINGS_CXX})
include(GoogleTest)
gtest_discover_tests(test_tensor_engine_core)

# Dual camera GPU
add_executable(dual_cam_pipeline src/main_dual_cam.cpp)
target_link_libraries(dual_cam_pipeline PRIVATE tensor_engine_core ${OpenCV_LIBS})
