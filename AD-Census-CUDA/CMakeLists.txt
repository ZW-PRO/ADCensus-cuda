cmake_minimum_required(VERSION 3.18)
project(ADCensusCUDA LANGUAGES CXX CUDA)

# 使用 C++14
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)

# 启用 CUDA 分离编译
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

# 自动找到 CUDA Toolkit
find_package(CUDAToolkit REQUIRED)

# CUDA 架构（可以按需调整）
set(CMAKE_CUDA_ARCHITECTURES 52 60 61 75 86)

# 收集所有源文件
file(GLOB SOURCES
    "${CMAKE_SOURCE_DIR}/*.cu"
    "${CMAKE_SOURCE_DIR}/*.cpp"
)

# 生成可执行文件
add_executable(ADCensusStereo ${SOURCES})

# 链接 CUDA Runtime
target_link_libraries(ADCensusStereo PRIVATE CUDA::cudart)

# 如果需要额外链接库，可以在这里添加
# find_package(OpenCV REQUIRED)
# target_link_libraries(ADCensusStereo PRIVATE ${OpenCV_LIBS})

# CUDA 编译选项
target_compile_options(ADCensusStereo PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler;-fPIC>
)

# 指定输出目录
set_target_properties(ADCensusStereo PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)
