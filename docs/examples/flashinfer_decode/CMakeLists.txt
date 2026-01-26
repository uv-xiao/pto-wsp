cmake_minimum_required(VERSION 3.16)
project(flashinfer_decode_example CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Include paths
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Option for CPU simulation
option(CPU_SIM "Build for CPU simulation" ON)

if(CPU_SIM)
    add_compile_definitions(__CPU_SIM)
endif()

# =============================================================================
# Runtime Library (Header-only for now)
# =============================================================================

add_library(pto_runtime INTERFACE)
target_include_directories(pto_runtime INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# =============================================================================
# Host Planning Example (CPU-only, standalone)
# =============================================================================

add_executable(host_planning_example
    src/host_planning.cpp
)
target_link_libraries(host_planning_example PRIVATE pto_runtime)
target_compile_definitions(host_planning_example PRIVATE HOST_PLANNING_STANDALONE)

# =============================================================================
# Tests
# =============================================================================

enable_testing()

add_executable(flashinfer_decode_test
    tests/test_decode.cpp
)
target_link_libraries(flashinfer_decode_test PRIVATE pto_runtime)

add_test(NAME flashinfer_decode_test
    COMMAND flashinfer_decode_test
)

# =============================================================================
# Device Kernel Example (requires PTO-ISA and CANN)
# =============================================================================

# Note: The device kernel (decode_attention.hpp) requires the full PTO-ISA
# library. To build with device support:
#
# cmake -DCPU_SIM=OFF -DPTO_ISA_ROOT=/path/to/pto-isa ..
#
# if(NOT CPU_SIM AND PTO_ISA_ROOT)
#     include_directories(${PTO_ISA_ROOT}/include)
#     add_executable(decode_attention_device
#         src/device_kernel.cpp
#     )
#     target_link_libraries(decode_attention_device PRIVATE pto_runtime)
# endif()

# =============================================================================
# Installation (optional)
# =============================================================================

install(TARGETS host_planning_example flashinfer_decode_test
    RUNTIME DESTINATION bin
)

install(DIRECTORY include/
    DESTINATION include/pto/runtime
)
