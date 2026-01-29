cmake_minimum_required(VERSION 3.16)
project(pto-wsp VERSION 0.1.0 LANGUAGES CXX)

# C++23 required
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Options
option(PTO_WSP_BUILD_TESTS "Build tests" ON)
option(PTO_WSP_BUILD_EXAMPLES "Build examples" ON)
option(PTO_WSP_BUILD_DOCS "Build documentation" OFF)
option(PTO_WSP_BUILD_PYTHON "Build Python bindings" ON)

# pto-isa dependency (default: git submodule in 3rdparty/)
if(NOT DEFINED PTO_ISA_PATH)
    set(PTO_ISA_PATH "${CMAKE_SOURCE_DIR}/3rdparty/pto-isa" CACHE PATH "Path to pto-isa")
endif()

if(EXISTS "${PTO_ISA_PATH}/include/pto/pto-inst.hpp")
    message(STATUS "Found pto-isa at: ${PTO_ISA_PATH}")
else()
    message(WARNING "pto-isa not found at: ${PTO_ISA_PATH}")
endif()

# Include directories
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${PTO_ISA_PATH}/include
)

# Collect source files
file(GLOB_RECURSE PTO_WSP_IR_SOURCES
    "src/pto/rt/ir/*.cpp"
)

file(GLOB_RECURSE PTO_WSP_GRAPH_SOURCES
    "src/pto/rt/graph/*.cpp"
)

file(GLOB_RECURSE PTO_WSP_BACKEND_SOURCES
    "src/pto/rt/backend/*.cpp"
)

# Codegen / artifact building utilities (v9 formal pipeline)
file(GLOB_RECURSE PTO_WSP_CODEGEN_SOURCES
    "src/pto/rt/codegen/*.cpp"
)

# Combine all sources
set(PTO_WSP_SOURCES
    ${PTO_WSP_IR_SOURCES}
    ${PTO_WSP_GRAPH_SOURCES}
    ${PTO_WSP_BACKEND_SOURCES}
    ${PTO_WSP_CODEGEN_SOURCES}
)

# Create library - STATIC if sources exist, INTERFACE otherwise
if(PTO_WSP_SOURCES)
    add_library(pto-wsp-ir STATIC ${PTO_WSP_SOURCES})
    target_include_directories(pto-wsp-ir PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    )
    target_compile_features(pto-wsp-ir PUBLIC cxx_std_23)
    target_compile_definitions(pto-wsp-ir PUBLIC
        PTO_WSP_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
        PTO_ISA_PATH="${PTO_ISA_PATH}"
    )
    # Enable PIC for use with Python bindings
    set_property(TARGET pto-wsp-ir PROPERTY POSITION_INDEPENDENT_CODE ON)
    message(STATUS "Building pto-wsp-ir as STATIC library with sources: ${PTO_WSP_SOURCES}")
else()
    # Fall back to header-only if no sources yet
    add_library(pto-wsp-ir INTERFACE)
    target_include_directories(pto-wsp-ir INTERFACE
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    )
    message(STATUS "Building pto-wsp-ir as INTERFACE (header-only) library")
endif()

# Backend libraries (to be implemented)
# add_library(pto-wsp-cpu-sim src/backend/cpu_sim.cpp)
# add_library(pto-wsp-npu src/backend/npu.cpp)
# add_library(pto-wsp-aie src/backend/aie.cpp)

# Tests
if(PTO_WSP_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# Examples
if(PTO_WSP_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# Python bindings
if(PTO_WSP_BUILD_PYTHON)
    find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
    find_package(pybind11 CONFIG QUIET)

    if(NOT pybind11_FOUND)
        # Try to find pybind11 via Python
        execute_process(
            COMMAND ${Python3_EXECUTABLE} -c "import pybind11; print(pybind11.get_cmake_dir())"
            OUTPUT_VARIABLE pybind11_DIR
            OUTPUT_STRIP_TRAILING_WHITESPACE
            RESULT_VARIABLE pybind11_RESULT
        )
        if(pybind11_RESULT EQUAL 0)
            find_package(pybind11 CONFIG REQUIRED PATHS ${pybind11_DIR})
        endif()
    endif()

    if(pybind11_FOUND)
        message(STATUS "Found pybind11: ${pybind11_VERSION}")

        pybind11_add_module(pto_ir_cpp
            src/python/pto_ir_bindings.cpp
            src/python/pto_codegen.cpp
        )
        target_link_libraries(pto_ir_cpp PRIVATE pto-wsp-ir)
        target_include_directories(pto_ir_cpp PRIVATE
            ${CMAKE_SOURCE_DIR}/include
            ${PTO_ISA_PATH}/include
        )
        target_compile_definitions(pto_ir_cpp PRIVATE
            PTO_WSP_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
            PTO_ISA_PATH="${PTO_ISA_PATH}"
        )

        # Copy built module to python/pto_wsp/ for development
        add_custom_command(TARGET pto_ir_cpp POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy
                $<TARGET_FILE:pto_ir_cpp>
                ${CMAKE_SOURCE_DIR}/python/pto_wsp/$<TARGET_FILE_NAME:pto_ir_cpp>
            COMMENT "Copying pto_ir_cpp to python/pto_wsp/ for development"
        )

        # Install Python module to the Python package directory
        install(TARGETS pto_ir_cpp
            LIBRARY DESTINATION ${Python3_SITELIB}/pto_wsp
            RUNTIME DESTINATION ${Python3_SITELIB}/pto_wsp
        )
    else()
        message(WARNING "pybind11 not found, Python bindings will not be built")
    endif()
endif()

# Install
install(DIRECTORY include/ DESTINATION include)

# Install library
if(PTO_WSP_SOURCES)
    install(TARGETS pto-wsp-ir
        EXPORT pto-wsp-targets
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
    )
endif()
