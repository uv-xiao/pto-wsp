# PTO Workload-Schedule Programming (PTO-WSP) framework Tests

# IR unit tests
add_executable(test_ir test_ir.cpp)
target_link_libraries(test_ir PRIVATE pto-wsp-ir)
target_compile_features(test_ir PRIVATE cxx_std_23)

add_test(NAME test_ir COMMAND test_ir)

# Graph infrastructure tests
add_executable(test_graph test_graph.cpp)
target_link_libraries(test_graph PRIVATE pto-wsp-ir)
target_compile_features(test_graph PRIVATE cxx_std_23)
# Need pthread for threading tests
find_package(Threads REQUIRED)
target_link_libraries(test_graph PRIVATE Threads::Threads)

add_test(NAME test_graph COMMAND test_graph)

# Backend tests
add_executable(test_backend test_backend.cpp)
target_link_libraries(test_backend PRIVATE pto-wsp-ir Threads::Threads)
target_compile_features(test_backend PRIVATE cxx_std_23)

add_test(NAME test_backend COMMAND test_backend)
