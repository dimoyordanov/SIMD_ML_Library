include(GoogleTest)

# ------------------------------------------------------------------
# Helper: build one test executable and register it with CTest.
# Usage: add_ml_test(<name> <source> [extra_lib ...])
# ------------------------------------------------------------------
function(add_ml_test test_name source)
    add_executable(${test_name} ${source})
    target_link_libraries(${test_name}
        PRIVATE
            ml_kernels
            GTest::gtest
            GTest::gtest_main
            ${ARGN}
    )
    gtest_discover_tests(${test_name})
endfunction()

# ISPC kernel tests (link ml_kernels only)
add_ml_test(test_activation test_activation.cpp)
add_ml_test(test_linear     test_linear.cpp)
add_ml_test(test_loss       test_loss.cpp)
add_ml_test(test_backprop   test_backprop.cpp)
