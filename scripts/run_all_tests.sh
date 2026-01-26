#!/bin/bash
# PTO-RT Unified Test Runner (L5)
# Runs all C++ and Python tests with summary reporting

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="${PROJECT_ROOT}/build"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Test results
CPP_PASSED=0
CPP_FAILED=0
PYTHON_PASSED=0
PYTHON_FAILED=0
PYTHON_SKIPPED=0

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}    PTO-RT Unified Test Runner${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Function to run C++ tests
run_cpp_tests() {
    echo -e "${YELLOW}=== C++ Tests ===${NC}"

    # Check if build directory exists
    if [ ! -d "$BUILD_DIR" ]; then
        echo -e "${YELLOW}Build directory not found. Running cmake...${NC}"
        cmake -B "$BUILD_DIR" -DPTO_ISA_PATH="${PROJECT_ROOT}/3rdparty/pto-isa" "$PROJECT_ROOT"
    fi

    # Build
    echo -e "${BLUE}Building C++ tests...${NC}"
    cmake --build "$BUILD_DIR" -j$(nproc) 2>&1 | tail -5

    # Run tests
    echo ""
    echo -e "${BLUE}Running C++ tests...${NC}"

    local test_executables=("test_ir" "test_graph" "test_backend")

    for test in "${test_executables[@]}"; do
        local test_path="${BUILD_DIR}/tests/${test}"
        if [ -f "$test_path" ]; then
            echo -e "\n${YELLOW}Running ${test}...${NC}"
            if "$test_path" 2>&1; then
                ((CPP_PASSED++))
                echo -e "${GREEN}${test}: PASSED${NC}"
            else
                ((CPP_FAILED++))
                echo -e "${RED}${test}: FAILED${NC}"
            fi
        else
            echo -e "${RED}Test executable not found: ${test_path}${NC}"
            ((CPP_FAILED++))
        fi
    done
}

# Function to run Python tests
run_python_tests() {
    echo ""
    echo -e "${YELLOW}=== Python Tests ===${NC}"

    cd "$PROJECT_ROOT"

    # Check for pytest
    if ! command -v pytest &> /dev/null; then
        echo -e "${RED}pytest not found. Install with: pip install pytest${NC}"
        return 1
    fi

    echo -e "${BLUE}Running Python tests...${NC}"

    # Run pytest and capture results
    local pytest_output
    pytest_output=$(python -m pytest tests/ -v --tb=short 2>&1) || true

    echo "$pytest_output"

    # Parse pytest output for summary
    local summary_line=$(echo "$pytest_output" | grep -E "^[=]+ .* passed" | tail -1)

    if [[ "$summary_line" =~ ([0-9]+)\ passed ]]; then
        PYTHON_PASSED=${BASH_REMATCH[1]}
    fi

    if [[ "$summary_line" =~ ([0-9]+)\ failed ]]; then
        PYTHON_FAILED=${BASH_REMATCH[1]}
    fi

    if [[ "$summary_line" =~ ([0-9]+)\ skipped ]]; then
        PYTHON_SKIPPED=${BASH_REMATCH[1]}
    fi
}

# Function to print summary
print_summary() {
    echo ""
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}           Test Summary${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo ""

    local total_passed=$((CPP_PASSED + PYTHON_PASSED))
    local total_failed=$((CPP_FAILED + PYTHON_FAILED))

    echo -e "C++ Tests:"
    echo -e "  ${GREEN}Passed: ${CPP_PASSED}${NC}"
    echo -e "  ${RED}Failed: ${CPP_FAILED}${NC}"
    echo ""

    echo -e "Python Tests:"
    echo -e "  ${GREEN}Passed: ${PYTHON_PASSED}${NC}"
    echo -e "  ${RED}Failed: ${PYTHON_FAILED}${NC}"
    echo -e "  ${YELLOW}Skipped: ${PYTHON_SKIPPED}${NC}"
    echo ""

    echo -e "${BLUE}----------------------------------------${NC}"
    echo -e "Total:"
    echo -e "  ${GREEN}Passed: ${total_passed}${NC}"
    echo -e "  ${RED}Failed: ${total_failed}${NC}"
    echo ""

    if [ $total_failed -eq 0 ]; then
        echo -e "${GREEN}All tests passed!${NC}"
        return 0
    else
        echo -e "${RED}Some tests failed!${NC}"
        return 1
    fi
}

# Main execution
main() {
    local run_cpp=true
    local run_python=true

    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --cpp-only)
                run_python=false
                shift
                ;;
            --python-only)
                run_cpp=false
                shift
                ;;
            --help|-h)
                echo "Usage: $0 [OPTIONS]"
                echo ""
                echo "Options:"
                echo "  --cpp-only     Run only C++ tests"
                echo "  --python-only  Run only Python tests"
                echo "  --help, -h     Show this help message"
                exit 0
                ;;
            *)
                echo "Unknown option: $1"
                exit 1
                ;;
        esac
    done

    # Run tests
    if [ "$run_cpp" = true ]; then
        run_cpp_tests
    fi

    if [ "$run_python" = true ]; then
        run_python_tests
    fi

    # Print summary
    print_summary
}

main "$@"
