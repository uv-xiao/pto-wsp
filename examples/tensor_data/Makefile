# Makefile for Tensor Data Example
#
# Usage:
#   make run   - Execute the example
#   make test  - Run with verification
#   make clean - Clean generated files
#   make help  - Show available targets

.PHONY: run test clean help

# Default target
run:
	@echo "Running Tensor Data Example..."
	@cd ../.. && python examples/tensor_data/tensor_data_example.py

# Run with verification
test: run
	@echo "Tensor data example completed successfully."

# Clean generated files
clean:
	@rm -f *.pyc
	@rm -rf __pycache__

# Help
help:
	@echo "Tensor Data Example - PTO-WSP v9"
	@echo ""
	@echo "Targets:"
	@echo "  make run   - Execute the example"
	@echo "  make test  - Run with verification"
	@echo "  make clean - Clean generated files"
	@echo "  make help  - Show this help message"
	@echo ""
	@echo "Features demonstrated:"
	@echo "  - NumPy array binding to Tensor objects"
	@echo "  - Workload validation against reference"
	@echo "  - Task enumeration and index validation"
	@echo "  - Schedule configuration and execution"
