# Makefile for CSP Pipeline Example
#
# Usage:
#   make run   - Execute the example
#   make test  - Run with verification
#   make clean - Clean generated files
#   make help  - Show available targets

.PHONY: run test clean help

# Default target
run:
	@echo "Running CSP Pipeline Example..."
	@cd ../.. && python examples/csp_pipeline/csp_pipeline_example.py

# Run with verification
test: run
	@echo "CSP Pipeline example completed successfully."

# Clean generated files
clean:
	@rm -f *.pyc
	@rm -rf __pycache__

# Help
help:
	@echo "CSP Pipeline Example - PTO-RT v9"
	@echo ""
	@echo "Targets:"
	@echo "  make run   - Execute the example"
	@echo "  make test  - Run with verification"
	@echo "  make clean - Clean generated files"
	@echo "  make help  - Show this help message"
	@echo ""
	@echo "Features demonstrated:"
	@echo "  - Channel[T] for typed communication"
	@echo "  - Process for pipeline stages"
	@echo "  - send/consume for channel operations"
	@echo "  - connect for pipeline composition"
	@echo "  - Event synchronization primitives"
