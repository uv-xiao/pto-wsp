# PTO-WSP v9 End-to-End RMSNorm Example
# Makefile for running and testing the example

PYTHON ?= python3
EXAMPLE = e2e_rmsnorm_example.py
PROJECT_ROOT = ../..

.PHONY: run test clean help

# Default target
run:
	cd $(PROJECT_ROOT) && $(PYTHON) examples/e2e_rmsnorm/$(EXAMPLE)

# Run with verification (same as run for CPU sim examples)
test: run

# Clean generated files
clean:
	rm -f *.pyc __pycache__

# Help
help:
	@echo "Available targets:"
	@echo "  run   - Execute the example"
	@echo "  test  - Run with verification"
	@echo "  clean - Clean generated files"
	@echo "  help  - Show this help message"
