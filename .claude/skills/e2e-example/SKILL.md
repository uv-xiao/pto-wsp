---
name: e2e-example
description: Manage and validate PTO-RT v9 examples with comprehensive E2E testing. Subcommands: list (show examples), run (validate example), add (create new example).
allowed-tools:
  - Bash
  - Read
  - Write
  - Glob
  - Grep
---

# PTO-RT E2E Example Management

This skill provides comprehensive example management and validation for PTO-RT v9 examples.

## Subcommands

Parse the user's command to determine which subcommand to execute:

- `/e2e-example list` → Execute "List Examples"
- `/e2e-example run <name>` → Execute "Run and Validate Example"
- `/e2e-example add <name>` → Execute "Add New Example"

---

## List Examples

**Trigger**: `/e2e-example list` or `/e2e-example` with no arguments

**Actions**:

1. Find all example directories:
   ```bash
   ls -d examples/*/
   ```

2. For each example, report:
   - Example name
   - Main script file
   - Has README.md (yes/no)
   - Has Makefile (yes/no)
   - Quick validation status (runs without error)

3. Output format:
   ```
   PTO-RT v9 Examples
   ==================

   | Example | Script | README | Makefile | Status |
   |---------|--------|--------|----------|--------|
   | attention | attention_example.py | ✓ | ✓ | ? |
   | bgemm | bgemm_example.py | ✓ | ✓ | ? |
   ...

   Use `/e2e-example run <name>` for detailed validation.
   ```

---

## Run and Validate Example

**Trigger**: `/e2e-example run <name>`

**Actions**:

1. **Locate example**:
   ```bash
   ls examples/<name>/*.py
   ```

2. **Read the example source** to analyze:
   - Use Read tool on the main .py file

3. **Perform validation checks** (report each):

### Check 1: Script Execution
```bash
python examples/<name>/<script>.py 2>&1
```
- PASS: Exit code 0
- FAIL: Non-zero exit code or Python exception

### Check 2: Import Errors
- Grep output for "ImportError", "ModuleNotFoundError"
- PASS: No import errors
- FAIL: Import errors found

### Check 3: Deprecation Warnings
- Grep output for "DeprecationWarning"
- Check specifically for `npu()` deprecation (ERROR - should be fixed)
- Note `batch_deps()`, `pipeline_depth()` deprecation (WARNING - documented)

### Check 4: JIT Kernel Connection (CRITICAL)
Analyze source code for pattern:
```python
# Look for @jit_kernel functions
@jit_kernel
def xxx_jit(...):
    ...

# Look for @kernel stubs
@kernel
def xxx_kernel(...):
    pass  # <-- ISSUE: Empty body
```

Report:
- List all `@jit_kernel` functions defined
- List all `@kernel` stubs used in workload
- Check if JIT kernel is actually used or just defined
- WARNING if JIT kernel defined but workload uses separate stub

### Check 5: Empty Stub Detection (CRITICAL)
Look for patterns:
```python
def cpu_xxx(...):
    pass  # Empty CPU stub

@kernel
def xxx(...):
    pass  # Empty kernel stub
```

Report:
- WARNING: "CPU stub 'cpu_xxx' has empty body (pass)"
- WARNING: "Kernel stub 'xxx_kernel' has empty body (pass)"

### Check 6: Task Count Verification
- Extract expected task count from output (e.g., "Total tasks: 4096")
- Verify it matches configuration (batch × tiles × etc.)

### Check 7: Execution Completion
- Check for "Execution complete!" or similar message
- Check for "Example completed successfully!"

4. **Generate validation report**:
   ```
   E2E Validation Report: <name>
   ==============================

   Script: examples/<name>/<script>.py

   Checks:
   [PASS] Script execution (exit code 0)
   [PASS] No import errors
   [WARN] Deprecation: batch_deps() (documented, acceptable)
   [WARN] JIT kernel 'gemm_tile_jit' defined but 'gemm_tile_kernel' stub used
   [WARN] CPU stub 'cpu_gemm_tile' has empty body
   [PASS] Task count: 4096 (4 × 16 × 16 × 4)
   [PASS] Execution completed

   Issues Found:
   1. [WARNING] JIT kernel not connected to workload
      - 'gemm_tile_jit' is defined with tl.* primitives
      - Workload uses 'gemm_tile_kernel' stub instead
      - CPU simulation runs empty stub (no computation)

   2. [WARNING] No output validation
      - Example doesn't verify computed values
      - Passes if script runs, not if results are correct

   Recommendations:
   - Connect JIT kernel to workload OR document as API demo
   - Add NumPy reference implementation for CPU validation
   - Add output comparison with tolerance check
   ```

---

## Add New Example

**Trigger**: `/e2e-example add <name>`

**Actions**:

1. **Create directory structure**:
   ```bash
   mkdir -p examples/<name>
   ```

2. **Create template files**:

### Template: `<name>_example.py`
```python
#!/usr/bin/env python3
"""
PTO-RT v9 Example: <Name>

<Description of what this example demonstrates>

Usage:
    python examples/<name>/<name>_example.py
"""

import sys
sys.path.insert(0, 'python')

from pto_wsp import (
    jit_kernel, tl, kernel,
    In, Out, InOut, Tile, Scalar,
    workload, P,
    Dense, Tensor, DType,
    DispatchPolicy, TimingPolicy,
)

# DType shortcuts
F16 = DType.F16
F32 = DType.F32

# ============================================================
# Configuration
# ============================================================

# TODO: Define configuration constants

# ============================================================
# JIT Kernel Definition (v9 recommended style)
# ============================================================

@jit_kernel
def <name>_kernel_jit(
    # TODO: Define typed parameters
):
    """<Description>

    Uses typed Value objects with tl.* primitives.
    """
    # TODO: Implement kernel using tl.* primitives
    pass

# ============================================================
# Workload Definition
# ============================================================

@workload
def <name>_workload():
    """<Description>"""
    # TODO: Define workload with P namespace
    pass

# ============================================================
# Main Entry Point
# ============================================================

def main():
    print("=" * 60)
    print("PTO-RT v9 <Name> Example")
    print("=" * 60)
    print()

    # TODO: Implement main

    print("Example completed successfully!")
    print("=" * 60)

if __name__ == "__main__":
    main()
```

### Template: `Makefile`
```makefile
# PTO-RT v9 <Name> Example

PYTHON ?= python3
EXAMPLE = <name>_example.py
PROJECT_ROOT = ../..

.PHONY: run test clean help

run:
	cd $(PROJECT_ROOT) && $(PYTHON) examples/<name>/$(EXAMPLE)

test: run

clean:
	rm -f *.pyc __pycache__

help:
	@echo "Available targets:"
	@echo "  run   - Execute the example"
	@echo "  test  - Run with verification"
	@echo "  clean - Clean generated files"
```

### Template: `README.md`
```markdown
# <Name> Example

## Overview

<Brief description>

## v9 Features Demonstrated

- `@jit_kernel` with `tl.*` primitives
- `@workload` with `P` namespace
- <other features>

## How to Run

\`\`\`bash
python examples/<name>/<name>_example.py
# or
cd examples/<name> && make run
\`\`\`

## Expected Behavior

### Successful Execution

- TODO: Document expected output

### Expected Output Sample

\`\`\`
TODO: Paste expected output
\`\`\`

## Checking Rules

### Pass Criteria

- [ ] Exit code is 0
- [ ] No Python exceptions
- [ ] "Example completed successfully!" printed
- [ ] No npu() deprecation warnings

## Troubleshooting

**ModuleNotFoundError**
- Run from project root directory
```

3. **Report created files**:
   ```
   Created example: <name>

   Files created:
   - examples/<name>/<name>_example.py
   - examples/<name>/Makefile
   - examples/<name>/README.md

   Next steps:
   1. Edit <name>_example.py to implement the example
   2. Run: python examples/<name>/<name>_example.py
   3. Validate: /e2e-example run <name>
   4. Update README.md with expected output
   ```

---

## Validation Issue Categories

When reporting issues, use these severity levels:

### ERROR (must fix)
- Script fails to execute
- Import errors
- `npu()` deprecation warning (should use @jit_kernel)

### WARNING (should address)
- JIT kernel defined but not connected to workload
- Empty kernel/CPU stubs (pass body)
- No output validation

### INFO (acceptable)
- `batch_deps()`, `pipeline_depth()` deprecation (documented R5 features)
- Demo-only features clearly documented

---

## Example Usage

```
User: /e2e-example list
Agent: [Lists all examples with status]

User: /e2e-example run bgemm
Agent: [Runs comprehensive validation on bgemm example]

User: /e2e-example add conv2d
Agent: [Creates new conv2d example from template]
```
