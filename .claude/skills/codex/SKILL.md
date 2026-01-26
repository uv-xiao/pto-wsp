---
name: codex
description: Guide for using OpenAI Codex CLI (codex exec) for non-interactive code analysis, code review, architecture review, design evaluation, and implementation planning. Use this skill when users need help with complex analysis tasks.
allowed-tools:
  - Bash
  - Read
---

# Codex CLI Usage Guide

This skill helps you use [OpenAI Codex CLI](https://developers.openai.com/codex/cli/) for non-interactive code analysis tasks using `codex exec`.

## When to Suggest Codex

Recommend `codex exec` when users need:

- **Code Review**: Quality assessment, bug hunting, best practices verification
- **Architecture Analysis**: Design pattern evaluation, coupling analysis, modularity assessment
- **Issue Identification**: Security vulnerabilities, performance bottlenecks, code smells
- **Document Analysis**: Reviewing specifications, requirements, design documents
- **Design Feedback**: API design evaluation, interface decisions, type safety
- **Implementation Planning**: Breaking down features, planning multi-file changes

## codex exec Command

Use `codex exec` (alias: `codex e`) for scripted, non-interactive runs:

```bash
# Basic execution
codex exec "Analyze the codebase architecture"

# Read prompt from stdin
echo "Review this code for security issues" | codex exec -

# With image input (diagrams, screenshots)
codex exec -i architecture.png "Explain this design"

# Enable web search for external context
codex exec --search "Compare this design with industry best practices"

# Full-auto mode (workspace-write sandbox + on-request approvals)
codex exec --full-auto "Fix all linting errors"

# JSON output for parsing in scripts
codex exec --json "List all TODO comments in the codebase"

# Save final message to file
codex exec -o result.md "Summarize the authentication flow"

# Set working directory
codex exec -C /path/to/project "Analyze this project"

# Skip git repo check for one-off directories
codex exec --skip-git-repo-check "Analyze these files"
```

## Key CLI Options

| Flag | Type | Description |
| ---- | ---- | ----------- |
| `PROMPT` | string or `-` | Task instruction (use `-` to read from stdin) |
| `-i, --image` | path[,path...] | Attach image files to the prompt |
| `-m, --model` | string | Override model (e.g., `gpt-5-codex`) |
| `-s, --sandbox` | enum | `read-only`, `workspace-write`, `danger-full-access` |
| `-a, --ask-for-approval` | enum | `untrusted`, `on-failure`, `on-request`, `never` |
| `--full-auto` | boolean | Low-friction mode (workspace-write + on-request) |
| `--search` | boolean | Enable web search tool |
| `-C, --cd` | path | Set working directory |
| `--json` | boolean | Output newline-delimited JSON events |
| `-o, --output-last-message` | path | Write final message to file |
| `--output-schema` | path | JSON Schema for validating response |
| `--skip-git-repo-check` | boolean | Allow running outside a Git repository |
| `-c, --config` | key=value | Override config values (repeatable) |
| `--oss` | boolean | Use local open source model provider (Ollama) |
| `-p, --profile` | string | Load configuration profile |

## Session Management

Resume previous sessions:

```bash
# Resume most recent session with follow-up
codex exec resume --last "Continue the analysis"

# Resume specific session by ID
codex exec resume SESSION_ID "Add more details"
```

## Sandbox and Approval Modes

### Sandbox Policies (`-s, --sandbox`)

| Mode | Description |
| ---- | ----------- |
| `read-only` | Browse files only, no modifications |
| `workspace-write` | Write access within working directory |
| `danger-full-access` | Full system access (use sparingly) |

### Approval Modes (`-a, --ask-for-approval`)

| Mode | Description |
| ---- | ----------- |
| `untrusted` | Prompt for every action |
| `on-failure` | Prompt only when commands fail |
| `on-request` | Prompt when Codex requests it |
| `never` | Never prompt (dangerous) |

### Automation Flags

```bash
# Full-auto: workspace-write + on-request (recommended for local work)
codex exec --full-auto "task"

# YOLO mode: bypass all approvals and sandboxing (DANGEROUS - only in isolated environments)
codex exec --dangerously-bypass-approvals-and-sandbox "task"
```

## Configuration

Config file: `~/.codex/config.toml`

```toml
model = "gpt-5-codex"
web_search_request = true
```

Use profiles for different configurations:

```bash
codex exec -p my-profile "task"
```

## Project-Specific Examples for PTO-ISA

### Runtime Extension Analysis

```bash
# Design review
codex exec "Review docs/uv/spec.md for API consistency and completeness"

# Requirements alignment
codex exec "Compare docs/uv/spec.md against docs/uv/requirements.md"

# Design questions review
codex exec "Analyze docs/uv/design-questions.md for any unaddressed concerns"

# Output to file
codex exec -o design-review.md "Comprehensive review of docs/uv/"
```

### PTO Implementation Analysis

```bash
# Dispatch mechanism
codex exec "Analyze the MAP_INSTR_IMPL dispatch pattern in include/pto/common/"

# CPU backend review
codex exec "Review include/pto/cpu/ for correctness and test coverage"

# NPU implementations comparison
codex exec "Compare include/pto/npu/a2a3/ and include/pto/npu/a5/ for consistency"

# Type system analysis
codex exec "Analyze the Tile type system in include/pto/common/pto_tile.hpp"
```

### Security and Quality

```bash
# Security review
codex exec "Security review of the codebase, focusing on input validation"

# Performance analysis
codex exec "Identify performance bottlenecks in the tile operations"

# Code quality
codex exec --full-auto "Fix all code style issues in include/pto/"
```

### CI/CD Integration

```bash
# JSON output for parsing
codex exec --json "List all public API functions" > api-list.json

# With output schema validation
codex exec --output-schema schema.json "Generate API documentation"

# Pipe results
codex exec "Summarize changes" | tee summary.md
```

## Tips

1. **Use `--full-auto`** for trusted local work that needs file modifications
2. **Use `--search`** when external context (papers, docs, standards) is needed
3. **Use `-o`** to save analysis results to files for later reference
4. **Use `--json`** for CI/CD integration and scripted parsing
5. **Use `codex exec resume --last`** to continue complex multi-step analyses
6. **Use `-i`** to attach architecture diagrams or error screenshots
7. **Prefer `read-only` sandbox** for pure analysis tasks
