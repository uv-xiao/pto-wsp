---
name: codex
description: Notes and workflow for using OpenAI Codex CLI (`codex exec`) for scripted, non-interactive analysis tasks (reviews, bug hunting, doc/spec review, planning).
---

# codex

Notes for using OpenAI Codex CLI (`codex exec`) for non-interactive analysis tasks.

## When to use `codex exec`

- Code review / bug hunting
- Architecture/design feedback
- Document/spec review
- Implementation planning

## Quick examples

```bash
codex exec "Review docs/spec.md for API consistency"
echo "List all TODOs" | codex exec -
codex exec --json "List all public API functions"
```

## Useful flags (common)

- `-C, --cd <path>`: set working directory
- `--search`: enable web search (when external context helps)
- `-o <path>`: write final response to a file
- `--json`: newline-delimited JSON events
