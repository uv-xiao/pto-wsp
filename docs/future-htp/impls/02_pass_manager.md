# Impl: Pass Manager

## Goals

- deterministic ordering
- explicit contracts (`requires/provides`)
- standard tracing and dump hooks

## Recommended pass manager API

- register passes by name and version
- build pipeline from pass list + parameters
- run pipeline with:
  - `before/after` dumps
  - per-pass timing
  - structured diagnostics emission

Each pass registration includes:

- `requires` capability set
- `provides` capability set
- optional: `invalidates` set (capabilities no longer guaranteed after the pass)

## Trace output

- `ir/pass_trace.jsonl` events:
  - pass name/version
  - requires/provides
  - start/end timestamps
  - dump file pointers
  - failure diagnostics
