# Feature: CSP (Process/Channel) Dialect

## Goal

Represent pipeline-parallel programs explicitly as:

- processes (concurrent logical tasks)
- channels (typed, bounded communication)
- consumption patterns (finite streams, termination, rendezvous events)

## Stream type requirements (Dato-inspired)

Channels should have a type shape like:

- `Channel[T, N, P]`
  - `T`: element type (often a tile/tensor block)
  - `N`: logical capacity/depth
  - `P`: optional packing factor (backend-dependent)

Operations:

- `put(x)` blocks when full
- `get()` blocks when empty

## Effect typing (deadlock prevention)

At minimum, provide:

- protocol checking: put/get counts must match along every control-flow path
- boundedness checking: ensure no unbounded producer without consumer progress in static CSP graphs

Stronger (future) option:

- linear capability tokens (Free/Used slots) as a type/effect system.

### Minimal typing sketch (linear slot tokens)

Model a bounded channel `c` with identity `C` and capacity `N` as a multiset of tokens:

- `Free(C)` tokens represent available slots
- `Used(C)` tokens represent occupied slots

Typing rules (informal):

- `put(c, x)` consumes one `Free(C)` and produces one `Used(C)`
- `get(c)` consumes one `Used(C)` and produces one `Free(C)`

Static checking uses these rules to reject programs that would require negative tokens (overflow/underflow) along any
path. This is the core idea behind making deadlocks/violations “untypeable by construction” (see Dato).

HTP does not need to start with the full research-grade system; it needs:

1. a canonical CSP graph representation (explicit producers/consumers),
2. a well-defined “effect check pass” that can be strengthened over time.

## Backend semantics

Different backends implement CSP differently:

- simulation backend: discrete-event simulation driven by kernel cycle models
- device backend: runtime scheduler/streams + synchronization primitives
- AIE backend: FIFOs + tile streams

The CSP dialect must provide a canonical graph form that backends can lower from.

## References

- Stream typing + linear slot reasoning: `docs/reference/16_dato.md`
- AIE FIFO/dataflow shape: `docs/reference/15_allo.md`
