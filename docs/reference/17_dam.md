# DAM (Dataflow Abstract Machine) Notes for PTO-RT v9 (CSPT)

> Sources: `references/dam.pdf` and `references/DAM-RS/` (DAM-RS)
>
> Goal: Extract the DAM/CSPT concepts we need to implement **true CSP/CSPT** execution in PTO‑RT’s **codegen-first CPU-sim** artifacts, with **time measured strictly in PTO‑ISA cycles** and **constant per-channel latency** (default 0).

---

## 1) DAM’s user-facing model (CSP + local time = CSPT)

The DAM paper augments CSP with **local simulated time** (“CSPT”):

- Each *context* owns a local **monotonic simulated time**.
- A context can advance forward in time (but never backward).
- Contexts can **view other contexts’ times** (lower bound of their progress) and can **wait until** another context reaches a target time.
- Contexts communicate through **channels** which:
  - are statically connected (sender ↔ receiver),
  - may be bounded (capacity),
  - simulate **backpressure** and **starvation** automatically.
- DAM emphasizes that channels do not “own global time”; they **bridge between sender and receiver time zones** to decouple execution and enable parallelism.

Why this matters for PTO‑RT v9:

- We want CSP semantics to be a first-class execution model (not a Python wrapper).
- We need CSPT semantics so scheduling and pipeline behavior can be modeled in CPU-sim with cycle-accurate timing (PTO‑ISA cycles).

---

## 2) Event-queue-free runtime via asynchronous distributed time

The DAM paper contrasts CSPT with event-driven (ED) simulators built around ordered event queues:

- ED requires explicit event alignment code, can be verbose, and (as described) can make it hard to model backpressure if channels are effectively unbounded.
- DAM’s CSPT design enables an **event-queue-free runtime** by relying on **local time** and synchronization only when needed.

Key term: **asynchronous distributed time**

- There is no single global “scheduler time” required by the runtime.
- Contexts may execute “far into the future” relative to each other until forced to synchronize by channel operations (backpressure/starvation).

For PTO‑RT v9, this suggests:

- Implement CSP processes as independently progressing state machines with per-process `time_cycles`.
- Synchronize only at channel operations and explicit waits (not via a global barrier every few cycles).

---

## 3) DAM synchronization mechanisms: SVA and SVP

DAM describes two “local-only” synchronization mechanisms that trigger only as needed:

### 3.1 SVA — Synchronization via Atomics

- Uses acquire/release semantics to “optimistically” synchronize when viewing another context.
- Idea: if you can observe enough state (e.g., queue front / absence), and the observed context time is ahead, you may not need heavier sync.

### 3.2 SVP — Synchronization via Parking

- When a viewing context is “ahead” of the viewed context (needs to wait), use OS primitives (futex park/unpark) rather than spin-waiting.
- DAM runtime attempts SVA first, falls back to SVP when needed.

For PTO‑RT v9 CPU-sim artifacts:

- We do not need to copy DAM’s exact OS-level futex strategy to get *correct CSPT semantics*.
- We do need the **semantic equivalent**: a blocked process must stop progressing until the channel/time condition is satisfied; runtime may “jump” simulated time to the next unblocking event.

---

## 4) Time-bridging channels and latency

The DAM paper describes channels as “time-bridging data structures”:

- Data is timestamped with simulated time.
- Sender and receiver each keep a local view of the channel state.
- A bounded channel requires the sender to know when capacity frees up.

Conceptual structure described (paper Fig. 2):

- A “simulated channel” can be backed by:
  - a real data channel (sender → receiver) carrying timestamped data, and
  - a real “response/time” channel (receiver → sender) carrying timestamps of dequeues so the sender can update its view and model capacity.

Latency:

- The paper describes that when the receiver dequeues, it can respond with a *future time* at which the sender should “see” the size change; this models a read-to-visible latency.

PTO‑RT v9 mapping:

- We need **constant per-channel latency** (default 0) primarily for synchronization.
- We can model this with `(token, deliver_time_cycles)` in the channel queue.
- On `send`, the sender enqueues with `deliver_time = sender_time + latency_cycles`.
- On `recv/consume`, if the head token’s `deliver_time > receiver_time`, the receiver must wait / advance time.

---

## 5) Local time acceleration

The DAM paper highlights “local time acceleration”:

- Blocking operations are opportunities to fast-forward:
  - if a blocking dequeue sees the next token timestamped at `T_data > T_ctx`, the context can advance to `T_data`.
  - similarly, a sender can jump forward to the next time it expects capacity in a bounded channel.

PTO‑RT v9 mapping (CPU-sim):

- When a process blocks on a channel, it should advance its local time to the earliest time it can make progress (or yield so the global runtime can advance).
- This is compatible with using PTO‑ISA kernel cycle reports as the only “compute-time” contributions:
  - kernel call: `time_cycles += kernel_cycles`
  - channel wait: `time_cycles = next_deliver_time_cycles` (or similar)

---

## 6) DAM-RS implementation cues (what to look at)

From `references/DAM-RS/README.md`:

- DAM models computation as **contexts** connected by **channels** (Sender/Receiver pairs).
- For timing, users “attach” sender/receiver ends to a context during initialization.
- A context’s `run` method encapsulates execution.

What PTO‑RT should learn from DAM-RS:

- How channels encode time and blocking semantics without a centralized event queue.
- How context-local time is advanced in response to operations.

---

## 7) Concrete PTO‑RT v9 design guidance (CPU-sim artifacts)

### 7.1 Core runtime objects

- `ProcessState`:
  - `time_cycles` (u64, monotonic)
  - `pc/state` for codegen-generated process step function
  - runnable/blocked status and block reason

- `ChannelState`:
  - capacity
  - latency_cycles (constant; default 0)
  - queue of `(token, deliver_time_cycles)`

### 7.2 Execution loop (non-deterministic interleavings allowed)

At a high level:

1. If there is any runnable process, pick one (policy can be schedule/dispatch-driven).
2. Let it run one “step”:
   - kernel step: call kernel → `time_cycles += kernel_cycles`
   - send step: enqueue `(token, time_cycles + latency_cycles)`
   - recv/consume step:
     - if token available and deliver_time ≤ time_cycles: dequeue
     - else: block and expose “next unblock time”
3. If all processes are blocked, advance to the minimum next unblock time and retry.

This matches the DAM paper’s notion of contexts progressing asynchronously and synchronizing only when needed.

### 7.3 How PTO‑ISA cycles plug in

- PTO‑ISA kernel return values are the **only** compute-time source.
- CSPT “time” is therefore:
  - kernel cycles + modeled channel delays + wait jumps

---

## 8) Scope constraints for v9

- Channel latency is **constant**, default **0**.
- Channels are mainly synchronization (not bulk data transfer), so token payload can be a small integer id/value.
- We do not require deterministic scheduling; tests must validate invariants only.
- `task_window` overflow behavior is **stall only** (no abort/benchmark in v9).

