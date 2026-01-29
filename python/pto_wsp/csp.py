"""
CSP primitives for PTO Workload-Schedule Programming (PTO-WSP) framework.

Pipeline-parallel programming with channels and processes.

STATUS (v9):
- Python APIs build a CSP workload graph (channels/processes/send/consume/connect).
- CPU-sim **codegen-first** artifacts execute CSP pipelines with CSPT time semantics:
  - timebase is PTO-ISA kernel cycle reports
  - constant channel latency (default 0; runtime symbol override)

LIMITATIONS (v9 scope):
- CSP execution is supported in CPU-sim codegen-first artifacts; NPU execution requires device toolchain/runtime.
- Schedule directives beyond `dispatch` + `task_window` are not fully enforced in v9 artifacts.
"""

from __future__ import annotations
from typing import Callable, Any, Generic, TypeVar
from pto_wsp.workload import Workload

T = TypeVar("T")


class Channel(Generic[T]):
    """Typed, bounded communication channel.

    Args:
        name: Channel name for debugging
        depth: Buffer capacity (0 for rendezvous/Event)

    Example:
        load_to_compute = Channel("l2c", depth=2)
    """

    def __init__(self, name: str, depth: int = 1):
        self.name = name
        self.depth = depth
        self._open = True
        self._buffer: list[T] = []

    def is_open(self) -> bool:
        """Check if channel is open."""
        return self._open

    def size(self) -> int:
        """Current elements in buffer."""
        return len(self._buffer)

    def empty(self) -> bool:
        """Check if buffer is empty."""
        return len(self._buffer) == 0

    def full(self) -> bool:
        """Check if buffer is full."""
        return len(self._buffer) >= self.depth

    def close(self) -> None:
        """Signal no more values."""
        self._open = False


# Event: unbuffered channel for synchronization
Event = Channel  # with depth=0


def process(name: str) -> ProcessBuilder:
    """Create a process builder.

    Args:
        name: Process name

    Returns:
        ProcessBuilder for fluent configuration

    Example:
        loader = (process("loader")
            .produces(channel)
            .body(for_each(axis, lambda i: ...)))
    """
    return ProcessBuilder(name)


class ProcessBuilder:
    """Fluent API for process construction."""

    def __init__(self, name: str):
        self._name = name
        self._consumes: list[Channel] = []
        self._produces: list[Channel] = []
        self._body = None

    def consumes(self, *channels: Channel) -> ProcessBuilder:
        """Declare input channels."""
        self._consumes.extend(channels)
        return self

    def produces(self, *channels: Channel) -> ProcessBuilder:
        """Declare output channels."""
        self._produces.extend(channels)
        return self

    def body(self, computation: Any) -> Process:
        """Set process body (must be declarative)."""
        return Process(
            self._name,
            self._consumes,
            self._produces,
            computation
        )


class Process:
    """Concurrent process with declarative body."""

    def __init__(self, name: str, consumes: list[Channel],
                 produces: list[Channel], body: Any):
        self.name = name
        self.consumes = consumes
        self.produces = produces
        self.body = body


def send(channel: Channel[T], value: T) -> Workload:
    """Put value into channel (blocks if full).

    Args:
        channel: Target channel
        value: Value to send (typically a Task workload)

    Returns:
        Workload representing the send operation
    """
    return Workload("send", channel=channel, value=value)


def try_send(channel: Channel[T], value: T) -> bool:
    """Non-blocking send attempt.

    Args:
        channel: Target channel
        value: Value to send

    Returns:
        True if send succeeded, False if channel was full
    """
    if not channel.full():
        channel._buffer.append(value)
        return True
    return False


def consume(channel: Channel[T], body: Callable[[T], Any]) -> Workload:
    """Declarative channel iteration.

    Processes all values until channel is closed.
    This replaces imperative `while (value := recv(channel))` which is not JIT-friendly.

    Args:
        channel: Source channel
        body: Lambda function processing each value

    Returns:
        Workload representing the consume loop

    Example:
        consume(input_ch, lambda t:
            send(output_ch, task("process", [t], resources)))
    """
    return Workload("consume", channel=channel, body=body)


def connect(processes: list[Process], channels: list[Channel]) -> Workload:
    """Wire processes and channels into a pipeline.

    Args:
        processes: List of processes
        channels: List of channels connecting them

    Returns:
        Pipeline workload

    Example:
        pipeline = connect([loader, computer, storer], [l2c, c2s])
    """
    return Workload("connect", processes=processes, channels=channels)


def replicate(proc: Process, count: int) -> list[Process]:
    """Create multiple instances of a process.

    Args:
        proc: Process template
        count: Number of instances

    Returns:
        List of process instances

    Example:
        workers = replicate(worker, 4)
    """
    return [proc for _ in range(count)]


# Event API (Event = Channel<Signal, 0>)
#
# Events use rendezvous semantics (depth=0):
# - record() signals and blocks until synchronize() consumes
# - synchronize() waits for record() to signal
# - query() non-blocking check if event was signaled


def record(event: Event) -> None:
    """Signal completion on an event.

    For depth=0 (rendezvous), this blocks until synchronize() is called.
    For depth>0, this adds a signal to the buffer.

    Args:
        event: Event channel to signal

    Example:
        event = Event("sync", depth=0)

        # In producer thread
        compute()
        record(event)  # Signal completion

        # In consumer thread
        synchronize(event)  # Wait for signal
        use_result()
    """
    import threading

    # Ensure event has synchronization primitives
    if not hasattr(event, '_lock'):
        event._lock = threading.Lock()
        event._cond = threading.Condition(event._lock)
        event._signal_count = 0
        event._wait_count = 0

    with event._cond:
        if event.depth == 0:
            # Rendezvous: wait for a synchronize() call
            event._signal_count += 1
            event._cond.notify_all()
            # Wait for consumer
            while event._wait_count < event._signal_count:
                event._cond.wait()
        else:
            # Buffered: add signal and notify
            event._signal_count += 1
            event._cond.notify_all()


def synchronize(event: Event) -> None:
    """Wait for completion signal on an event.

    For depth=0 (rendezvous), this blocks until record() is called.
    For depth>0, this consumes one signal from the buffer.

    Args:
        event: Event channel to wait on

    Example:
        event = Event("sync", depth=0)

        # Wait for producer to signal
        synchronize(event)
        print("Producer completed")
    """
    import threading

    # Ensure event has synchronization primitives
    if not hasattr(event, '_lock'):
        event._lock = threading.Lock()
        event._cond = threading.Condition(event._lock)
        event._signal_count = 0
        event._wait_count = 0

    with event._cond:
        # Wait for a signal
        while event._signal_count <= event._wait_count:
            event._cond.wait()

        # Consume the signal
        event._wait_count += 1
        event._cond.notify_all()


def query(event: Event) -> bool:
    """Non-blocking check if event is signaled.

    Returns True if there is at least one unconsumed signal.

    Args:
        event: Event channel to check

    Returns:
        True if event has been signaled but not yet consumed

    Example:
        event = Event("sync", depth=0)

        if query(event):
            print("Event signaled")
        else:
            print("Event not yet signaled")
    """
    # Ensure event has synchronization primitives
    if not hasattr(event, '_signal_count'):
        return False

    if not hasattr(event, '_wait_count'):
        return event._signal_count > 0

    return event._signal_count > event._wait_count
