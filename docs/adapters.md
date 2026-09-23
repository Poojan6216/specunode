# Writing an adapter

Two things plug into SpecuNode: a **graph adapter**, which tells the runtime how your agent is
driven, and a **tool adapter**, which is the function behind a declared tool. This page is the
contract for both.

## Tool adapters

A tool is an async function. Four requirements, and each one has a failure behind it.

**Cancellable.** Squashing a branch cancels its task. A tool that blocks the event loop cannot
be cancelled, so a wrong guess keeps paying for it until it finishes. Use `await` on real I/O;
if you must call blocking code, use `asyncio.to_thread`, which also copies the context the
runtime needs. A raw `threading.Thread` loses the run scope entirely and its calls become
invisible — the runtime cannot stage or journal what it cannot see.

**Idempotent on the key, if you declared it so.** The runtime passes a deterministic
idempotency key with every dispatch. If your tool declares `idempotent=True`, a second delivery
under the same key must be a no-op upstream. If it is not, say `idempotent=False` and the
runtime will dead-letter rather than redeliver in the one window where it cannot tell.

**A witness, if you want staleness checked.** A `READ` declared `witness=True` returns
`{"value": ..., "witness": ...}`, where the witness is a version, ETag or row counter that
changes when the underlying data does. Without one, the runtime cannot tell whether the value
went stale between speculating and confirming, and reports the read as *unwitnessed* rather
than fresh.

**Honest about what it does.** See [effect-classes.md](effect-classes.md). A `READ` that writes
defeats the store buffer, and nothing in the runtime can notice.

### Failures

Raise `ToolDispatchError` and say whether the request left the process:

```python
from specunode.buffer.dispatcher import ToolDispatchError

raise ToolDispatchError("connection refused", sent="no")     # safe to retry
raise ToolDispatchError("gateway timeout", sent="maybe")     # may already have happened
```

That single bit is what keeps the ambiguous crash window narrow. `sent="no"` means a retry
cannot duplicate anything; `sent="maybe"` — the default, because it is the safe assumption —
means the upstream may already have acted, and a non-idempotent tool is dead-lettered instead
of guessed at.

## Graph adapters

Implement `GraphAdapter`. The shape that matters is `capabilities().drives_itself`.

**`drives_itself = False`** — the scheduler drives. It calls `next(state)` for the node and
`run_node(node, session)` for its decision. This is the plain-Python path.

**`drives_itself = True`** — your framework owns the loop and the scheduler waits inside
`drive(session, inputs)`. Your node bodies call back through `session.run_in_node(name, body)`,
which mints a branch, runs the body, and retires it. This is the LangGraph path, and it exists
because re-deriving a framework's routing and reducers outside it breaks the one property that
makes wrapping worthwhile: that a wrapped graph reaches the same final state as an unwrapped
one.

### Run the body as a task, not inside a context manager

A node parked on a staged write's result has to retire **while its body is still suspended**.
An `async with` around the body only reaches its exit after the body returns, and the body is
waiting for something only the retirement produces. That is a deadlock, and it fires on the
first write of the first sequential run, before any speculation is involved. Hand the runtime a
thunk; let it own the task.

### Naming several nodes at once

On the plain path a router may return a list of node names instead of one. The runtime reads
that as "these are independent": it forks every one of them from the same committed state and
program position, runs their bodies side by side, and retires them one at a time in the order
the list names them. So their effects reach the world in that order, and their idempotency keys
are the same whether they overlapped or not. `policy.parallel_nodes: false` runs the same group
one body at a time; it changes the wall clock and nothing else.

What the list promises is yours to keep, and the runtime checks what it can:

- **Two nodes may not both write one state key** unless a reducer is declared for it. The clash
  is refused before anything is sent whenever it is visible by then -- that is, whenever the
  writes to the key come before the nodes' own writes to the world. A node that writes the key
  only after its own write returns can be checked only after that write has gone out.
- **A node that read what an earlier one in the list then changed is refused** at its
  retirement, if the read was witnessed, and nothing it staged is sent. It is not re-run.
- **Bodies overlap up to each one's first write.** A body parked on a staged write stays parked
  until its turn to retire, so a group of nodes that each write early overlaps little. The
  shape that gains is several model-bound investigations followed by one node that acts on all
  of them -- `examples/fanout_agent`.

### Retirement can end mid-node

A node that stages a write and then reads the result retires in the middle of itself: the
branch is confirmed, the buffer drains, the node resumes with the real value. Do not assume a
node body runs to completion before its effects are dispatched.

## Speculation and node bodies

**A speculative branch does not run your node bodies by default.** A node body is unbounded
code — it can touch the filesystem, a socket or a global, none of which the runtime can see —
so speculation executes *predicted tool calls* and, in read-only stretches, the next model turn.

Opting a node in with `@specunode.node(speculable=True)` is you saying its body is safe to run
and throw away. A predicted route into a node that has not opted in stalls with
`NODE_NOT_SPECULABLE`, which is named rather than silent so the benchmark's hazard histogram
stays complete.

## Anything the runtime did not derive must be declared

Hard Rule 13's check is **designed** to rebuild each prompt from the journal and compare it to
what was sent. Material the runtime cannot re-derive — a system message assembled from graph
state, a retrieved document, a templated turn — is meant to pass through
`PromptBuilder.inject()`, which records it so the derivable part stays under an exact comparison
and the rest is counted.

**That rebuild is not implemented, and the runtime fails closed instead.** A branch that sent no
request while speculating has nothing to rebuild, which is every run any shipped configuration
produces: a speculative child runs a single tool call and never opens a turn of its own. A
branch that *did* send one is refused at retirement — the store buffer declines to drain it and
the run stops — rather than being stamped as checked. The ledger reads `context_identity:
unchecked` on every run, and that stamp is the truth rather than a formality.

Why it is refused rather than approximated: `fold_context` reconstructs the message list, while
the recorded `request_hash` covers the whole projected envelope, so hashing one against the
other can never match. The repair that suggests itself — rebuild from the branch's own message
list — compares that list to itself and passes every time. A Rule 13 implementation that never
fires is worse than none, because it still prints a stamp; for a while this one printed the
stamp with no implementation behind it at all, which is how that sentence came to be written
about its own author.

## The adapter suite

`tests/test_adapter_suite.py` runs every bundled adapter and the fake world's tools through
these requirements. If you write an adapter, run it through the same suite.
