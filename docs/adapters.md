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

**A way to be asked, if you want a lost reply answered rather than escalated.** A crash, or a
reply that times out, can land after the upstream took a write and before its reply came back.
The runtime knows that write may be out -- it was claimed in the journal under its key before it
was sent -- but not whether it is. An idempotent tool is simply sent again under the same key. A
non-idempotent one is never sent again on a guess: it is dead-lettered for a human, unless it
says how to find out:

```python
async def charge_was_taken(key: str, args: dict) -> dict | None:
    """The upstream's own record of a request key, e.g. a payment's idempotency key."""
    charge = await payments.find_by_request_key(key)
    return None if charge is None else {"charge_id": charge.id}

@specunode.tool(effect="write", reconcile=charge_was_taken)
async def charge_card(customer_id: str, amount: float) -> dict:
    key = specunode.current_idempotency_key()
    return await payments.charge(customer_id, amount, request_key=key)
```

Return the upstream's result if the request took effect -- the node gets it exactly as if the
reply had arrived -- or `None` if it did not, and it is sent once. If asking fails, the write is
dead-lettered, as it would have been. Answer from a record the upstream writes *atomically with
the effect*; a lookup that can miss a request still in flight can say "no" to one that lands a
moment later.

### Failures

Raise `ToolDispatchError` and say whether the request left the process:

```python
from specunode.buffer.dispatcher import ToolDispatchError

raise ToolDispatchError("connection refused", sent="no")     # safe to retry
raise ToolDispatchError("gateway timeout", sent="maybe")     # may already have happened
```

That single bit is what keeps the ambiguous window narrow. `sent="no"` means a retry cannot
duplicate anything, and the dispatcher retries it. `sent="maybe"` — the default, because it is
the safe assumption, and what any other exception counts as — means the upstream may already
have acted: an idempotent tool is retried, and a non-idempotent one is not called again. The
runtime asks its `reconcile`, if it has one, and otherwise dead-letters it and stops for a human.

A dead letter whose request may have left is not retried by a resume either. Check the upstream,
then record what you found -- `specunode resolve <run> <key> --landed` or `--not-sent`, with the
key as `specunode ledger` prints it -- and the resume skips it or sends it once. A dead letter
whose request demonstrably never left is retried by a plain resume: heal the upstream and
resume. A retry is marked in flight again before it is sent, so a crash while it is out is the
lost reply it may be, not another "never sent".

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

An adapter is shared; a Scheduler is not. One wrapped graph serves any number of runs at once
-- how a web handler calls it -- each driven by a Scheduler of its own, which is what `wrap()`
builds per call. So keep nothing about a run on the adapter: what a node needs about its run
comes from the session it is handed.

### Run the body as a task, not inside a context manager

A node parked on a staged write's result has to retire **while its body is still suspended**.
An `async with` around the body only reaches its exit after the body returns, and the body is
waiting for something only the retirement produces. That is a deadlock, and it fires on the
first write of the first sequential run, before any speculation is involved. Hand the runtime a
thunk; let it own the task.

### Naming several nodes at once

On the plain path a router may return a list (or tuple) of node names instead of one. The
runtime reads that as "these are independent": it forks every one of them from the same
committed state and program position, runs their bodies side by side, and retires them one at a
time in the order the list names them. So their effects reach the world in that order, and their
idempotency keys are the same whether they overlapped or not. `policy.parallel_nodes: false`
runs the same group one body at a time; it changes the wall clock and nothing else.

The order is load-bearing, so a router must return a list or a tuple: a set is refused, because
its order is decided afresh in every process and a resume would mint the lanes under different
ids. An empty list is refused too (return `None` to end the run), as is a node named twice.

**A group is one decision, and a crash does not re-make it.** The journal records the group
before any lane forks. If the process dies in the middle, a resume finishes that group -- its
unretired lanes, under their own node ids, from the position and the state the group forked
from -- instead of asking the router again, which would see the state some lanes had already
committed. A lane whose writes went out before the crash derives the same keys, and the dedupe
table claims them rather than sending them twice.

**A reducer combines lanes by value.** Each lane's delta was taken against the group's starting
state, so it is not replayed on top of a sibling's commit. For each key a lane touched, the
reducer sees the committed value and the value the lane wrote -- what it sees for nodes run one
after another, and how a fan-out's updates are combined in LangGraph. A lane's value is taken
whole: two lanes editing parts of one object combine only through a reducer that merges them.

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

### A write waits for the turn that decided it

A node that reads `session.model.stream()` itself may make reads as blocks parse, but not
writes. A write made before the stream reaches `TurnComplete` is refused with a
`SchedulerError`: staged, it would go out as soon as the node parked on it, before the turn
that decided it was on disk, and a crash in between would leave a sent effect whose decision a
resume could not find. Read the stream to its end, or use `complete()`, or `call_turn`, which
issues a turn's reads early and stages its writes once the turn is journaled.

The runtime cannot tell which turn decided a write, so the rule covers the whole node run: a
write is refused while *any* model turn the node started is unjournaled. That includes a turn
still in flight in a background task, and one the node read from `stream()` and stopped
reading, or that failed, after blocks had reached it -- those stay unjournaled for good, so
that node writes nothing more; let it fail and resume. A `complete()` or `call_turn` that fails
hands the node nothing of the turn, and does not block it.

### A model call that fails raises `ModelError`

Whatever the client raised -- the SDK's own error, a dropped connection, a reply cut off -- a
node sees `specunode.ModelError`, with the client's error as its cause. That is what a resume
serves back, and what a replay raises, at the same point: the failure is journaled as the turn's
outcome, so a node that catches it and asks again is matched with its second question. Catch
`ModelError`, not the client's own type: a node that catches the client's type is not the same
node on resume. A turn the node stops waiting for -- its timeout, a cancel -- is journaled as
cancelled too, and served as one that never answers, until well past when the node stopped
waiting the first time; a node still waiting then ends with `specunode.TurnAbandoned`. It is a
`BaseException`, not a `ModelError` or any `Exception` -- asking again would be asking something
the recorded run never asked -- and the node is closed to writes and model asks before it is
raised, so a `finally` cannot act on the wrong path either. Do not catch it: a node that does,
and returns, is not committed, and the run stops all the same. Served answers come back at their
recorded pace and order -- a stream piece by piece, as its caller had it, each piece no sooner
than the model sent it -- so a node that races two calls, falls back on a timeout without
cancelling, or gives up on a model slow to start, decides on resume as it did. An answer waits
for an earlier one only until the node is done with that one -- answered, failed or given up on
-- and a node that holds an earlier answer open far longer than the recorded run did, while it
waits for a later one, is stopped with `TurnAbandoned` too.

## Model clients

A model client -- `specunode.integrations.anthropic.AnthropicModel`, or your own -- has two
methods. `complete(envelope)` returns the whole reply. `stream(envelope)` yields its pieces as
they parse -- `TextDelta` for text, `ToolUseComplete` for each tool call as it finishes -- and
then `TurnComplete` with the whole reply. The runtime reads the stream as it arrives, whoever is
reading it, and records each piece with when it came, so a resume and a replay hand the pieces
back at that pace.

**Each piece's `index` is the position of its block in the finished reply.** A block that
streams nothing -- a thinking block -- still takes its position, so the pieces after it keep
theirs. A resume serves a piece against the block at its position; one that does not fit its
block stops the node with `TurnAbandoned` rather than be served as something else.

**A failure may come at any point**, and is recorded as the turn's outcome wherever it comes:
`stream()` itself raising before it yields anything -- a client's own rate limiter -- a failure
mid-stream, or a reply cut off. Raise `ModelError` for a failure the node may catch and ask
again after; anything else is raised to the node as a `ModelError` caused by it.

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
