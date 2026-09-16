# What SpecuNode cannot do

This page is not a disclaimer. Everything on it is a property the runtime does not have, that
a reader might reasonably assume it does, and most entries name the benchmark that measures
how badly it bites.

## The effect class is your word, and nothing checks it

A tool is `READ`, `WRITE`, `COMPENSABLE` or `IRREVERSIBLE` because a developer said so. No
heuristic reads the tool's name, description or arguments, and no model is asked — that is
Hard Rule 2, and it exists because "it is called `get_*`, so it is safe" is exactly the
inference that eventually goes wrong on the one tool where it mattered.

The cost of that rule is that a tool declared `READ` which actually writes defeats the store
buffer completely. It will be executed speculatively, on a branch that may never retire, and
nothing in the runtime can notice. `bench/adversarial` measures the leak rate for this case
rather than arguing it away.

An undeclared tool is a `WRITE`. That direction is deliberate: being wrong about a write costs
a lost speculation, and being wrong about a read costs an effect that escaped.

## A read whose upstream enqueues work is a write in disguise

If a tool returns `{"status": "queued", "job_id": ...}` and something downstream of it later
writes, that tool is a `WRITE`, whatever its HTTP verb says and whatever its synchronous
response looks like. There is no way for the runtime to tell the difference by observing the
response, because by construction the response is indistinguishable from a read's.

Declare it `WRITE`. If you do not, a squashed branch's queued job still runs.

## Speculative reads reach real systems

A read issued on a branch that is later squashed has still been sent. If your reads are
metered, rate-limited or audited, speculation costs you those reads whether or not the guess
was right. Every ledger reports `speculative reads upstream` for exactly this reason, and
`max_speculative_reads` bounds it. The runtime does not hide this number and does not net it
off against the latency it saved.

## Reads without a witness cannot be checked for staleness

A branch reads, then time passes, then the model's decision confirms the branch. Between those
two moments the value may have changed. A read that returns a version or ETag can be re-checked
before the branch retires; a read that does not, cannot, and is reported as **unwitnessed**
rather than as fresh. That distinction is the honest number, and the benchmark publishes the
fraction of stale reads that were undetectable rather than only the fraction it caught.

## Dispatch is at-least-once, and the ambiguous window is real

An effect is dispatched with a deterministic idempotency key, and the dispatcher deduplicates
its own retries. It cannot deduplicate the network beyond itself.

If the process dies between a request leaving it and the acknowledgement being recorded, nobody
can tell afterwards whether the effect took place. That is the two-generals problem and no
amount of bookkeeping removes it. What the runtime does instead is refuse to guess: a tool that
declared `idempotent=True` is redelivered, and a tool that did not is **dead-lettered** and the
run halts for a human. The consequence, visible in the kill/resume tests, is that a resumed run
can reach a *prefix* of the effects an uninterrupted run reached. It will never reach effects
the uninterrupted run did not, and it will never deliver one twice.

The docs say "at-least-once dispatch with deterministic idempotency keys". They do not say
"exactly-once", and a vocabulary test fails the build if they ever do.

## Past-write speculation hides tool latency, not model latency

A staged write returns a placeholder, and a placeholder never enters a prompt — the sequential
run would have shown the model the real result, and a model conditioned on a placeholder is
deciding on a different premise. So the model call after a staged write always waits for the
real value.

A workload shaped `model → write → model(reads the write's result)` therefore gains nothing at
all from speculating past the write. That is by design, not a gap, and the benchmark reports
what fraction of each workload has that shape instead of quietly excluding it.

## Under-declared `forward_keys` silently defeats one hazard check

`forward_keys` is how the runtime knows that a read touches something a staged write touches.
It is the **only** detector for that class: witness validation happens before the drain, so the
branch's own staged write has not been dispatched and cannot have made anything stale.

An undeclared `forward_keys` fails closed — any read after any staged write becomes a hazard —
so the failure mode is lost speculation rather than a wrong answer. An *under*-declared one
does not, and sits on the same trust boundary as a misdeclared effect class.

## A placeholder transformed inside a string can evade detection

Hazard analysis scans the exact canonical bytes of a call's arguments for the placeholder
prefix, and walks the structure to attribute a hit. It catches a handle used as a value, buried
in a longer string, nested at depth, used as an object key, or case-mangled. It does not catch
one that has been base64-encoded, hex-encoded or otherwise transformed. `bench/adversarial`
reports the measured miss rate rather than a claim.

## This is not an authorization layer

If the model actually emits `send_email`, the runtime dispatches it. SpecuNode decides *when* a
call may take effect, not *whether* it is allowed to. Per-call authorization — scopes, money
ceilings, default-deny on concrete argument values — is a different mechanism, and SCOPEGATE
makes the case for it better than this page can.

## Free-text nodes are barriers

A node that emits prose cannot be predicted token for token, so it cannot be confirmed by
equality, so nothing speculates on it. A workload dominated by prose-to-prose handoffs gets
little or no speedup, and the benchmark says how much of each workload is behind such a barrier.

## Projections are not implemented

A staged write cannot supply a projected result to anything, in any configuration. A projection
has no resolution signal — there is no later model output to compare it against — so it would
retire unverified; and a call whose arguments were computed from a wrong one reaches the world
with different arguments from the sequential run, which would make Hard Rule 9's mandatory
equivalence test fail in a supported configuration.

## State on the LangGraph path belongs to the checkpointer

The plain-Python integration journals its state deltas, so a resume rebuilds state from the
journal alone. LangGraph owns its own reducers and channel semantics, and a second copy in the
journal would be a second answer to what the run's state is — so on that path, resume needs a
LangGraph checkpointer, and the journal's hash chain does not cover the state it holds.

## A node that escapes the ports is invisible

The runtime sees a node's tool calls and model calls because they go through its ports. Anything
else a node body does — writing a file, opening a socket, mutating a global — is outside them,
which is why a speculative branch does not run node bodies unless the developer marks the node
`speculable`. A node that starts a raw thread without propagating the context loses the run
scope entirely; `asyncio.to_thread` is fine, because it copies the context.

## It is a research prototype

One target-model provider adapter. LangGraph, plain Python and MCP. Three sample apps. A local
SQLite or Postgres journal, and no hosted anything. Running a SpecuNode run *inside* Temporal,
DBOS or Restate is documented as a pattern, not shipped as an integration — and SpecuNode is
not a replacement for any of them.
