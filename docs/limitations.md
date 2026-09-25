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
was right. Every ledger reports `speculative reads upstream` for exactly this reason. That
number counts every read that reached upstream without a durable decision behind it, including
a read issued early for a turn that is not yet journaled; `max_speculative_reads` bounds the
narrower half of it — reads made on a forked guess — and the ledger prints that count beside
the wider one, because the gate can close on one while the other keeps growing. The runtime
does not hide either number and does not net them off against the latency it saved.

## Reads without a witness cannot be checked for staleness

A branch reads, then time passes, then the model's decision confirms the branch. Between those
two moments the value may have changed. A read that returns a version or ETag can be re-checked
before the branch retires; a read that does not, cannot, and is reported as **unwitnessed**
rather than as fresh. That distinction is the honest number, and the benchmark publishes the
fraction of stale reads that were undetectable rather than only the fraction it caught.

## Dispatch is at-least-once, and the ambiguous window is real

An effect is dispatched with a deterministic idempotency key, and the dispatcher deduplicates
its own retries. It cannot deduplicate the network beyond itself.

If the process dies between a request leaving it and the acknowledgement being recorded -- or
the reply times out without a crash -- nobody can tell whether the effect took place. That is the
two-generals problem and no amount of bookkeeping removes it. What the runtime does instead is
refuse to guess: a tool that declared `idempotent=True` is redelivered, and a tool that did not
is asked about through its `reconcile` or **dead-lettered**, and the run halts for a human. A
resume does not retry that dead letter either, until someone who has checked the upstream says
what happened (`specunode resolve`). The consequence, visible in the kill/resume tests, is that
a resumed run can reach a *prefix* of the effects an uninterrupted run reached.

Until 2026-09-25 the dispatcher retried every failure, whatever the tool declared, so a gateway
that took a charge and timed out on the reply was charged again on the next attempt -- with the
default settings and no crash at all; a resume retried every dead letter, including one whose
request had landed; and a retry of one that had never left stayed marked "never sent" while it
was out, so a crash during it made the next resume send it again. Independent reviews found all
three.

**The dedupe guarantee holds only while a resumed node makes the calls it made before.** An
idempotency key is derived from the run, the node, the program position, the tool and the
*arguments*, so a call is recognised as already sent only if it is made again with the same
arguments. A resume keeps the model's side of that: a node whose earlier answer may already
have sent something -- an effect dispatched, a claim with no outcome, a dead letter that may
have left -- is served that answer from the journal when it asks the same question again,
rather than asking the model (`RecordedTurns`), and so are the answers before it in the same
conversation, which its question was built on. An answer after the last thing that may have
been sent is asked for again: there is nothing to protect, and a resume can then recover from
an answer that failed, such as a call to a tool that does not exist. And nothing is sent before
the model turn that decided it is journaled: a node that reads `session.model.stream()` itself
and writes before the turn ends is refused (docs/adapters.md), because its write would go out on
a decision not yet on disk.

What a resume cannot keep the same is everything else that shapes a call. Reads are made again
on resume, not served, so a node that reads and then asks in the same step asks something new if
the world changed in between -- and the journaled answer, which answers a different question, is
not served. A node can also build a call's arguments from a fresh read, a clock, or code that
changed, without any of it reaching the prompt: then the same question is served the same
answer, and the call still comes out different. Either way the resumed run makes a different
call at the same position, under a different key, and the dedupe table has nothing to match it
against. The world then receives both, and no bookkeeping in this design connects them.

So the honest statement is: **a resumed run never applies the same call twice, and can deliver
a second, different call only when something that shapes it changed across the crash.** A
tool declared idempotent may be handed the *same* call again, after a crash lost the reply to
its first delivery; that is what declaring it idempotent permits.

It also assumes one driver per run. `run` and `resume` hold the run for as long as they drive
it -- within a process, and across processes through a lock the operating system (or, for a
Postgres journal, the database) releases when the process holding it dies -- and a second
attempt to drive it raises `RunBusy`. Two resumes of one run at once used to take up the same
claim, and between them send it twice. The lock is not taken on Windows, where only the
in-process half applies.

There is no setting that makes a model answer a changed question the same way, and on the
current models there is not even one that narrows it: `temperature`, `top_p` and `top_k` are
rejected outright by Claude Sonnet 5 and Claude Opus 5 (HTTP 400, "`temperature` is deprecated
for this model"), so the advice this page used to give — pin the temperature to zero — is no
longer available to take. Claude Haiku 4.5 still accepts them. Either way a model is free to
answer differently, and the design does not assume otherwise.

Until 2026-09-25 this section was wider. A resume asked the model again for every turn of a
node that had not finished, even one whose answer had already sent something, so the dedupe
guarantee depended on the model answering the same way twice however little had changed; and a
node that wrote as a streamed block parsed sent its write before the turn was journaled. Three
independent reviews found them and what was wrong with the first fixes. The kill/resume test now resumes every kill point with a model
that would decide differently if asked -- including a node that asks and charges in one step --
and holds each to the outcome what the kill left on disk requires. The case left over, a call
shaped by something that changed, it does not exercise.

Serving applies to runs continued with `resume`. On the LangGraph path a crashed run is continued
from LangGraph's checkpointer by running the graph again, and an unfinished node asks the model
again, as it did before any of this.

The ambiguous window itself -- the upstream took the call, the reply never came back -- is
closed only by the upstream. A tool that declares a `reconcile` is asked, on resume, whether the
call under its key took effect, and the runtime acts on the answer; one that does not is
dead-lettered. `reconcile` is only as good as the record it reads: it must be one the upstream
writes atomically with the effect, or a request still in flight can be reported as absent and
then land (`docs/adapters.md`).

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

## Guessing buys at most one block, and guessing a write buys nothing

This is the finding that decides what the project should claim, so it is stated plainly.
Every wall-clock saving measured against a real model in this repository comes from **tier-0
early issue** — a read
starts the moment its block finishes parsing instead of after the turn — and none of it,
within any interval the benchmarks resolve, from guessing. Three properties of the scheduler
explain why, and `bench/offline/run_break_even.py` measures each with a guesser of controlled
accuracy (`RESULTS.md`, "What a guess is worth"):

- **A guess runs at most one block ahead of the model.** The drafter is asked after each block
  the model emits, and the next block resolves the guess. So a correct guess starts its call
  one block's streaming time early, and that is worth something only when the call is slower
  than early issue could already hide.
- **A staged write cannot leave before its branch retires, and the drafter does not chain past
  one.** So guessing a write correctly moves nothing forward in time. Speculating on writes as
  well as reads — the setting the store buffer makes *safe* — is measured to be no *faster*
  than PASTE's reads-only rule, even with a guesser that is always right.
- **A miss costs no wall clock.** It is squashed when the real block arrives and the real call
  is issued as it would have been. What a miss costs is money: the upstream read its branch
  made, and for a draft model the tokens. That is why the gate that matters for a real
  deployment is `max_speculative_reads` and `max_wasted_tokens`, not the alpha floor.

So the store buffer's contribution is safety — no guessed effect can reach the world, and the
model's own writes are held until the turn that asked for them is durable — and early issue's
is speed. A drafter that chained several calls ahead, or a scheduler that let a confirmed
branch's writes dispatch before retirement, would change this; neither is built.

## Fewer model replies need an app that runs every call a reply asks for

When the model is the slow part of a run, the lever infrastructure has is the number of model
replies. Measured against Claude Sonnet 5 (`RESULTS.md`), the model already asks for
independent calls together when nothing tells it otherwise, and went on doing so when its prompt
told it not to; only the API's `disable_parallel_tool_use` held it to one. What keeps a run at
one call per reply is the app around the model running only one — as the 300 trajectories in
`bench/corpus` do, and as two of this repository's three original sample apps still do: they
take the first call of each reply and drop the rest. The runtime is built for
the reply that asks for several: `specunode.core.loop.agent_loop` runs its reads together as
they parse, holds its writes until the reply is durable, and hands every result back in one
message.

Two limits remain, and neither is the runtime's to remove:

- **Calls that depend on each other still need a reply each.** The restarts wait for the
  statuses and the summary waits for the restarts, so the task sets the floor on replies.
- **Fewer replies save each reply's fixed cost, not the writing.** Eleven replies became five,
  and the time saved was smaller than that, because a reply that asks for several calls takes
  longer to write. Prompt caching cut the bill and not the time at that prompt size.

## Parallel nodes overlap only up to their first write, and must be independent

A router that names several nodes at once gets their bodies run side by side and retired in the
order named (`docs/adapters.md`). Two consequences follow, and neither is hidden:

- **Overlap stops at each body's first write.** A write is sent only when its node retires, and
  nodes retire in order, so a body waiting on its own write's result waits for every node named
  before it. Nodes that investigate and hand their findings to one node that acts overlap
  completely; nodes that each act early barely overlap.
- **Independence is declared, and only partly checked.** Two nodes writing one state key are
  refused, at the earliest point the clash is visible. A node that read what an earlier node
  then changed is refused at its retirement -- but only if the read was witnessed; an
  unwitnessed read cannot be checked (above) and retires on the value it saw. Refused means the
  run fails, loudly, with nothing of the refused node sent. It is not re-run on the fresh value.
- **A clash found after a lane's write went out cannot un-send it.** A lane that writes a state
  key only after its own write returns is checked only then; the lane is recorded as faulted,
  the error says how many of its effects were already dispatched, and the lanes after it are
  abandoned unsent. A resume re-runs the unfinished lanes under the same keys -- so nothing is
  sent twice -- and refuses the same clash, unless a reducer has been declared for the key.

## A drafter cannot use the result of the call it was just asked about

The drafter is consulted immediately after a `tool_use` block finishes parsing. At that instant
the call in that block has only been *issued* — its task is created and the drafter is asked
before it has had a chance to run — so the drafter's view of prior results never includes the
one belonging to the block it is predicting from.

That halves the reach of the data-flow idea SpecuNode borrows from PASTE. An argument the
previous call *returns* is not available for filling the next call; the earliest usable result
comes from a block at least two back. In practice a prediction can chain off block *j-1*'s
result while standing at block *j*, and nothing shorter.

It is a structural property of early issue rather than a tuning problem. Waiting for the read
before asking the drafter would serialise the exact thing early issue exists to overlap, and
would make the predictor's latency a function of the tool's. `tests/integration/test_t1_end_to_end.py`
is shaped around it, with a three-call turn rather than a two-call one.

## A tool call that opens its own model turn cannot be speculated past, and most of them do

The drafter's history is the calls *within the current turn*. A turn that emits exactly one
tool call therefore offers nothing to predict from — there is no block *j+1* to guess, and the
next call belongs to a turn that has not started.

This is not a corner case. It is 1.0000 of the tool calls in the offline corpus, and it is why
the measured speculable span past a write is 0.0000 there. Two of the three sample apps have
that shape deliberately, so the test suite keeps resembling the thing it is a model of.

Measured directly (`RESULTS.md`, "The acceptance rate, measured"): under this policy no tier-1
guess on that corpus can ever be resolved, and carrying guesses across turns — which the runtime
does not do — would confirm three of them. The limit is the corpus's turn shape and, behind it,
the fact that a tier-1 guess can only copy values it has already seen.

## Two of the three sample apps route around speculation entirely

`support_agent` and `research_agent` call the model directly and then issue each tool
themselves. That is a supported pattern — Demo 1 uses it — but tier-0 early issue and the
drafters live inside the turn the *runtime* drives, reached through `session.call_turn`. On a
node that calls `session.model.complete` and then `session.call_tool`, no drafter is ever
consulted and no read is issued early.

Nothing warns about this. The run is correct, journaled, replayable and equivalent; it is
simply sequential. `ops_agent` is the one sample app that hands its turn over, and the
equivalence test asserts the distinction per workload so that "tier 1 declined" and "tier 1 was
never asked" cannot be confused for one another.

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
