# SpecuNode

**Speculative out-of-order execution for agent graphs, with a store buffer.**

An agent graph runs strictly serially: the model thinks, emits a tool call, waits for the tool,
thinks again. Most of that time is spent waiting on one thing at a time.

A processor has had this problem since the 1960s and solved it. It guesses which way a branch
goes, runs the next instructions early, and keeps any *writes* in a holding area called a store
buffer. Only once the guess is confirmed do the writes leave. If the guess was wrong, the buffer
is discarded and nothing outside the processor ever saw it.

SpecuNode applies that to agents. Reads run early. Writes go into a store buffer tied to the
speculative branch that produced them. When the model's real decision arrives, it is compared to
the guess — exactly, not approximately. Match: the buffer drains to the world. Mismatch: the
buffer is discarded, unsent. **The model's real decision is the only thing that can ever release
a write.**

---

## The problem this fixes, in one table

Prior work on speculative tool execution has to stop at the first tool call that changes
something, because running it on a guess cannot be undone. Here is what happens if you do it
anyway. Fifty journaled transcripts, ten of which the predictor gets wrong, executed by three
runtimes against the same fake world:

```
runtime             runs  mispredictions  charges reaching world  charges from squashed branches
------------------------------------------------------------------------------------------------
naive-parallel        50              10                      60                              10
specunode             50              10                      50                               0
sequential            50               0                      50                               0
```

```
python bench/demo.py --demo leak
```

The last column is the point, and so is the fact that the second and third rows are equal.
`naive-parallel` charged ten cards for a decision the model never made. SpecuNode staged those
same ten predicted charges and discarded them unsent.

Those figures come from that command; they are committed in
[`bench/results/demo_leak.json`](bench/results/demo_leak.json) and a test asserts the file still
matches a fresh run.

---

## Status

Under construction, and honest about where it is. Working today:

- the canonical form, the hash-chained journal, replay and crash recovery
- effect classes, the store buffer, at-least-once dispatch with deterministic idempotency keys
- the sequential scheduler, the LangGraph integration, the plain-Python API, `resume`
- tier-0 early issue and the tier-1 pattern index
- the three invariant tests below

Not yet: the MCP proxy, the offline and online benchmarks, the adversarial suite, `RESULTS.md`.
No latency figure appears anywhere in this repository, because none has been measured yet.

---

## The three tests that are never skipped

**The leak test** (Hard Rule 3). Five hundred randomly generated branch trees, random resolution
outcomes, random faults. After every run: the branches that touched the world are a subset of the
branches that retired. It checks a second invariant too, because the first cannot see the bug the
spec names as the planted one — in a single process, a drain that runs before its confirming
journal entry is durable still happens on a branch that does retire. So every dispatched effect is
also traced to a confirming entry that was durable when it went out.

**The equivalence test** (Hard Rule 9). The effect ledger of a run with speculation on equals the
ledger of the same run with speculation off. The workload declares how many effects it produces,
so a runtime that dispatched nothing cannot pass by comparing two empty ledgers.

**The context-equivalence test** (Hard Rule 13). The speculative arm asked the model the same
questions as the sequential arm. It asserts on what a model *received*, never on what the runtime
says it sent — the live check and the retirement-time rebuild share a prompt builder and can be
wrong in the same way while agreeing with each other.

---

## What it deliberately does not do

A staged write returns a placeholder, and **a placeholder never enters a prompt**. The sequential
run would have shown the model the real result, and a model conditioned on a placeholder is
deciding on a different premise even when it happens to decide the same thing. So the model call
after a staged write waits for the real value.

What speculating past a write buys, therefore, is that *tool* latency is hidden — independent
reads and further staged writes proceed while the in-flight turn and the drain are still going.
Model latency is hidden only in read-only stretches. A workload shaped `model → write →
model(reads the write's result)` gains nothing at all from it, by design, and the benchmark will
report how much of each workload has that shape.

---

## What beats it

These are measured, not argued away. Each is a real hole, and
[`docs/limitations.md`](docs/limitations.md) explains what it costs.

- **A tool declared `READ` that writes** defeats the store buffer completely. The effect class is
  your word and nothing checks it.
- **A `READ` whose upstream enqueues work** is a write in disguise. Its synchronous response is
  indistinguishable from a real read's, and a squashed branch's queued job still runs.
- **Speculative reads reach real systems**, squashed branch or not. If your reads are metered or
  audited, speculation costs you those reads. The ledger counts them rather than netting them off.
- **Reads without a witness cannot be checked for staleness.** They are reported *unwitnessed*,
  never as fresh.
- **The ambiguous crash window is real.** If a process dies between a request reaching the world
  and its acknowledgement being recorded, nobody can tell whether it took effect. A tool that
  declared itself idempotent is redelivered; one that did not is dead-lettered for a human.
- **A placeholder transformed inside a string** — base64, hex — can evade hazard analysis.
- **This is not an authorization layer.** If the model actually emits `send_email`, it is sent.

The docs say "at-least-once dispatch with deterministic idempotency keys" and never claim
"exactly-once". A vocabulary check fails the build if any document here makes an absolute claim
without naming the condition that makes it true, and a traceability check fails the build if any
number appears that no committed results file contains.

---

## What this is not

- **Not a durable-execution platform.** Temporal, DBOS, Restate and Inngest do that, and do it
  better. A SpecuNode run is designed so it could later be hosted *inside* one of them.
- **Not an authorization layer.** See SCOPEGATE below.
- **Not a context manager.** No pruning, no summarisation, no distillation anywhere in the runtime.
- **Not a new agent framework.** It wraps yours.

---

## Prior art, and where each one stops

Every project below exists, works, and solves its own problem. SpecuNode is the combination of
three of them, and the combination is the only part that is new.

**PASTE** — *Act While Thinking: Accelerating LLM Agents via Pattern-Aware Speculative Tool
Execution* (Sui et al., Microsoft Research, arXiv 2603.18897). Mines patterns from prior
trajectories and pre-executes the predicted next call.
Reports a 48.5% reduction in average task completion time and 1.8x tool throughput [cited].
*Stops at:* side effects are handled by policy
exclusion — a tool with side effects is simply not speculated, so speculation ends at the first
mutating call. **The tier-1 pattern drafter here is a re-implementation of their idea and is
credited as theirs.**

**Claude Code's streaming tool executor** (*Dive into Claude Code*, arXiv 2604.14228). Starts each
tool the instant its `tool_use` block is parsed from the stream. *Stops at:* early issue, not
prediction — it only runs what the model has already emitted, and writes are serialised. **This is
exactly the tier-0 drafter here, credited as theirs.**

**langchain-nvidia-langgraph.** Compile-time parallelisation plus speculative execution of both
branches of a conditional edge. *Stops at:* both branches' tools run for real. That is the failure
the table above exhibits.

**ToolAhead.** Prefetches read tool results for coding agents. *Stops at:* reads only.

**SagaLLM** (Chang & Geng, VLDB 2025, arXiv 2503.11951). Sagas, compensation and validation for
multi-agent planning. *Stops at:* no speculation, and an LLM in the recovery path. Its
compensation vocabulary is adopted here for the `COMPENSABLE` class.

**ATP / Mnemosyne** (arXiv 2607.00269). A generated action holds no authority until a
deterministic gate admits it. *Stops at:* admission happens on the serial path; there is no
concept of an action that executed early and awaits admission. Retirement here is an ATP-style
gate, and the store buffer is what sits in front of it.

**SCOPEGATE** (arXiv 2606.28679). Finds that no mainstream framework re-authorises each
model-emitted call against its concrete argument values. *Stops at:* authorization, no execution
model. Its point is why effect classes here are declared out of band and never inferred.

**Temporal, DBOS, Restate, Inngest, LangGraph checkpointers.** Journal non-deterministic results so
replay reuses them. *Stop at:* a step is a step — nothing runs before its predecessor completes,
and there is no buffer and no branch to squash.

**Out-of-order CPUs** — Tomasulo (1967), store buffers, squash-on-mispredict, in-order retirement.
Not LLM work, but the model this design copies deliberately, down to the vocabulary.

---

## Why a store buffer and not a saga

A saga runs the write and undoes it if something later fails. Compensation is a *second* effect
that reaches the world, and for sending an email, charging a card or delivering a webhook it is
imperfect or impossible. A store buffer never lets the first effect out. Compensation is kept — as
the `COMPENSABLE` class, for effects that have already retired — but it is not what makes
speculation safe.

## Why exact equality and not similarity

A branch that guessed has already run reads and staged writes on that premise. If the real
decision differs in any argument — a different ticket id, a different amount, a different path —
every downstream call it made was computed from something false. There is no useful notion of
"close enough" for a tool call, and any tolerance is a route for a wrong branch's effects to
retire.

---

## License

Apache-2.0. See [LICENSE](LICENSE).
