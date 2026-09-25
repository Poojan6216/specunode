# SpecuNode

[![CI](https://github.com/Poojan6216/specunode/actions/workflows/ci.yml/badge.svg)](https://github.com/Poojan6216/specunode/actions/workflows/ci.yml)

**Agents that take real actions, and never take one twice.**

SpecuNode is a runtime for AI agents whose tools change the world: charging a card, restarting a
job, sending a message. It holds every write until the model's decision behind it is durable,
claims every effect under a deterministic key so that no crash can send it twice, journals
everything so any run can be replayed exactly, and runs a reply's independent calls -- and
independent nodes -- side by side. Plain Python, LangGraph and MCP.

## Pull the plug

A billing run -- look up three customers, charge each, send each a receipt, post one summary --
killed at each of its 14 dangerous moments and restarted the way each system restarts:

| | Finished, every effect once | Stopped for a human | Sent something twice | Extra effects |
|---|---|---|---|---|
| A plain async loop | 1 | 0 | 13 | 49 |
| LangGraph, a checkpointed node per customer | 4 | 0 | 10 | 13 |
| LangGraph, a `@task` per call -- its recommended pattern | 7 | 0 | 7 | 7 |
| **SpecuNode** | 0 | 14 | **0** | **0** |
| **SpecuNode, with a `reconcile` per tool** | **14** | 0 | **0** | **0** |

```
python bench/offline/run_crash_safety.py    # no model, no network, a few seconds
```

The crash that charges a customer twice is the one where the charge went through and the reply
was lost. A checkpoint cannot see it: the call either finished or it did not. SpecuNode claims
every effect in its journal under a deterministic key *before* sending it, so after a crash it
knows exactly which effects may already be out -- and then it asks the upstream (a tool's
`reconcile`) or stops for a human. It never guesses. Details and caveats in
[RESULTS.md](https://github.com/Poojan6216/specunode/blob/main/RESULTS.md#pull-the-plug-what-a-crash-sends-twice).

Nor does a resumed step ask the model again for a decision that may already have sent
something. Nothing is sent before the answer that decided it is journaled, and a resume is
served that answer from there -- so a model that would decide differently the second time
cannot add a second, different charge, as long as the step asks the same question and builds
the same call
([limitations](https://github.com/Poojan6216/specunode/blob/main/docs/limitations.md#dispatch-is-at-least-once-and-the-ambiguous-window-is-real);
this applies to runs continued with `resume`, not to a LangGraph graph re-run from its
checkpointer).

## Quickstart

```
pip install "specunode @ git+https://github.com/Poojan6216/specunode"     # PyPI release to come
git clone https://github.com/Poojan6216/specunode && cd specunode
python examples/quickstart.py
```

Extras: `anthropic` (the Claude adapter), `langgraph` (wrap a compiled LangGraph graph), `mcp`
(the MCP proxy), `postgres` (a Postgres journal) -- e.g.
`pip install "specunode[anthropic,langgraph] @ git+https://github.com/Poojan6216/specunode"`.

```python
import specunode

@specunode.tool(effect="write", reconcile=charge_was_taken)   # how to ask after a crash
async def charge_card(customer_id: str, amount: float) -> dict:
    key = specunode.current_idempotency_key()                 # the same key on every retry
    return {"charge_id": await payments.charge(customer_id, amount, request_key=key)}

@specunode.node()
async def bill(session: specunode.RunSession) -> specunode.Decision:
    charge = await session.call_tool("charge_card", {"customer_id": "cus-1", "amount": 25.0})
    session.state["billed"] = True
    return specunode.FreeText.of("billed")

runtime = specunode.Runtime(specunode.graph([bill], route), tools=[charge_card])
await runtime.run({"customer_id": "cus-1"}, run_id="bill-cus-1")
# the process dies with the charge made and its reply lost -- then:
await runtime.resume("bill-cus-1")
```

```
billing cus-1 ...
  the process died: killed after the charge went through, before its reply came back
resuming from the journal ...
  asked the payments API about request fe58acad3144...: ch_1
  finished: True
charges made: 1, receipts sent: 1
```

The whole example is [examples/quickstart.py](https://github.com/Poojan6216/specunode/blob/main/examples/quickstart.py), and a test runs it.

## And faster, when the model is the slow part

Against `claude-sonnet-5`, running every call a reply asks for together and handing the results
back at once took an on-call task from 11 replies to 5: 35.4% less time and, with prompt caching,
77.7% less cost, with every run correct. Three independent checks side by side took 60.2% less
time than one after another. Nothing reached the world from a branch that never retired.
[Details](https://github.com/Poojan6216/specunode/blob/main/RESULTS.md#fewer-replies-caching-and-parallel-nodes-against-a-real-model).

What the safety costs is measured too. Against the fastest loop you would write by hand --
every call of a reply at once, no journal, nothing a crash could be resumed from -- the runtime
is 0.6% slower with instant tools and 3.3% slower with 300 ms tools on the same alert; a fan-out
of read-only checks pays 13.1% with slow tools, one round of re-checking its reads.
[Details](https://github.com/Poojan6216/specunode/blob/main/RESULTS.md#when-the-model-is-the-slow-part-fewer-replies-and-replies-side-by-side).

## What did not work

It started as CPU-style speculation: guess the model's next tool call and run it before the
model finishes, holding its writes in a store buffer until the guess is confirmed. The safety
half works -- a wrong guess never reaches the world -- but measured honestly, **guessing buys
almost no time**: a guess can run at most one block ahead of the model, and on real agent
traces the predictors are rarely right. The measurements are below, negative half first, and
they are why the project's claim is now safety plus the model-bound speedups above, not
speculation.

## How it works

A processor solved "waiting on one thing at a time" in the 1960s. It guesses which way a branch
goes, runs the next instructions early, and keeps any *writes* in a holding area called a store
buffer. Only once the guess is confirmed do the writes leave. If the guess was wrong, the buffer
is discarded and nothing outside the processor ever saw it.

SpecuNode applies that to agents. Reads run early. Writes go into a store buffer tied to the
branch that produced them. When the model's real decision arrives, it is compared to the
guess — exactly, not approximately. Match: the buffer drains to the world. Mismatch: the
buffer is discarded, unsent. **The model's real decision is the only thing that can ever release
a write** -- and the same journal that holds it is what makes a crash safe to resume.

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
[`bench/results/demo_leak.json`](https://github.com/Poojan6216/specunode/blob/main/bench/results/demo_leak.json) and a test asserts the file still
matches a fresh run.

---

## What issuing a read early buys, past a staged write

```
python bench/demo.py --demo past-write
```

One ops workload — `get_pipeline_status` → turn 1 → `restart_job` (write) + `fetch_runbook`
(read, independent of the write) → turn 2 → `post_summary` (write) — under three execution
styles, timed at the tool boundary. Turn 1 emits the write first, so a runtime that stops at
the first tool with side effects has nothing before it to run ahead into.

The demo prints measured durations, so the figures differ between machines and runs and none
of them is reproduced here. What does not vary is the shape, and the demo asserts it:

- **`readonly-spec` matches `sequential`.** Both baselines take the model's turn and then
  make its calls in order, and turn 1 emits the write first, so nothing runs ahead of it.
- **`specunode` is faster by the part of `fetch_runbook` that overlapped the staged write**,
  and the demo prints that overlap as a measured number rather than inferring it from the
  wall-clock difference.
- **No guess is made anywhere in this demo.** The runtime arm has no predictor. The saving is
  tier-0 early issue — the read is issued the moment its block parses, instead of after the
  write before it — and the store buffer is what makes that safe: it holds the model's own
  write until the turn that asked for it is durable, while the independent read overtakes it.
  The `readonly-spec` arm models a runtime that runs a turn's calls in order after the model
  finishes; it is not this runtime's reads-only policy, under which the saving is the same.
- **All three change the world identically** — same tools, same canonical arguments, same
  order. The demo prints the digest and fails if they diverge.
- **Model turn 2 is not hidden.** It needs `restart_job`'s real result in its prompt, so it
  waits for the drain. The demo says so in its own output.

The saving is one read's latency. That is what issuing a read early buys on this shape, and
the demo is written to make that hard to mistake for more — including for the speculation this
project is named after, which the benchmarks below find adds nothing on top of it.

---

## The measured result, negative half first

Measured on 300 real OpenHands trajectories from
`nebius/SWE-rebench-openhands-trajectories` — 19,484 tool calls. Full numbers with confidence
intervals in [`RESULTS.md`](https://github.com/Poojan6216/specunode/blob/main/RESULTS.md). The derived corpus is redistributed under the dataset's
CC BY 4.0 licence, with what was changed listed in [`bench/corpus/NOTICE.md`](https://github.com/Poojan6216/specunode/blob/main/bench/corpus/NOTICE.md).

```
python bench/corpus/fetch.py
python bench/offline/run_opportunity.py --out bench/results/opportunity.json
```

**On this corpus, running ahead *past* a write buys nothing at all.** The measured span is
0.0000, with a bootstrap interval that does not move off zero.

That is not an implementation limitation. Every tool call in these trajectories opens a new
model turn. A staged write does not block the next tool *call*, but it does block the next
model *turn*, because that turn would have to contain a placeholder where the real result
belongs. When there is never a second call inside a turn, the store buffer has nothing to run
ahead into. 95.5% of this corpus is that shape.

**What the store buffer does unlock is a different quantity, and it is not zero.** PASTE
excludes a tool with side effects from speculation entirely, so it can speculate on 4.5% of
steps — the reads. SpecuNode stages a write instead of refusing it, so a *predicted* write can
be run ahead like any other call and discarded if the model decides otherwise. That covers the
other 95.5%.

That is an upper bound on opportunity, not a speedup. It is realisable only where the predictor is
right, and **how often the predictor is right is now measured, and it is almost never.** *Signature*
accuracy — the right tool with the right argument *names* — is 53.4% top-1 over all 300
trajectories. The runtime releases a write only on exact canonical equality of argument **values**,
and graded by that gate on the same steps, the tier-1 acceptance rate is **0.0002** with guesses
carried across model turns (3 of 19,184 graded steps) and **0.0000** under the policy the runtime
actually runs, where every guess is squashed at the turn boundary because every call in this corpus
opens a new turn. Nor is that a flaw this predictor could fix: every argument value of a call has
already appeared in an earlier call at only 9.8% of steps, which is the ceiling for tier 1 *as
graded here* — a predictor copying values out of earlier calls, with no tool results to draw on,
because this corpus keeps none — and half this corpus is `execute_bash` with a free-form command
string. An argument of 20.5% of steps did come from a prior result, so the bound for a drafter that
can read those is higher and is not measured. Anything above the ceiling has to come from a
predictor that generates values — a draft model. That is now measured too: `claude-haiku-4-5`,
shown the last 12 calls with their real argument values and graded by the same gate with its
guesses carried across turns, is right **6.9%** of the time (95% interval 5.4% to 8.6%, over
1000 sampled steps), where tier 1 was right 0.0002 of the time under the same rule. Under the
rule the runtime actually runs it is zero for any predictor, for the reason above.

**Against a real model, the only wall-clock saving is issuing reads early, and none of it is from
guessing.** On the one sample app that hands its model turn to the runtime, with every tool
slowed to 500 ms, issuing each read the moment its block parses saves 11.8% [7.8%, 15.5%]; with
2000 ms tools, 16.7% [15.1%, 18.2%]. Guessing added nothing any interval resolves, and a guesser
of controlled accuracy is measured to be worth at most 4.8% even when it is always right, because
a guess can run at most one block ahead of the model.

**When the model is the slow part, what helps is fewer model replies and replies side by side,
and both are measured against a real model.** On an on-call task against `claude-sonnet-5`,
running every call a reply asks for and handing the results back together took the job from 11
replies to 5: 35.4% less time and, with prompt caching, 77.7% less cost, with every run correct.
Three independent checks run side by side took 60.2% less time than one after another. Caching
alone cut the bill by 72.7% and did not change the time at this prompt size. Nothing reached the
world from a branch that never retired. Details in [RESULTS.md](https://github.com/Poojan6216/specunode/blob/main/RESULTS.md).

---

## Killed mid-run, resumed, and refused when the question changes

```
python bench/demo.py --demo replay
```

The same ops run, `SIGKILL`ed at a point measured to land inside its own work — not inside the
interpreter startup that dominates a subprocess's lifetime, which is how a kill demo ends up
killing nothing and reporting success. Then resumed from the journal, then replayed twice.

The demo asserts, and prints, that:

- the resumed run's effects are a **prefix of the uninterrupted run's, in order** — it can fall
  short, and can never do something the clean run did not
- **no effect was applied twice** -- a tool declared idempotent may be handed its key again
  when a crash lost the reply to its first delivery, and the demo says when it was
- the **journal's hash chain still verifies** after a process died mid-append
- replaying with a different system prompt is **refused at the first turn that would differ**,
  with the step index and a field-level diff of the request — not a silent re-run down a
  trajectory the recorded run never took
- replaying with speculation disabled completes and prints its ledger digest

It ends with the run's **effect ledger**, which is the artifact this project actually produces:
every effect that reached the world, the node and program position that authorised it, and the
idempotency token the tool was handed.

Where a resume falls short rather than completing, it is because the process died between a
request reaching the world and its acknowledgement being recorded. Nobody can tell afterwards
whether it took effect. A tool that declared a repeat harmless is redelivered; one that did not
is dead-lettered for a human. The demo says which happened.

---

## Status

v0.1.0, and honest about where it is. Working today:

- the canonical form, the hash-chained journal, replay and crash recovery
- effect classes, the store buffer, at-least-once dispatch with deterministic idempotency keys
- the sequential scheduler, the LangGraph integration, the plain-Python API, `resume`
- tier-0 early issue, the tier-1 pattern index, and a tier-2 draft model behind an extra —
  "working" here means they run and are measured, not that they pay: tier 1's measured
  acceptance on the corpus above is 0.0000 within a turn and 0.0002 across, and tier 2's is
  0.069
- agent loops that hand every result of a reply back in one message
  (`specunode.core.loop.agent_loop`), prompt caching on by default, and parallel nodes — a
  router may name several nodes at once, and they run side by side and retire in the order named
- the MCP proxy's rules, the offline benchmark, the overhead benchmark, the adversarial suite
- the three invariant tests below

Known gaps, stated rather than left to be discovered:

- **Guessing has not produced a wall-clock saving against a real model.** Issuing reads early
  has; guessing on top of it added nothing any interval resolves.
- **The real-model gains for fewer replies and parallel nodes come from one task.** One on-call
  alert and one three-way check, against one model. How much another task gains depends on how
  many of its calls are independent of each other; calls that depend on one another still need
  a reply each.
- One sample workload, not three. The invariant tests hold on it and on tiers 0 and 1.

`BUILD_SPEC.md`'s Final Report lists every one of these with the reason it is open. Every
subcommand of the CLI is documented in [docs/cli.md](https://github.com/Poojan6216/specunode/blob/main/docs/cli.md).

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
says it sent.

Be clear about what it does **not** prove. Every workload has one model decision point, so each
run records one request, sent before anything forks — the two arms are compared byte for byte
and must match, which catches a runtime perturbing a prompt by speculating near it. The harder
claim, that a request sent *by a speculative branch* is one the sequential run could send, is
never evaluated, because no shipped path produces one: a speculative child runs a single tool
call and never opens a turn. The test asserts that too, so if speculation ever crosses a model
turn it fails rather than passing vacuously. The retirement-time rebuild the rule describes is
not implemented; a branch that did send a request while guessing is refused at retirement
instead. See [`docs/adapters.md`](https://github.com/Poojan6216/specunode/blob/main/docs/adapters.md).

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

Ten strategies are run; eight of them defeat the runtime, and those eight are listed below with
measured rates. The other two are held — a poisoned pattern index forks and stages writes that
are all discarded, costs no model tokens, and closes the alpha gate once its window has filled
with misses, but cannot put an effect in the world; and a replay after a changed prompt or tool
list diverges at the very first step rather than continuing down a trajectory the recorded run
never took.

This section is generated from `bench/adversarial/run_attacks.py` rather than written from
memory, and a test fails the build if any of the eight stops defeating the runtime — either a
hole was genuinely closed and this list should shrink, or the attack stopped exercising what it
claims to. Both need a look.

```
python bench/adversarial/run_attacks.py --all
```

[`docs/limitations.md`](https://github.com/Poojan6216/specunode/blob/main/docs/limitations.md) explains what each one costs.

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

Apache-2.0. See [LICENSE](https://github.com/Poojan6216/specunode/blob/main/LICENSE).
