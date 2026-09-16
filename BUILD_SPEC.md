# BUILD SPEC — SpecuNode
### Agents wait for the world one tool call at a time. SpecuNode lets an agent graph run ahead of its own model — reads execute early, writes wait in a store buffer, and nothing reaches the world until the model has actually decided it. Every effect that leaves the runtime traces to a committed decision, and the whole run replays.

---

## HOW TO USE THIS FILE (read first, agent)

You are building this project end to end. Work through the phases in order. Do not skip ahead, do not ask which phase to start with, do not stop to ask for approval between tasks.

**Your working loop:**

1. Read the phase you're on. Read every task in it.
2. Implement each task in order.
3. After each task, run its **Verify** step. If it fails, fix it before moving on.
4. Tick the checkbox in the task list and append one line to the **Progress Log** at the bottom of this file.
5. When a phase's **Phase Gate** passes, move to the next phase.
6. When all phases are done, write the **Final Report** section and stop.

**Rules for the whole build:**

- Do not ask for permission to continue. Keep going until every phase is complete.
- If a decision isn't specified here, pick the simplest option that satisfies the Hard Rules, and log the decision in the Progress Log. Don't stall on it.
- If you get genuinely blocked (an API key is missing, a model won't download, a public trace corpus is gone), write the blocker in the Progress Log, implement the closest working alternative that still satisfies the Hard Rules, and continue. Don't halt the whole build over one task.
- Commit after each completed task. Conventional Commits.
- Every phase must leave the package in a working, installable state. Never end a phase with a broken build.
- **Numbers in this file that come from published research are marked `[cited]`. Never invent a number. Any figure that appears in the README or `RESULTS.md` must come from a run you actually executed, and the command that produced it must be in the repo.** The proposal this spec was derived from contained latency and safety figures that were never measured by anyone. None of them appear here and none may appear in any output.
- **Correctness claims are held to the same standard as benchmark numbers.** If the README says "no effect from a rejected branch ever reaches the world", there must be a test in the repo that runs a fake world with every mutation tagged by branch, injects faults, and asserts it. If the runtime cannot guarantee a property for a class of tool, the docs say so and the receipt for that run says so.

**Do not build:** a SaaS control plane, a hosted service, accounts, telemetry, a "Pro" tier, a Kubernetes operator, a new agent framework, a new LLM orchestration DSL, a vector database, a model-serving system, a CRDT library, an LLM-based judge that decides whether a speculation "matched", an LLM-based context summariser in the runtime path, or a general workflow engine competing with Temporal. If you find yourself writing a login page, a prompt that asks a model "are these two tool calls the same?", or a merge function that reconciles two speculative branches into one, stop — you've misread the spec.

---

## 1. What this is

An agent graph runs in a strictly serial loop: the model thinks, emits a tool call, waits for the tool, thinks again. For a ten-step task where each step is a two-second model turn and a one-second tool, that is thirty seconds of wall clock, most of it spent waiting on one thing at a time.

Two families of work attack this from opposite ends, and each stops exactly where the other starts.

**Speculative tool execution** (PASTE, Microsoft Research, March 2026 — see §3) predicts the next tool call from patterns in prior trajectories and runs it while the model is still generating. It reports a 48.5% reduction in average task completion time `[cited]`. But it can only speculate on tools that have no side effects. The moment the predicted next call is `create_ticket` or `send_email` or `UPDATE accounts`, speculation stops, because running it on a guess would corrupt the world. In a workload where every third tool call writes something, that is a hard ceiling on how far ahead the agent can run.

**Transactional gating** (SagaLLM, ATP, SCOPEGATE — see §3) treats every model-emitted tool call as an untrusted proposal that a deterministic gate must admit before it takes effect. That makes writes safe. But the gate is a checkpoint on the serial path; it does not let anything run early.

**Durable execution** (Temporal, DBOS, Restate, LangGraph checkpointers) journals every model output and tool result so a crashed run resumes without re-asking the model. That makes replay deterministic. But it has no concept of a tool call that has executed and is not yet admitted.

SpecuNode is the combination, and the combination is the contribution:

> An agent graph is executed the way an out-of-order CPU executes instructions. Reads issue early. Writes go into a **store buffer** keyed to the speculative branch that produced them. The **target model's actual output** is the branch-resolution signal: when it matches the speculation, the branch **retires** and the store buffer drains to the world with deterministic idempotency keys; when it does not, the branch is **squashed** and its buffer is discarded, never dispatched. Every model output and tool result is **journaled** before use, so the run replays exactly, speculation on or off. A **hazard** — a downstream call that depends on the value a staged write would have returned, a tool whose effect class is undeclared, a node that emits free text rather than a structured decision — **stalls** the branch to sequential execution. It never guesses.

**What speculation can and cannot hide.** A staged write returns a placeholder handle, and a placeholder must never appear in a prompt: the sequential run would have shown the model the real tool result, and a model conditioned on a placeholder is making its decision on a different premise. So a staged write is a barrier for the branch's *next model call* (hazard `MODEL_TURN_AFTER_STAGED_WRITE`). What speculation past a write buys is that independent tool calls — reads, and further staged writes — run concurrently with the in-flight target turn and the drain, so their latency is hidden. Model latency itself is hidden only in read-only stretches, where the branch's prompt is byte-identical to the sequential prompt and the branch may run the next model call speculatively. A workload of strict `model → write → model(reads the write's result)` chains gets no wall-clock gain from past-write speculation, by design, and the bench reports how much of each workload is that shape.

The claim is narrower than "agents run 3× faster". The claim is: **speculation is now safe past a write, the runtime can say exactly how much speculation a given workload actually exposes, and the effect log of a speculative run is byte-identical to the effect log of the sequential run.** The benchmark measures all three, including the workloads where speculation buys nothing — those are results too.

**What a "branch" is here.** Not a git branch and not a conditional edge. A branch is a speculative continuation of the graph from a *predicted* decision at step *i*: everything the runtime does on the assumption that the model will decide *ŷᵢ*, before the model has actually produced *yᵢ*. Branches fork state and context copy-on-write, share nothing with siblings, and end in exactly one of retired, squashed, or stalled.

**Who this is for.** A developer with a LangGraph app, a plain-Python agent loop, or an MCP client, whose tools include writes, who wants the latency of running ahead without the exposure of running ahead. They wrap the graph or point their MCP client at the proxy, declare the effect class of each tool (or rely on MCP tool annotations), and get a run that is faster where the workload allows, identical in effect where it does not, and replayable either way.

---

## 2. The promise: three demos

Everything below has to work from the published package. These are acceptance tests, not marketing.

### Demo 1 — speculation without a store buffer double-charges

`bench/demo.py --demo leak`

A support agent: `lookup_customer` → decide → `charge_card(amount)` → `send_receipt`. Two runtimes execute the same 50 journaled model transcripts. In 10 of the 50, the model's real decision differs from what a pattern drafter predicted.

The naïve runtime (`bench/baselines.py:B_naive_parallel`) runs the predicted `charge_card` early and discards the branch on mismatch. The fake world (`specunode.testing.World`) records every mutation with the branch id that issued it. The output is a table:

```
runtime              runs  mispredictions  charges reaching world  charges from squashed branches
naive-parallel         50              10                      60                              10
specunode              50              10                      50                               0
sequential             50               0                      50                               0
```

The numbers must come from the run. The point of the demo is the last column and the fact that the second and third rows are equal.

### Demo 2 — running ahead past a write

`bench/demo.py --demo past-write`

An ops agent: `get_pipeline_status` (read) → decide → `restart_job(id)` (write) → `fetch_runbook(section)` (read, independent of the write) → `post_summary(channel)` (write). Under PASTE-style speculation the runtime must stop before `restart_job`. Under SpecuNode, `restart_job` is staged, `fetch_runbook` runs speculatively while the target's turn 1 is still streaming and the drain is in flight, and both writes retire together when the model confirms. Model turn 2 is *not* hidden: it needs `restart_job`'s real result in its prompt, so it waits for the drain. The demo says so on its output. The demo prints a timeline:

```
step                   sequential   read-only speculation   specunode
get_pipeline_status        0.9s         0.9s                  0.9s
model turn 1               2.1s         2.1s                  2.1s
restart_job (write)        0.7s         0.7s     [barrier]    staged, 0.0s on path
fetch_runbook (read)       1.4s         1.4s                  hidden behind turn 1 + drain
model turn 2               2.0s         2.0s                  2.0s  (not hidden: needs write result)
post_summary (write)       0.5s         0.5s                  retired with restart_job
-----------------------------------------------------------------------------
wall clock                 7.6s         7.6s                  measured
effects reaching world     2            2                     2 (identical ledger)
```

The figures in the table above are illustrative placeholders for the *shape*; the demo prints measured values only.

### Demo 3 — the honest one, which is the point of the project

`bench/demo.py --demo replay`

The same ops run is killed with SIGKILL at a random point mid-branch. `specunode resume <run_id>` continues from the journal. The effect ledger after resume is byte-identical to the ledger of the uninterrupted run. Then the run is replayed with a different system prompt: `specunode replay <run_id>` refuses at the first journaled model output that the new prompt would not have produced, with the step index and the diff, instead of silently re-running a divergent trajectory. Finally, the run is replayed with speculation disabled: identical ledger.

The demo ends with the run's **effect ledger**, which is the artifact this project produces:

```
EFFECT LEDGER  run 01K5…  speculation=on  drafter=pattern  target=<model>
 #  effect                         key            branch  decided-by(step)  status
 1  restart_job {id:"etl-7"}       b2f9…          br-03   step 4 (retired)  DISPATCHED ack=1
 2  post_summary {ch:"#ops"}       11ac…          br-03   step 4 (retired)  DISPATCHED ack=1
squashed branches: 2   staged effects discarded: 3   stalls: 1 (free-text node "draft_reply")
reads validated at retirement: 4/4 fresh   wasted tokens: 1,842   alpha (window 20): 0.70
```

Anything can print "done". This says which model decision authorised each effect, what was thrown away, where it had to fall back to sequential, and how much the speculation cost.

---

## 3. Prior art — who is already here, and exactly where they stop

Read this before designing anything. Credit every project below in the README by name. The proposal this spec replaced claimed several of these did not exist or did not solve their problem; they do, and the README must not repeat that mistake.

**PASTE — *Act While Thinking: Accelerating LLM Agents via Pattern-Aware Speculative Tool Execution*** (Sui et al., Microsoft Research, arXiv 2603.18897, March 2026). Characterises agent traces from SWE-bench, MetaGPT and OpenHands and finds strong temporal locality in tool sequences ("strong chains", "refinement loops") and predictable data-flow between tool arguments. Runs as a tool-serving proxy that mines patterns and speculatively pre-executes predicted calls. Reports 48.5% lower average task completion time and 1.8× tool throughput `[cited]`. Has an operator policy per tool for whether speculation is permitted and how side effects are handled.
*Where it stops:* side effects are handled by *policy exclusion* — a tool with side effects is not speculated. There is no mechanism to hold a write, so speculation ends at the first mutating call. No journal, no replay. **This is the closest prior work and the pattern-mining drafter in §7 is a re-implementation of its idea, credited as theirs.** Follow-ups: *B-PASTE* (arXiv 2604.16469) and *Speculate with Memory* (arXiv 2607.12236) refine the predictor; neither adds a write path.

**Claude Code's streaming tool executor** (described in *Dive into Claude Code*, arXiv 2604.14228, and the *Claude Code from Source* study). Starts each tool the instant its `tool_use` block is fully parsed from the stream, before the response finishes; concurrent for read-only tools, serial for writes; preserves result order.
*Where it stops:* it is early-issue, not prediction — it only runs what the model has already emitted. Writes are serialised. No cross-step speculation. This is exactly SpecuNode's Tier-0 drafter, and it is credited as such.

**langchain-nvidia-langgraph** (LangChain + NVIDIA, 2026). Compile-time parallelisation of independent nodes plus "speculative execution" that runs both branches of a conditional edge and discards the loser.
*Where it stops — in its own docs:* speculation is over static routing edges only, and the speculative mode does not support checkpointers, streaming, interrupts or human-in-the-loop. No side-effect handling: both branches' tools run for real. That is the exact failure Demo 1 exhibits.

**ToolAhead** (MCP server). Prefetches read tool results (file reads, searches) before a coding agent asks for them and replays them on request.
*Where it stops:* reads only; writes execute normally and reset prediction. Single-workspace coding tools. No journal.

**SagaLLM** (Chang & Geng, VLDB 2025, arXiv 2503.11951). Wraps multi-agent planning in the saga pattern with persistent memory, compensating transactions and independent validation agents. Relaxes ACID; ensures workflow-wide recoverability through checkpoints and compensation.
*Where it stops:* the saga coordinator uses an LLM for state tracking and recovery orchestration, which SpecuNode forbids in the control path (Hard Rule 1). No speculation. Its compensation vocabulary (each transaction declares its compensator) is adopted for the `COMPENSABLE` effect class, credited.

**ATP / Mnemosyne — *Agentic Transaction Processing*** (arXiv 2607.00269, July 2026). A transaction model where a generated action holds no authority until a deterministic gate admits it against an effective-state witness; committed-state correctness is proven independent of the proposer. Explicitly positioned as complementary to SagaLLM: one disciplines the proposer, the other governs the committer.
*Where it stops:* admission happens on the serial path. It has no concept of an action that executed early and is awaiting admission, which is what a store buffer is. SpecuNode's retirement stage is an ATP-style gate; the store buffer is what sits in front of it.

**SCOPEGATE — *Capability Gates Are Not Authorization*** (arXiv 2606.28679). Audits LangChain/LangGraph, LlamaIndex and the Stripe Agent Toolkit and finds none re-authorises each model-emitted call with its concrete argument values by default; proposes a five-stage deterministic PDP/PEP with scope, authorization, money ceiling, idempotency and default-deny.
*Where it stops:* authorization only, no execution model. Its point — that tool exposure is not per-call authority — is why SpecuNode's effect classes are declared out of band and never inferred from the model (Hard Rule 2).

**Durable execution** — Temporal (with Pydantic AI and LangGraph integrations), DBOS, Restate, Inngest, Azure Durable Task; LangGraph's own checkpointers. All journal non-deterministic results (model outputs, tool results) so that replay reuses recorded values rather than re-running; idempotency keys derived from (workflow id, step id).
*Where they stop:* a step is a step. Nothing runs before its predecessor completes, there is no store buffer, and there is no speculative branch to squash. SpecuNode's journal follows the same discipline — recorded first time, reused on replay — and is designed so a run can later be hosted *inside* one of these engines (each retired branch is a deterministic step). The README says plainly that SpecuNode is not a replacement for a durable-execution platform.

**Out-of-order CPU execution** — Tomasulo (1967), store buffers, branch prediction, squash-on-mispredict, retirement in program order. Not LLM work, but the model this design copies deliberately: speculative loads issue, speculative stores wait in a buffer, the branch resolves, the buffer retires or is squashed. The vocabulary of this spec (retire, squash, hazard, stall, store buffer) is taken from there and the README explains the analogy in one paragraph.

**SpecuNode's actual contribution, in three sentences.** It is the first runtime that lets an agent graph speculate *past* a mutating tool call, by holding the call's effect in a branch-scoped store buffer that drains only when the target model's real decision confirms the branch, and is discarded — never dispatched — when it does not. It journals every model output and tool result so that the effect ledger of a speculative run is provably identical to that of the sequential run, and both replay. And it measures, per workload, how much speculation the dependency structure actually exposes past writes, publishing the cases where the answer is "little" with the same prominence as the cases where it is "a lot".

**Naming.** *SpecuNode*: speculation at the granularity of a graph node. The PyPI name `specunode` was free on 15 September 2026; the CLI is `specunode`.

---

## 4. Hard rules — never violate these

These are not preferences. Breaking any one makes the runtime a liability to whoever runs it.

1. **No LLM in the control path.** Which tool calls run, when, and whether a speculation matched are decided by deterministic code. The only models in the runtime are the *target model* (whose output is the ground truth for branch resolution) and the *draft model* (an optional Tier-2 predictor whose output is only ever a guess). There is no prompt anywhere in `src/specunode/core/`, `src/specunode/buffer/`, `src/specunode/verify/`, or `src/specunode/journal/`. A test greps for `messages=` and `prompt` in those packages and fails the build if found.
2. **Effect classes are declared, never inferred.** Every tool carries exactly one of `READ`, `WRITE`, `COMPENSABLE(compensator)`, `IRREVERSIBLE`, declared by the developer in code or by MCP tool annotations. A tool with no declaration is `WRITE`. No heuristic on the tool's name, description or arguments ever assigns a class. No model ever assigns a class. A tool whose upstream enqueues, schedules or triggers work asynchronously is a `WRITE` regardless of its synchronous response or HTTP verb; the docs say this on the first page.
3. **Nothing reaches the world from an unretired branch.** A `WRITE`, `COMPENSABLE` or `IRREVERSIBLE` effect is dispatched only by `retire()`, only for a branch in state `CONFIRMED`, only after the journal entry that confirmed it is durable. Speculative execution of a `READ` is permitted; speculative execution of anything else is a bug, and the leak test (Rule 12) catches it.
4. **Branch resolution is exact canonical equality.** A speculation is confirmed iff `canonical(ŷ) == canonical(y)` where both are `Decision` values (a tool call name plus canonicalised arguments, a route label, or a structured object). Never semantic similarity, never fuzzy argument matching, never an LLM judge. A node whose output is free text is a speculation barrier: nothing speculates on it.
5. **Journal before use.** Every target-model output and every tool result is written to the journal and fsynced before the runtime acts on it. Replay never calls a model; it reads the journal. A replay that reaches a step whose journaled input does not match what the current code would send raises `ReplayDivergence` with the step and the diff. It never re-asks the model and continues.
6. **Branch isolation.** A branch reads only (a) the committed state as of its fork point and (b) its own staged effects via declared store-buffer forwarding. It never observes a sibling's staged writes, state deltas, or context. State is forked copy-on-write per branch; the retiring branch's delta is applied to committed state; a squashed branch's delta is dropped. There is no merge of two speculative branches, ever.
7. **Hazards stall, they never guess.** If a speculative call's arguments depend on the *return value* of a staged write, or the tool is undeclared, or the node emits free text, or the read's key is one a staged write in the same branch touches and the tool declares no forwarding, or the branch's next model call would contain a placeholder in its prompt, the branch enters `STALLED` and the runtime proceeds sequentially from the last confirmed decision. `STALLED` is recorded with its reason in the ledger.
8. **Idempotency keys are deterministic and stable.** `key = blake2b(run_id ‖ branch_lineage ‖ node_id ‖ step_index ‖ tool_name ‖ canonical(args))`, where `step_index` is a per-run monotonically increasing counter that is itself journaled (so it survives resume), and `node_id` disambiguates loop iterations that revisit a node. The same logical effect gets the same key on retry, resume and replay. Dispatch is at-least-once with dedupe on the key. The docs say "at-least-once with idempotent dedupe" and never say "exactly-once" for a tool that has not declared itself idempotent.
9. **Speculation preserves semantics.** For any journaled run, the effect ledger with speculation enabled equals the effect ledger with speculation disabled, ignoring timestamps and keys' branch components. This is the equivalence test (Phase 5). It runs on every workload in CI. A mismatch is a build failure, not a flaky test.
10. **Wasted work is bounded and measured.** Every run enforces `max_inflight_branches`, `max_speculation_depth` and `max_wasted_tokens`. The drafter's rolling acceptance rate α is tracked over a fixed window; when it falls below the measured break-even for the workload, speculation is disabled for that run by deterministic policy, and the ledger says so. Speculation must never make a run *slower* than sequential by more than the journaling overhead measured in Phase 6.
11. **No telemetry, no hosted components, no accounts.** The runtime makes no network calls the developer did not configure. Model calls go to the endpoint the developer named. Model downloads for the optional local drafter happen once, explicitly, at install.
12. **Never report a number you did not measure.** Every figure in `README.md`, `RESULTS.md` and the report traces to a committed command and a committed JSON file. `RESULTS.md` is generated. A traceability check (`bench/check_numbers.py`) fails CI if a number in the README appears in no results file. A vocabulary check fails CI if "ACID", "exactly-once", "guaranteed", "zero-latency", "eliminates", "100% safe" or "context rot" appears in any output or doc without an adjacent qualifier that names the condition.
13. **Context identity.** A speculative branch may send a model request only if the prompt it would send is one the sequential run could send: no placeholder anywhere in it, tool results assembled in program order (not completion order), and no message from any squashed sibling. When the canonical run reaches that step, the prompt it constructs from real results must be byte-identical to the one the branch sent; a difference is a `ContextDivergence` fault that squashes the branch before its model output is used. This is the live counterpart of Rule 9: Rule 9 proves the effect plumbing on journaled outputs; Rule 13 proves the model was never asked a different question.

---

## 5. Locked technical decisions

Don't re-litigate these.

| Decision | Choice |
|---|---|
| Language | Python 3.11+ (floor enforced in CI matrix 3.11/3.12/3.13; do not raise it), `from __future__ import annotations`, full type hints |
| Typing | mypy `--strict` on `src/`, no `Any` in public signatures |
| Packaging | `uv`, `pyproject.toml`, hatchling backend |
| Lint/format | `ruff` (lint + format), line length 100 |
| Concurrency | `asyncio` with `TaskGroup`; every branch is a task; cancellation is the squash primitive. No threads except inside tool adapters that need them |
| Tests | `pytest`, `pytest-asyncio`, `hypothesis` for the store buffer, canonicaliser and journal |
| Journal | SQLite (`sqlite3` stdlib, WAL mode, `synchronous=FULL`), append-only table, default `./.specunode/journal.db`. Postgres via `psycopg` 3 optional, same DDL. One fsync per entry, never batched: ~100 fsyncs on a 30-step run is tens of milliseconds against multi-second model turns, and batching would open a crash window for nothing |
| Canonical form | `canonical(obj)` = JSON with sorted keys, no whitespace, NFC-normalised strings, floats via `repr`, `-0.0 → 0.0`, NaN rejected. Hash = blake2b-256 of the canonical bytes |
| Identifiers | ULID for runs, branches, steps, effects |
| Decision types | `ToolCall(name, args)`, `Route(label)`, `Structured(schema_id, value)`, `FreeText(hash)` — the last is a barrier |
| Effect classes | `READ`, `WRITE`, `COMPENSABLE(compensator: str)`, `IRREVERSIBLE`. Tool-level flags: `idempotent: bool`, `forward_keys: Callable[[args], set[str]] | None` (resource keys a call touches, enabling read-after-staged-write forwarding), `witness: bool` (returns a version/ETag with reads) |
| MCP mapping | `readOnlyHint=true` → `READ`; `destructiveHint=false, idempotentHint=true` → `WRITE(idempotent=True)`; `destructiveHint=true` → `IRREVERSIBLE`; annotations absent → `WRITE(idempotent=False)`. A per-tool override table in `specunode.yaml` takes precedence |
| Drafters | **T0 early-issue**: parse `tool_use` blocks from the target stream as they complete (Claude Code's pattern, credited). **T1 pattern index**: order-*k* (k ≤ 3) Markov over tool signatures with argument templates that reference prior tool outputs by JSONPath (PASTE's idea, credited); trained per workload from the journal; on disk as JSON. **T2 draft model**: optional extra; any model behind the same `ModelClient` protocol, default `claude-haiku-4-5` when the target is a larger Claude model, or a local `Qwen/Qwen2.5-0.5B-Instruct` via the `[local-draft]` extra |
| Target model (bench) | `claude-sonnet-5` via the Anthropic API for online workloads; the fake `ReplayModel` for offline and CI. A bench-spend cap `SPECUNODE_BENCH_BUDGET_USD` (default 25) halts the online bench when exceeded and the report says so |
| State | Committed state is a JSON-serialisable dict. Branch state is a copy-on-write fork (`copy.deepcopy` at fork; delta computed as RFC 6902 JSON Patch at retirement). Per-key reducers declared in config (`append`, `last_write`, `max`, custom callable) apply only when *sequential* nodes write the same key; two speculative siblings never merge |
| Context | Per-branch message list forked at the branch point. Tool results are appended in *program order* (the order the model requested them), never completion order — Claude Code's ordering rule, credited. A placeholder never enters a message. The retiring branch's context becomes canonical. Squashed branches' messages are discarded entirely. No pruning, no summarisation, no distillation in the runtime |
| Framework integrations | `specunode.integrations.langgraph`: wraps a compiled `StateGraph` (LangGraph ≥ 0.6) by substituting its node runner and tool executor; the graph definition is unchanged. `specunode.integrations.plain`: a decorator-based API for hand-written loops. Both share one core |
| MCP proxy | `specunode mcp-proxy --upstream "<cmd>"`, stdio; forwards `tools/list` with annotations, intercepts `tools/call`; protocol versions `2025-11-25` and `2026-07-28` via the official `mcp` SDK |
| Fake world | `specunode.testing.World`: in-memory tables + HTTP-like endpoints, every mutation recorded with `(branch_id, effect_key, wall_time)`; fault injection: `partition(at=…)`, `timeout`, `duplicate_delivery`, `slow(read|write, ms)` |
| Offline trace corpus | `nebius/SWE-rebench-openhands-trajectories` (Hugging Face, CC-BY-4.0, 67,074 OpenHands trajectories with per-step `tool_calls` name + JSON arguments and observations; verified available 2026-09-15). A seeded 2,000-trajectory sample by manifest; the manifest and the per-tool effect-class table are committed, the raw data is fetched by `bench/corpus/fetch.py`. Fallback (Decision Gate D1): traces generated by running the three sample apps with the target model and committed under `bench/corpus/` |
| Sample apps | `examples/support_agent`, `examples/ops_agent`, `examples/research_agent` — each a LangGraph graph with 4–8 tools of mixed effect class against `World` |
| Plots | `matplotlib` only. Committed as PNG + the script that made them |
| License | Apache-2.0 |
| Package name | `specunode` (PyPI, free as of 2026-09-15). CLI: `specunode` |
| Config | `./specunode.yaml`, then `$XDG_CONFIG_HOME/specunode/config.yaml`. Pydantic v2, schema versioned |

**Why a store buffer and not a saga:** a saga runs a write and undoes it if something later fails. Compensation is a *second* effect that reaches the world, and for many tools (send email, charge card, POST webhook) it is imperfect or impossible. A store buffer never lets the first effect out. Compensation is kept — as the `COMPENSABLE` class for retired effects that later need undoing — but it is not the mechanism that makes speculation safe.

**Why not CRDTs:** speculative branches are mutually exclusive alternatives, not concurrent collaborators. At most one of them is right. Merging them would be merging a decision the model made with one it did not make. Isolation is the correct primitive; the branch that retires wins wholesale.

**Why exact equality and not similarity for resolution (Hard Rule 4):** a speculative branch has already executed reads and staged writes based on *ŷ*. If *y* differs in any argument — a different ticket id, a different amount, a different file path — every downstream call was computed on the wrong premise. There is no useful notion of "close enough" for a tool call, and any tolerance is a way for a wrong branch's effects to retire.

**Why a placeholder never enters a prompt:** the tempting alternative — let a tool declare a "projected" result such as `{"ok": true}` so the branch's next model call can proceed — still conditions the model on synthetic context wherever the real result would have carried an id, a timestamp or a status. The prompt would differ from the sequential run's, and the model's decision would be made on a different premise even if it happened to match. Rule 13 makes that a fault, not a feature.

**Why free text is a barrier:** a node that emits prose which feeds the next prompt cannot be predicted token-for-token, so it cannot be confirmed by equality. Speculating on it would mean retiring branches on approximate matches, which Hard Rule 4 forbids. The bench measures how much of each workload is behind free-text barriers, because that is part of the honest answer about how much speculation is available.

**Why Python 3.11 and not 3.12:** the previous project shipped with a 3.12 floor and its first two installs failed on 3.11 interpreters with a misleading pip error. Nothing here needs 3.12 syntax. The floor is 3.11 and the CI matrix enforces it.

**Why Sonnet as the bench target and Haiku as a draft:** the online bench needs a target turn slow enough that hiding tool latency behind it is visible, and a draft cheap enough that a miss costs little. Sonnet/Haiku is the pairing the bench can actually afford under the spend cap. The claim is about the runtime, not the models; the config makes both swappable.

---
## 6. Project structure

Create exactly this. Don't reorganise.

```
specunode/
├── pyproject.toml
├── README.md                      # generated numbers only; see Hard Rule 12
├── RESULTS.md                     # generated by bench/report.py — never hand-edited
├── LICENSE
├── specunode.yaml.example
├── src/specunode/
│   ├── __init__.py                # lazy public API: run, resume, replay, tool, World
│   ├── api.py                     # the three convenience entry points over one core
│   ├── cli.py                     # typer app: init, run, resume, replay, ledger, mcp-proxy, bench
│   ├── config.py                  # pydantic models, schema_version, loader
│   ├── ids.py                     # ULID
│   ├── canonical.py               # canonical(), chash()  — Hard Rule 4, Rule 8
│   ├── core/
│   │   ├── decision.py            # ToolCall | Route | Structured | FreeText
│   │   ├── effects.py             # EffectClass, ToolSpec, registry, MCP annotation mapping
│   │   ├── branch.py              # Branch, BranchState, fork/retire/squash/stall
│   │   ├── state.py               # committed state, COW fork, JSON Patch delta, reducers
│   │   ├── scheduler.py           # the run loop: sequential mode + speculative mode
│   │   ├── hazards.py             # dependency analysis over staged effects
│   │   ├── policy.py              # budgets, alpha window, break-even gate (Rule 10)
│   │   └── model.py               # ModelClient protocol, streaming, journaled wrapper
│   ├── buffer/
│   │   ├── store_buffer.py        # stage(), forward(), drain(), discard()
│   │   ├── idempotency.py         # key derivation (Rule 8), dedupe table
│   │   ├── dispatcher.py          # at-least-once dispatch, ack, dead-letter
│   │   └── compensation.py        # COMPENSABLE undo of already-retired effects
│   ├── journal/
│   │   ├── schema.sql             # one DDL for sqlite + postgres
│   │   ├── journal.py             # append, read, fsync discipline (Rule 5)
│   │   ├── replay.py              # ReplayModel, ReplayDivergence
│   │   └── ledger.py              # effect ledger rendering, signing
│   ├── verify/
│   │   ├── gate.py                # resolve(ŷ, y) → CONFIRMED | SQUASHED (Rule 4)
│   │   ├── witness.py             # read-set validation at retirement (OCC)
│   │   └── equivalence.py         # ledger(spec=on) == ledger(spec=off)  (Rule 9)
│   ├── drafters/
│   │   ├── base.py                # Drafter protocol
│   │   ├── t0_stream.py           # early-issue from partial stream
│   │   ├── t1_pattern.py          # PASTE-style pattern index (credited)
│   │   └── t2_model.py            # draft model, optional extra
│   ├── integrations/
│   │   ├── langgraph.py           # wrap a compiled StateGraph
│   │   ├── plain.py               # @specunode.tool, @specunode.node, run_loop()
│   │   └── mcp_proxy.py           # stdio proxy, tools/list + tools/call interception
│   └── testing/
│       ├── world.py               # fake external world with branch-tagged mutation log
│       ├── faults.py              # partition / timeout / duplicate / slow
│       └── models.py              # ScriptedModel, ReplayModel
├── tests/
│   ├── unit/ … integration/ … property/ … chaos/
│   ├── test_no_llm_in_control_path.py     # Hard Rule 1
│   ├── test_leak.py                       # Hard Rule 3 — never skipped
│   ├── test_equivalence.py                # Hard Rule 9 — never skipped
│   ├── test_vocabulary.py                 # Hard Rule 12
│   └── test_numbers_traceable.py          # Hard Rule 12
├── bench/
│   ├── corpus/                    # committed offline traces + manifest
│   ├── workloads/                 # synthetic generators + the three sample apps as workloads
│   ├── baselines.py               # B_seq, B_readonly_spec (PASTE-style), B_naive_parallel, B_specunode
│   ├── offline/run_opportunity.py # how much speculation a trace exposes (no model calls)
│   ├── online/run_latency.py      # wall clock with real target model, budget-capped
│   ├── chaos/run_chaos.py         # crash/partition/duplicate under load
│   ├── adversarial/run_attacks.py # §PHASE 7 strategies
│   ├── demo.py
│   ├── report.py                  # → RESULTS.md
│   ├── check_numbers.py
│   ├── plots/make_plots.py
│   └── make_report_pdf.py
├── examples/
│   ├── support_agent/  ops_agent/  research_agent/
│   └── mcp_client_config.json
├── docs/
│   ├── effect-classes.md  hazards.md  replay.md  mcp-proxy.md  adapters.md  limitations.md
└── .github/workflows/ci.yml  nightly-offline.yml  nightly-online.yml
```

---

## 7. Core contracts

These signatures are the spec. Implement to them. Changing a public signature requires a Progress Log entry saying why.

### Decisions and canonical form

```python
# src/specunode/core/decision.py
@dataclass(frozen=True)
class ToolCall:
    name: str
    args: Mapping[str, JsonValue]

@dataclass(frozen=True)
class Route:
    label: str

@dataclass(frozen=True)
class Structured:
    schema_id: str
    value: JsonValue

@dataclass(frozen=True)
class FreeText:
    content_hash: str          # blake2b of NFC text; the text itself lives in the journal

Decision = ToolCall | Route | Structured | FreeText

# src/specunode/canonical.py
def canonical(obj: JsonValue) -> bytes: ...     # sorted keys, no whitespace, NFC, repr floats
def chash(obj: JsonValue) -> str: ...           # blake2b-256 hex of canonical(obj)
def decisions_equal(a: Decision, b: Decision) -> bool:
    """Exact canonical equality. FreeText never equals anything, including itself,
    for the purpose of branch resolution (it is a barrier)."""
```

### Effect classes and tools

```python
# src/specunode/core/effects.py
class EffectClass(Enum):
    READ = "read"
    WRITE = "write"
    COMPENSABLE = "compensable"
    IRREVERSIBLE = "irreversible"

@dataclass(frozen=True)
class ToolSpec:
    name: str
    effect: EffectClass
    fn: Callable[..., Awaitable[JsonValue]]
    idempotent: bool = False
    compensator: str | None = None              # required iff effect == COMPENSABLE
    forward_keys: Callable[[Mapping[str, JsonValue]], frozenset[str]] | None = None
    witness: bool = False                        # READ returns {"value":…, "witness":…}
    schema: JsonSchema | None = None

class ToolRegistry:
    def register(self, spec: ToolSpec) -> None: ...
    def get(self, name: str) -> ToolSpec:
        """Unknown tool → ToolSpec(effect=WRITE, idempotent=False) synthesised and
        logged at WARNING once. Never raises for unknown; never guesses READ."""
    @classmethod
    def from_mcp_tools(cls, tools: list[McpTool], overrides: Mapping[str, ToolSpec]) -> "ToolRegistry": ...
```

### Branches, state, store buffer

```python
# src/specunode/core/branch.py
class BranchStatus(Enum):
    SPECULATIVE = "speculative"    # running on ŷ, unresolved
    CONFIRMED   = "confirmed"      # y == ŷ, journaled; retirement in progress
    RETIRED     = "retired"        # store buffer drained, delta applied
    SQUASHED    = "squashed"       # y != ŷ or parent squashed; buffer discarded
    STALLED     = "stalled"        # hazard; runtime continues sequentially

@dataclass
class Branch:
    id: str
    parent_id: str | None
    fork_step: int
    predicted: Decision
    lineage: tuple[str, ...]        # branch ids root→self, part of every idempotency key
    status: BranchStatus
    reason: str | None              # for STALLED / SQUASHED
    state: BranchState              # COW fork
    context: list[Message]          # forked messages
    read_set: list[ReadRecord]      # (tool, args_hash, result_hash, witness|None, at_step)
    prompts_sent: list[tuple[int, str]]  # (step, chash(prompt)) for Rule 13 checking at retirement
    staged: list[StagedEffect]      # ordered

# src/specunode/buffer/store_buffer.py
@dataclass(frozen=True)
class StagedEffect:
    id: str
    branch_id: str
    step: int
    call: ToolCall
    effect: EffectClass
    key: str                        # idempotency key, Rule 8
    depends_on: frozenset[str]      # ids of staged effects whose *return value* this needs
    placeholder: str | None         # symbolic handle returned to the branch instead of a value

class StoreBuffer:
    def stage(self, branch: Branch, call: ToolCall, spec: ToolSpec) -> StagedEffect:
        """Never executes. Returns a StagedEffect whose placeholder stands in for the
        tool's return value. If a later call's args contain a placeholder, that is a
        hazard (see hazards.py) unless the tool declares it can accept a handle."""
    def forward(self, branch: Branch, read: ToolCall, spec: ToolSpec) -> ForwardResult:
        """Store-to-load forwarding: if a staged WRITE in this branch touches a key
        this READ touches (both via forward_keys), return HAZARD unless the write's
        spec provides an in-branch projection. Sibling branches are invisible."""
    async def drain(self, branch: Branch, dispatcher: Dispatcher) -> DrainReport:
        """Only callable when branch.status == CONFIRMED and the confirming journal
        entry is durable. Dispatches in stage order. Each effect: DISPATCHED(ack) |
        DEAD_LETTER(reason). Idempotent on the key: draining twice sends nothing twice."""
    def discard(self, branch: Branch) -> int:
        """Drops every staged effect. Returns the count. Journals the discard."""
```

### Scheduler, hazards, policy

```python
# src/specunode/core/scheduler.py
class Scheduler:
    def __init__(self, graph: GraphAdapter, registry: ToolRegistry, journal: Journal,
                 buffer: StoreBuffer, drafters: Sequence[Drafter], policy: Policy,
                 target: ModelClient) -> None: ...
    async def run(self, run_id: str, inputs: JsonValue) -> RunResult:
        """Sequential when policy.speculation is off or the drafter list is empty.
        Otherwise: at each decision point, ask drafters for ŷ (T0 first, then T1, T2);
        fork a branch per distinct ŷ up to max_inflight_branches; execute the branch
        under hazard analysis; when y arrives (journaled), resolve every open branch."""

# src/specunode/core/hazards.py
class Hazard(Enum):
    RETURN_VALUE_DEPENDENCY = "depends on staged write's return value"
    UNDECLARED_TOOL         = "tool has no declared effect class"
    FREE_TEXT_NODE          = "node emits free text"
    READ_AFTER_STAGED_WRITE = "read touches a key a staged write touches; no forwarding"
    BUDGET                  = "speculation budget exhausted"
    IRREVERSIBLE_ON_PATH    = "irreversible effect would need staging"  # policy-configurable
    MODEL_TURN_AFTER_STAGED_WRITE = "next model call would contain a placeholder"  # Rule 13
    READ_BUDGET             = "speculative read budget exhausted"

def analyse(branch: Branch, next_call: ToolCall, spec: ToolSpec, policy: Policy) -> Hazard | None: ...

# src/specunode/core/policy.py
@dataclass(frozen=True)
class Policy:
    speculation: bool = True
    max_inflight_branches: int = 1          # top-1 by default; top-k multiplies upstream reads
    max_speculation_depth: int = 3
    max_wasted_tokens: int = 20_000
    max_speculative_reads: int = 50         # ReadBudget: upstream reads a squashed branch may have cost
    alpha_window: int = 20
    alpha_floor: float | None = None      # None → use break-even measured for the workload
    stage_irreversible: bool = False      # default: IRREVERSIBLE is a barrier, not staged
    on_stale_read: Literal["squash", "stall"] = "squash"
```

### Journal, replay, ledger

```python
# src/specunode/journal/journal.py
class Journal:
    def append(self, entry: Entry) -> int:
        """Durable before return (fsync). Returns offset. Entries are immutable."""
    def read(self, run_id: str, after: int = 0) -> Iterator[Entry]: ...

# entry kinds (schema.sql): run_started, model_request, model_response, tool_request,
#   tool_result, branch_forked, branch_resolved, effect_staged, effect_dispatched,
#   effect_dead_lettered, effect_discarded, state_delta_applied, read_validated,
#   policy_event, run_finished

# src/specunode/journal/replay.py
class ReplayModel(ModelClient):
    """Serves model_response entries from the journal, in order. If the incoming
    request's canonical form differs from the journaled model_request at that step,
    raises ReplayDivergence(step, diff). Never calls a model."""

# src/specunode/journal/ledger.py
@dataclass(frozen=True)
class LedgerRow:
    effect_id: str; call: ToolCall; key: str; branch_id: str
    authorised_by_step: int             # journal offset of the confirming model_response
    status: Literal["DISPATCHED", "DEAD_LETTER", "COMPENSATED"]
    ack: JsonValue | None

@dataclass(frozen=True)
class Ledger:
    run_id: str
    rows: tuple[LedgerRow, ...]
    squashed_branches: int
    discarded_effects: int
    stalls: tuple[tuple[int, Hazard], ...]
    reads_validated: tuple[int, int]     # (fresh, total)
    wasted_tokens: int
    speculative_reads_upstream: int      # reads that reached upstream from squashed branches
    context_divergences: int             # Rule 13 faults
    alpha: float | None
    signature: str                       # Ed25519 over canonical(rows…)

def normalise_for_equivalence(l: Ledger) -> bytes:
    """Rows in dispatch order with branch ids and timestamps stripped; keys re-derived
    with an empty lineage. Two runs are equivalent iff these bytes match (Rule 9)."""
```

### Drafters and the gate

```python
# src/specunode/drafters/base.py
class Drafter(Protocol):
    tier: Literal[0, 1, 2]
    async def predict(self, ctx: DraftContext) -> list[Decision]:
        """Zero or more candidate decisions, most likely first. T0 returns only
        decisions the target has already emitted in-stream (α = 1 by construction).
        T1/T2 return guesses. Never raises; an empty list means 'no opinion'."""

# src/specunode/verify/gate.py
def resolve(branch: Branch, actual: Decision) -> BranchStatus:
    """CONFIRMED iff decisions_equal(branch.predicted, actual). Otherwise SQUASHED.
    Pure. Deterministic. Tested by property: for all d, resolve(d, d) is CONFIRMED
    except FreeText; for all d != e, SQUASHED."""

# src/specunode/verify/witness.py
async def validate_reads(branch: Branch, registry: ToolRegistry) -> ReadValidation:
    """For each ReadRecord with a witness, re-fetch the witness; stale iff changed.
    Reads without witnesses are reported as 'unwitnessed', never as 'fresh'."""
```

### The status lattice

Per staged effect, in resolution order (the order matters, and the first rule that matches wins):

| Rule | Condition | Outcome |
|---|---|---|
| E1 | branch SQUASHED (own mismatch or ancestor squashed) | `DISCARDED` — never dispatched |
| E2 | branch STALLED | `DISCARDED` — the sequential re-execution will re-stage it |
| E3 | branch CONFIRMED, read validation found a stale witnessed read | branch → SQUASHED (policy `squash`) or STALLED; effect `DISCARDED` |
| E4 | branch CONFIRMED, dispatch acked | `DISPATCHED` |
| E5 | branch CONFIRMED, dispatch failed after retries | `DEAD_LETTER(reason)` — run finishes `ok=False` |
| E6 | previously DISPATCHED, later compensation requested | `COMPENSATED` or `DEAD_LETTER` |

Rule E3 before E4 is the ordering that carries most of the integrity: a branch whose reads went stale between speculation and confirmation must not retire just because the model's decision matched.

### Configuration

```yaml
# specunode.yaml
schema_version: 1
journal:
  kind: sqlite            # sqlite | postgres
  path: ./.specunode/journal.db
target:
  provider: anthropic
  model: claude-sonnet-5
drafters:
  - tier: 0               # early-issue from stream; always on
  - tier: 1               # pattern index
    index_path: ./.specunode/patterns.json
    order: 2
  # - tier: 2
  #   provider: anthropic
  #   model: claude-haiku-4-5
policy:
  max_inflight_branches: 1
  max_speculative_reads: 50
  max_speculation_depth: 3
  max_wasted_tokens: 20000
  alpha_window: 20
  stage_irreversible: false
  on_stale_read: squash
tools:                    # overrides for effect class when code/MCP annotations are wrong
  send_email: {effect: irreversible}
  create_ticket: {effect: write, idempotent: true, forward_keys: "ticket:{args.customer_id}"}
state:
  reducers:
    findings: append
    summary: last_write
```

---
## PHASE 0 — Scaffold, canonical form, journal

Goal: an installable package with a journal that records and replays a scripted run — no speculation, no tools yet.

- [x] **0.1 Scaffold.** `uv init`, hatchling, ruff, mypy strict, pytest, CI matrix 3.11/3.12/3.13 on Ubuntu + macOS. `specunode --version` works from a built wheel.
  *Verify:* `uv build && uv tool install dist/*.whl && specunode --version` in a clean venv on 3.11.
- [x] **0.2 Canonical form.** `canonical()`, `chash()`, `decisions_equal()`. Property tests: round-trip through `json.loads` is a fixed point; key order and whitespace never change the hash; NFC vs NFD strings hash equal; `-0.0` and `0.0` hash equal; NaN raises; two `FreeText` never resolve equal.
  *Verify:* `hypothesis` suite passes 2,000 examples per property.
- [x] **0.3 Journal.** SQLite WAL, `synchronous=FULL`, one `entries` table (`run_id, offset, kind, payload_json, payload_hash, prev_hash, ts`). `append()` returns only after `fsync`. Hash-chained per run. Same DDL file loads on Postgres 16 under `testcontainers`.
  *Verify:* a test appends 1,000 entries, kills the process (`os._exit`) mid-append in a subprocess at a random point, reopens, and asserts the chain verifies and the last entry is either fully present or absent — never partial.
- [x] **0.4 ModelClient + journaled wrapper.** Protocol with `complete()` and `stream()`. `JournaledModel` writes `model_request` before the call and `model_response` before returning the result to the caller. Anthropic adapter behind an optional extra; `ScriptedModel` for tests.
  *Verify:* `ReplayModel` fed the journal of a `ScriptedModel` run reproduces every response; a changed request at step *k* raises `ReplayDivergence(k, diff)` and the test asserts no further entries were written.
- [x] **0.5 Vocabulary + no-LLM-in-control-path tests.** `tests/test_vocabulary.py` and `tests/test_no_llm_in_control_path.py` exist and pass on the empty packages.
  *Verify:* planting the word "guaranteed" in `README.md` fails the vocabulary test; planting `messages=[` in `src/specunode/core/policy.py` fails the control-path test.

**Phase Gate 0:** wheel installs on 3.11; journal survives the kill test; replay diverges loudly; both honesty tests fire on planted violations.

---

## PHASE 1 — Effects, store buffer, fake world, the leak test

Goal: writes can be staged and drained, and there is a world that will tell us if one ever escapes.

- [x] **1.1 Effect classes + registry.** `ToolSpec`, `ToolRegistry`, MCP annotation mapping per §5. Unknown tools synthesise `WRITE(idempotent=False)` and log once.
  *Verify:* a tool registered with no class is treated as WRITE; an MCP tool with `readOnlyHint=true` maps to READ; one with no annotations maps to WRITE; overrides in config win.
- [x] **1.2 Fake world.** `World` with tables (`customers`, `tickets`, `jobs`, `messages`) and endpoint-style tools; every mutation appended to `world.mutations` as `(branch_id, effect_key, tool, args_hash, ts)`. Fault injection per §5. `World.reads_with_witness()` returns `{value, witness}` where witness is a per-row version counter.
  *Verify:* `world.mutations` is the only way state changes; a test monkeypatches every public write path and asserts each appends exactly one record.
- [x] **1.3 Idempotency keys + dedupe.** Rule 8 derivation. Dedupe table in the journal keyed on `key`; dispatch checks it first.
  *Verify:* property test — same `(run, lineage, step, tool, args)` → same key across processes; any one component changed → different key; draining the same buffer twice sends each effect once (asserted against `world.mutations`).
- [x] **1.4 Store buffer.** `stage()`, `forward()`, `drain()`, `discard()` per §7. Placeholders are `"$specunode.handle:<effect_id>"` strings; `hazards.analyse` finds them anywhere in a later call's canonical args.
  *Verify:* staging a WRITE causes no `world.mutations`; draining a CONFIRMED branch causes exactly one per staged effect, in stage order; `discard()` on a SQUASHED branch causes none and journals the count.
- [x] **1.5 Dispatcher + dead-letter.** At-least-once with bounded exponential backoff; on exhaustion `DEAD_LETTER(reason)` and the run finishes `ok=False`. `COMPENSABLE` effects record their compensator call for later.
  *Verify:* `world.partition(at=2)` during drain: effects 1 dispatched, 2 dead-lettered after retries, 3 not attempted; resume after partition heals dispatches 2 and 3 with no duplicate of 1.
- [x] **1.6 THE LEAK TEST.** `tests/test_leak.py`: 500 randomly generated branch trees (hypothesis), random resolution outcomes, random faults; after every run assert `{m.branch_id for m in world.mutations} ⊆ {b.id for b in branches if b.status == RETIRED}`. **Mandatory. Never skipped. Never marked xfail.**
  *Verify:* the test exists, runs in CI, and a deliberately planted bug (drain on CONFIRMED before the confirming journal entry is durable) makes it fail.

**Phase Gate 1:** leak test green on 500 trees; partition test green; unknown tools are WRITE.

---

## PHASE 2 — State, sequential scheduler, LangGraph integration, resume

Goal: a real graph runs end to end through SpecuNode in sequential mode with journaling, and survives a kill.

- [x] **2.1 Committed state + COW fork.** `state.py` per §5; JSON Patch delta at retirement; reducers only for sequential same-key writes.
  *Verify:* property — fork, mutate the fork, assert committed unchanged; retire, assert committed == committed ⊕ patch; two forks of the same parent never see each other.
- [x] **2.2 GraphAdapter + sequential scheduler.** `GraphAdapter` protocol (`nodes()`, `next(state) → NodeRef | END`, `run_node(node, state, model, tools) → Decision | FreeText`). `Scheduler.run()` in sequential mode: every model call journaled, every tool call classified, READ executed, WRITE staged then immediately retired (a sequential run is a chain of single-branch retirements — one code path, not two).
  *Verify:* the support example runs to completion against `World` with `ScriptedModel`; ledger has one row per write; `world.mutations` matches ledger rows exactly.
- [x] **2.3 LangGraph integration.** `wrap(compiled_graph, registry, config) → SpecuNodeGraph` with `.ainvoke()` and `.astream()`; nodes' tool calls are routed through `ToolRunner`; the model client inside nodes is replaced by `JournaledModel` via LangGraph's configurable runnable binding — the developer's graph file does not change. Document what is not supported (interrupts inside a speculative branch: stall).
  *Verify:* `examples/support_agent` runs unchanged under vanilla LangGraph and under `wrap()`; both produce the same final state; only the wrapped one produces a ledger.
- [x] **2.4 Plain-Python integration.** `@specunode.tool(effect=…)`, `@specunode.node`, `specunode.run_loop(step_fn, …)`.
  *Verify:* the ops example implemented both ways yields identical ledgers under `ReplayModel`.
- [x] **2.5 Resume.** `specunode resume <run_id>`: reads the journal, restores committed state, re-drives from the last durable entry; effects already `DISPATCHED` are not re-sent (dedupe table); an in-flight drain resumes.
  *Verify:* `tests/chaos/test_kill_resume.py` — SIGKILL the subprocess at 15 random points across 20 runs; resume; assert the final ledger and `world.mutations` equal the uninterrupted run's, byte for byte after normalisation.
- [x] **2.6 Replay CLI.** `specunode replay <run_id> [--speculation on|off]` runs the graph with `ReplayModel`; `specunode ledger <run_id>` prints the ledger.
  *Verify:* replay of every committed example produces a ledger equal to the original.

**Phase Gate 2:** kill/resume test green at 15 points; LangGraph example unchanged; replay equal.

---

## PHASE 3 — Speculation: drafters, branches, the gate, hazards, policy

Goal: the runtime runs ahead, and everything it runs ahead is either confirmed exactly or thrown away.

- [x] **3.1 T0 early-issue drafter.** Parse the target's stream; each `tool_use` block that completes becomes a `Decision` immediately available to the scheduler. Reads issue at once; writes stage at once. α is 1 by construction and the ledger records tier 0 separately.
  *Verify:* with a `ScriptedModel` streaming three tool calls over 300 ms, the READ among them completes before the stream ends (asserted via timestamps); the WRITEs stage and retire only after the stream's end-of-turn entry is journaled.
- [x] **3.2 T1 pattern drafter.** Mine the journal's `tool_request` sequences per workload into an order-*k* transition table over tool *signatures* (name + argument shape); argument templates that reference prior outputs by JSONPath; predictions carry the template-instantiated args. This is PASTE's mechanism (credited in the module docstring and README). Deterministic given the index file.
  *Verify:* trained on 200 scripted ops runs, the index predicts the next call with a measured top-1 rate reported in the test output (no threshold asserted — the number is a result, not a requirement); predictions are byte-identical across three runs.
- [x] **3.3 Branch fork/execute.** Scheduler speculative mode: at each decision point collect candidates from T0 → T1 → T2, dedupe by canonical form, fork up to `max_inflight_branches` as `asyncio` tasks with COW state and forked context. Each branch executes under `hazards.analyse` before every call; a hazard sets `STALLED` and cancels the task.
  *Verify:* with two distinct candidates, two branches run; their `world.mutations` footprint is zero until resolution; a call whose args include a placeholder stalls with `RETURN_VALUE_DEPENDENCY`.
- [x] **3.4 The gate.** `resolve()` per §7; scheduler resolves all open branches when the journaled `model_response` for the step arrives; at most one CONFIRMED; the rest SQUASHED with cancellation; the confirmed branch's context becomes canonical, squashed contexts are dropped.
  *Verify:* property — over random decision pairs, exactly-equal → CONFIRMED else SQUASHED; `FreeText` never confirms; after resolution the canonical context contains no message from a squashed branch (asserted by message ids).
- [x] **3.5 Read validation at retirement (E3).** For witnessed reads, re-fetch witnesses before drain; stale → policy. Unwitnessed reads are reported, never counted fresh.
  *Verify:* mutate a `World` row between the speculative read and confirmation; assert the branch squashes (policy `squash`), no effect dispatches, and the sequential re-execution reads the new value.
- [x] **3.6 Policy + budgets.** Rule 10: inflight/depth/wasted-token caps; rolling α over `alpha_window`; `alpha_floor` gate; when speculation is disabled mid-run a `policy_event` is journaled and the ledger states it.
  *Verify:* a `ScriptedModel` whose decisions never match the drafter drives α to 0 within one window; the scheduler disables speculation; total wasted tokens ≤ `max_wasted_tokens`; wall clock of the run ≤ sequential wall clock + journaling overhead × 1.1 (overhead measured in the same test).
- [x] **3.7 Cancellation is squash.** Squashing cancels the branch task; a tool adapter mid-READ is cancelled cooperatively; a tool adapter that cannot be cancelled finishes and its result is discarded. Never a staged write dispatched by a cancelled task.
  *Verify:* a slow READ (`world.slow(read, 2000)`) on a branch squashed at 100 ms: the read's result never enters the canonical context; `world.mutations` unchanged.

- [x] **3.8 Context identity (Rule 13).** Before a branch sends a model request: reject if any placeholder appears in the canonical prompt (`MODEL_TURN_AFTER_STAGED_WRITE` → stall); assemble tool results in program order; record `chash(prompt)` on the branch. At retirement, rebuild the prompt for each recorded step from the canonical context and real results and compare hashes; mismatch → `ContextDivergence`, branch squashed, its model output never used, fault journaled.
  *Verify:* a branch that stages a write and then tries a model call stalls with the named hazard; a scripted world whose read results arrive out of order still produces the program-order prompt; a planted bug that appends results in completion order fails the test.
- [x] **3.9 Read budget.** `max_speculative_reads` per run; exhausted → `READ_BUDGET` hazard; ledger reports `speculative_reads_upstream`.
  *Verify:* a drafter that always guesses reads is capped at the budget and the ledger count equals `world` read calls attributed to squashed branches.

**Phase Gate 3:** leak test still green with speculation on; context-identity check stalls a model call after a staged write; T0/T1 drafters produce measured (not asserted) α on the examples; stale-read squash works; budget gate disables speculation on a hostile script.

---

## PHASE 4 — MCP proxy

Goal: a developer who cannot change their agent's code still gets the store buffer.

- [x] **4.1 Proxy skeleton.** `specunode mcp-proxy --upstream "<cmd>" --config specunode.yaml`; stdio; forwards `initialize`, `tools/list` (annotations passed through, overrides applied), and everything not tool-related unchanged.
  *Verify:* a reference MCP client lists tools through the proxy and sees the upstream's list with effect-class annotations merged from config.
- [x] **4.2 `tools/call` interception.** READ → forwarded upstream immediately, result journaled. WRITE/COMPENSABLE/IRREVERSIBLE → staged in the run's current branch; the proxy returns the placeholder handle *as the tool result* with `isError=false` and a structured `_specunode: {staged: true, effect_id}` field; drained on retirement.
  *Verify:* a scripted client issues read, write, read; the upstream receives the two reads immediately and the write only after the proxy receives the `specunode/retire` notification (below).
- [x] **4.3 Decision boundary over MCP.** The proxy cannot see the model, so branch resolution needs a signal: the client sends `notifications/specunode/decision` with the step's actual `Decision` (the LangGraph and plain integrations do this automatically; for a foreign client, `specunode retire <run_id> --step N --decision <json>` does it from a terminal). Until a decision arrives, staged writes stay staged and the proxy's `status` tool says so.
  *Verify:* without the notification, no write ever reaches upstream (timed test, 5 s); with it, the buffer drains in stage order.
- [x] **4.4 Six MCP tools of its own.** `specunode.status`, `specunode.ledger`, `specunode.stall` (force sequential), `specunode.discard` (squash current branch — requires elicitation on protocol versions that support it; refused with the terminal command otherwise), `specunode.retire`, `specunode.replay_check`.
  *Verify:* a client that cannot elicit is refused `discard` with the exact terminal command in the error text.
- [x] **4.5 Security posture.** Loopback only for the optional streamable-HTTP mode; per-run token; no cookies; Origin check. Documented in `docs/mcp-proxy.md`.
  *Verify:* a cross-origin request is rejected; a request without the token is rejected.

**Phase Gate 4:** the support example, driven by a generic MCP client through the proxy, produces the same ledger as the LangGraph integration.

---

## PHASE 5 — Equivalence, chaos, concurrency

Goal: prove Rule 9 and Rule 6 under hostile conditions, not on happy paths.

- [x] **5.1 THE EQUIVALENCE TEST.** `tests/test_equivalence.py`: for every workload in `bench/workloads/` and every committed corpus trace, run under `ReplayModel` with speculation off and with speculation on (each drafter tier, and all tiers), and assert `normalise_for_equivalence(ledger_off) == normalise_for_equivalence(ledger_on)`. **Mandatory. Never skipped.**
  *Verify:* a planted bug (retire on `SQUASHED`) fails it; a planted bug (dispatch order reversed) fails it.
- [x] **5.2 Chaos.** `bench/chaos/run_chaos.py`: SIGKILL at 25 random points per run × 40 runs with speculation on; partition during drain; duplicate delivery of acks; slow reads that outlive their branch. After each: resume, then assert ledger equivalence with the clean run and the leak invariant.
  *Verify:* 0 leaks, 0 duplicates, 0 equivalence failures across the matrix; the numbers are written to `bench/results/chaos.json`.
- [x] **5.3 Concurrency.** 20 runs sharing one journal file and one `World`; branches interleaved; assert per-run ledgers equal their solo runs and that no branch ever observed a sibling's staged effect (instrumented `forward()` records every lookup).
  *Verify:* `bench/results/concurrency.json` shows 0 cross-branch observations.
- [x] **5.4 Forged / tampered ledgers.** `specunode verify-ledger` checks the Ed25519 signature and the journal chain; a ledger whose rows were edited, and a ledger signed with a key that signs no other ledger in the store, are both rejected, with the reason stated (integrity vs origin).
  *Verify:* both forgeries rejected; a genuine ledger accepted.

- [x] **5.5 THE CONTEXT-EQUIVALENCE TEST (live, shadow mode).** `tests/test_context_equivalence.py`: run each sample app with a `ScriptedModel` that records every prompt it receives, once sequentially and once with speculation on; assert the sequence of canonical-step prompts is byte-identical, and that every speculative prompt that was sent equals the canonical prompt at that step. This is the test Q1 needs and the replay-based test cannot provide. **Mandatory. Never skipped.**
  *Verify:* a planted bug that lets a placeholder reach a prompt fails it; a planted completion-order append fails it.

**Phase Gate 5:** equivalence, context-equivalence and leak tests green across all workloads and tiers; chaos matrix at zero on all three counters.

---

## PHASE 6 — Measurement: how much speculation is really there

Goal: numbers, with the negative ones first.

- [x] **6.1 Corpus.** `bench/corpus/fetch.py` pulls the seeded sample from `nebius/SWE-rebench-openhands-trajectories`; normalise to `(tool_name, args_shape, effect_class_guess_for_analysis_only, refs_prior_output)` sequences. If unavailable (Decision Gate D1), run the three sample apps with the target model on 60 seeded tasks each and commit the journals. The manifest is byte-identical on rebuild. **Effect classes in the corpus are labelled by a committed hand-written table per tool name, never by heuristic on unlabelled tools; unlabelled tools are `WRITE` in the analysis just as in the runtime.**
  *Verify:* manifest hash committed; rebuild reproduces it.
- [x] **6.2 Offline opportunity analysis.** `bench/offline/run_opportunity.py` (no model calls): for each trace, compute (a) fraction of steps predictable by T1 at top-1/top-3 (leave-one-trace-out); (b) fraction of predicted calls that are READ vs WRITE; (c) fraction of steps blocked by each hazard class; (d) mean/median speculable run length past a write under PASTE-style policy (writes are barriers) vs SpecuNode (writes stage); (e) fraction of steps behind free-text barriers; (f) fraction of steps that are a model call immediately consuming a write's result (`MODEL_TURN_AFTER_STAGED_WRITE` shape — no past-write gain possible). Bootstrap 95% CIs over traces. Output `bench/results/opportunity.json` + plots.
  *Verify:* runs on the committed corpus in under 10 minutes on CPU; every number carries a CI.
- [x] **6.3 Baselines.** `bench/baselines.py`: `B_seq` (sequential through SpecuNode, speculation off), `B_readonly_spec` (speculation on, all non-READ tools are barriers — PASTE's policy, credited), `B_naive_parallel` (speculation on, writes execute for real, branch discarded on mismatch — the langchain-nvidia failure mode), `B_specunode`. All four share the journal so they replay the same model outputs.
  *Verify:* `B_naive_parallel` leaks on the Demo 1 workload and the leak count is reported, not hidden.
- [ ] **6.4 Online latency bench.** `bench/online/run_latency.py`: the three sample apps × 30 seeded tasks × {B_seq, B_readonly_spec, B_specunode} with the real target model, budget-capped; per run: wall clock, tokens (target, draft), wasted tokens, α per tier, stalls by hazard, stale reads, effects dispatched, leaks (must be 0). Report the break-even α per workload (the α at which `B_specunode` wall clock equals `B_seq`). Bootstrap CIs over tasks.
  *Verify:* `bench/results/latency.json`; spend stays under the cap and the report records the spend.
- [x] **6.5 Overhead.** Journaling + classification overhead of `B_seq` vs the same graph on vanilla LangGraph with no SpecuNode, same `ReplayModel`. Reported as absolute ms per step and as a fraction of wall clock.
  *Verify:* `bench/results/overhead.json`.
- [x] **6.6 Report generation.** `bench/report.py` → `RESULTS.md`; `bench/plots/make_plots.py` → PNGs; `bench/check_numbers.py` enforces README traceability.
  *Verify:* `RESULTS.md` regenerates identically from committed JSON; CI fails on a planted untraceable number.

**Phase Gate 6:** opportunity + latency + overhead results committed with CIs; `RESULTS.md` generated; the anti-results (workloads where speculation buys ≤ 5% or is disabled by policy) are in the README with the same prominence as the wins.

---

## PHASE 7 — Break your own runtime

Goal: publish the attacks that beat it, with measured rates. Each strategy is one file that returns a rate; a strategy that fails to run is an error row, never a dropped row.

- [x] **7.1 Misdeclared tool.** A tool declared READ that actually writes. Measure: effects from squashed branches reaching the world. Expected non-zero — this is the runtime's trust boundary and the README says so.
- [x] **7.2 Hidden side effect in a read.** A READ with logging/rate-limit/billing side effects upstream. Measure: upstream calls from squashed branches (they happen; they are reported as "speculative reads that reached upstream" in every ledger, never hidden).
- [x] **7.3 Stale reads under contention.** Another actor mutates rows between speculative read and confirmation at rates 1/s, 10/s, 100/s. Measure: stale-read squash rate, and the fraction of stale reads that were *unwitnessed* (undetectable). The second number is the honest one.
- [x] **7.4 Non-idempotent tool with duplicate delivery.** `world.duplicate_delivery` on a tool declared `idempotent=False`. Measure: duplicates reaching the world with vs without the dedupe table; document that dedupe protects the *dispatcher's* retries, not the network beyond it.
- [x] **7.5 Return-value laundering.** A branch copies a placeholder handle into free text and a later tool receives it inside a string. Measure: does hazard analysis catch placeholders embedded in strings, in nested arrays, base64-encoded? Report the miss rate; fix what can be fixed (substring scan of canonical args), document what cannot (encoded).
- [x] **7.6 Prompt-injected tool call.** A READ result contains text instructing the model to call `send_email`. The drafter (T1) predicts it; the target model does not emit it. Measure: 0 dispatches expected (it is a squash). Then the target model *does* emit it: it dispatches, because SpecuNode is not an authorization layer — the README says so and points to SCOPEGATE-style per-call policy as the missing piece.
- [x] **7.7 Drafter poisoning.** Train the T1 index on traces with an adversarial "strong chain" that ends in a write. Measure: wasted tokens and stall/squash counts; assert leaks stay 0.
- [x] **7.8 Replay under model drift.** Replay a journal after changing the system prompt by one token, and after changing the tool list. Measure: step of first `ReplayDivergence`. Expected: step 1 in both cases.
- [x] **7.10 Asynchronous side effect behind a READ.** A tool declared READ whose synchronous response is `{"status": "queued", "job_id": …}` and whose upstream enqueues a background job that writes and notifies. The branch is squashed; the job runs anyway. Measure: leaked effects per squashed branch. Expected non-zero. Document: *a tool that enqueues, schedules or triggers anything asynchronously is not a READ, whatever its HTTP verb.*
- [x] **7.9 Speculation past an IRREVERSIBLE with `stage_irreversible=true`.** Measure the latency gain and show the ledger row that says an irreversible effect was retired on a decision the model made — correct, but the docs must say the default is off and why.

*Verify for the phase:* `bench/adversarial/run_attacks.py --all` writes `bench/results/attacks.json` with a row per strategy; `RESULTS.md` has a section "What beats it" placed before "What it does well".

**Phase Gate 7:** every strategy has a measured rate or an error row; README's limitations section lists 7.1, 7.2, 7.3-unwitnessed, 7.4, 7.5-encoded, 7.6 and 7.10 as things the runtime cannot fix.

---

## PHASE 8 — Survive contact with a real pipeline

- [x] **8.1 Interrupts / human-in-the-loop.** A LangGraph `interrupt()` inside a speculative branch is a hazard (`STALLED`); on the canonical path it works as in vanilla LangGraph, with the pending interrupt journaled.
- [x] **8.2 Streaming to the user.** `.astream()` yields only canonical-path tokens; speculative branches' model output never streams to the user (it may be squashed).
- [x] **8.3 Sub-graphs.** A node that is itself a graph forks its own branch tree under the parent's lineage; retirement is nested; the leak test covers nesting.
- [x] **8.4 Adapter contract doc + suite.** `docs/adapters.md` specifies what a tool adapter must satisfy (cancellable, idempotent on key when declared, witness format); `tests/test_adapter_suite.py` runs every bundled adapter and the `World` tools through it.
- [x] **8.5 Postgres journal in CI** under `testcontainers`; the same suite passes.

**Phase Gate 8:** interrupts, streaming, sub-graphs covered by tests; adapter suite green on all adapters.

---

## PHASE 9 — Ship

- [x] **9.1 README** with: one-paragraph CPU analogy; the three demos with their printed outputs; the opportunity plot; the latency table with CIs; the break-even α per workload; "What beats it"; "What this is not" (not a durable-execution platform, not an authorization layer, not a context manager); credits to PASTE, Claude Code's executor, langchain-nvidia, ToolAhead, SagaLLM, ATP, SCOPEGATE, Temporal/DBOS/Restate.
- [x] **9.2 Docs** (`docs/*.md`) complete; `specunode init` writes `specunode.yaml` + `.specunode/`.
- [ ] **9.3 Release.** Tag `v0.1.0`, `uv build`, publish to PyPI as `specunode`, install from PyPI in a clean 3.11 venv on both OSes and run Demo 1 from the published wheel. Record the exact install command that failed, if any, in the docs the same day.
- [x] **9.4 Report.** `bench/make_report_pdf.py` regenerates the technical report from `bench/results/*.json` — every number in the PDF is read from a file.

**Phase Gate 9:** `pip install specunode` works on 3.11; demos run from the published wheel; `check_numbers.py` green against the published README.

---

## Definition of done

- [ ] Wheel installs on Python 3.11, 3.12, 3.13 on macOS and Ubuntu
- [ ] Leak test (Rule 3), equivalence test (Rule 9) and context-equivalence test (Rule 13) run on every workload, every tier, every CI job; none is skippable
- [x] Kill/resume at 15 points and chaos matrix at zero leaks, zero duplicates, zero equivalence failures
- [ ] LangGraph integration works on an unchanged graph file; plain-Python integration works; MCP proxy works with a generic client
- [x] T0, T1 drafters shipped; T2 behind an extra
- [ ] Offline opportunity analysis, online latency bench (budget-capped), overhead bench, adversarial suite — all with committed JSON, CIs, and commands
- [x] `RESULTS.md`, README numbers, and the PDF are generated; `check_numbers.py` and the vocabulary check are green
- [x] "What beats it" section in README and report, before the wins
- [x] Every prior-art project in §3 credited by name in the README
- [ ] Published to PyPI; installed and demoed from the published wheel

---

## Known limitations to document, not fix

- The effect class is the developer's word. A READ that writes defeats the store buffer completely (7.1). The runtime cannot detect this and does not try.
- A READ whose upstream enqueues asynchronous work is a write in disguise (7.10). The synchronous response looks harmless; the side effect happens later. Such tools must be declared WRITE, and the docs say so on the first page.
- Past-write speculation hides tool latency, not model latency. The model call after a staged write always waits for the real result (Rule 13). Workloads shaped `model → write → model(reads result)` gain nothing from it, and the bench reports that fraction.
- Speculative reads reach upstream systems even when the branch is squashed (7.2). Any read with billing, rate-limit or audit side effects is a speculative cost, and the ledger counts them.
- Reads without a witness cannot be validated at retirement (7.3). The ledger reports them as unwitnessed, never as fresh.
- Idempotency dedupe covers the dispatcher's own retries. Network-level duplication beyond the dispatcher needs the tool's own idempotency (7.4).
- A placeholder handle encoded or transformed inside a string can evade hazard analysis (7.5).
- SpecuNode is not authorization. A tool call the target model actually emits is dispatched (7.6).
- Free-text nodes are barriers. Workloads dominated by prose-to-prose handoffs get little or no speedup, and the bench says how much.
- The journal is local. Hosting a run inside Temporal/DBOS/Restate is documented as a pattern, not shipped as an integration.
- Research prototype: one target-model provider adapter, LangGraph + plain Python + MCP, three sample apps.

---

## Decision gates — stop and reassess if any of these fire

- **D1 — Corpus.** If `nebius/SWE-rebench-openhands-trajectories` is unavailable at build time or its license changes, generate the corpus from the sample apps (6.1) and state in the README that the opportunity analysis is on self-generated traces.
- **D2 — Online bench budget.** If the spend cap is hit before 30 tasks per app, report what completed with the reduced *n* and its wider CI; do not raise the cap silently.
- **D3 — T1 never beats break-even.** If the measured break-even α exceeds the measured T1 α on every workload, the headline is "T0 early-issue plus the store buffer is the useful part; pattern drafting did not pay for itself on these workloads". That is a publishable result. Do not tune the workloads until it flips.
- **D4 — LangGraph API drift.** If the node-runner substitution needs private APIs, use them, pin the version, and document the pin; do not fork LangGraph.
- **D5 — MCP annotations sparse in the wild.** If tested upstream servers ship no annotations, the proxy defaults everything to WRITE (Rule 2) and the docs say the per-tool override table is mandatory for any speedup.

---

## Suggested schedule

| Phase | Days |
|---|---|
| 0 Scaffold, canonical, journal | 2 |
| 1 Effects, store buffer, leak test | 3 |
| 2 State, scheduler, LangGraph, resume | 4 |
| 3 Speculation | 5 |
| 4 MCP proxy | 3 |
| 5 Equivalence, chaos, concurrency | 3 |
| 6 Measurement | 4 |
| 7 Break it | 3 |
| 8 Real pipeline | 2 |
| 9 Ship | 2 |

---

## Progress Log

*Agent: append one line per completed task, decision, blocker or defect. Format: `[task-id] what — date`. This is the audit trail the Final Report is written from.*

```
[0.1] Scaffold: uv + hatchling + ruff(100) + mypy --strict, CI matrix 3.11/3.12/3.13 x {ubuntu,macos}. Wheel built and installed clean on all three locally. — 2026-09-15
[0.1] Decision: repo built in place at the project root rather than a nested specunode/ directory; the section 6 tree is reproduced exactly under it. — 2026-09-15
[0.1] Decision: CI runs the three mandatory invariant tests as an explicit step with --runxfail and greps the report for skipped/xfailed/deselected, so none can be quietly disabled. — 2026-09-15
[0.2] canonical()/chash()/decisions_equal(). 28 property tests at 2,000 examples each. — 2026-09-15
[0.2] Decision: canonical() rejects two object keys that collide only after NFC normalisation, rather than silently dropping one — a silent drop would let two different calls share an idempotency key. Not specified; chosen as the option that satisfies Hard Rules 4 and 8. — 2026-09-15
[0.2] Decision: ULID implemented in ids.py rather than taking a dependency; monotonic within a millisecond so effects staged in the same tick still render in stage order. — 2026-09-15
[0.2] Defect found and closed: Python's == says 1 == 1.0 and True == 1, so a dataclass __eq__ would confirm a speculation that predicted {"amount": 1} against an actual {"amount": 1.0}. decisions_equal compares canonical bytes only; a regression test pins the trap. — 2026-09-15
[0.5] Honesty tests: vocabulary, no-LLM-in-control-path, number traceability. Each carries parametrised proofs that it fires on planted violations and stays quiet on legitimate prose. — 2026-09-15
[0.5] Decision: the control-path check parses the AST instead of grepping for `messages=` and `prompt`. A grep cannot tell a protocol signature that forwards the developer's messages from code that authors a prompt, and fires on every docstring Hard Rule 13 needs. The literal `messages=[` grep the spec names is retained alongside it, so the planted-bug check still fires. — 2026-09-15
[0.5] Decision: BUILD_SPEC.md is excluded from the vocabulary scan — it is the input specification, not a claim this project makes, and it necessarily quotes the whole forbidden vocabulary in order to forbid it. — 2026-09-15
[ARCH] Ran a 9-agent design pass over the seven cross-cutting mechanisms before writing runtime code, then audited the result against the 13 Hard Rules. Agents verified CPython 3.11.9 TaskGroup cancellation semantics and LangGraph 1.2.11 task-id stability empirically rather than assuming them. Note in the session scratchpad; key resolutions recorded below as they are implemented. — 2026-09-15
[ARCH] Resolved conflict: dispatch dedupe is keyed on nkey (the Rule 8 key derived with an EMPTY lineage), not on the lineage-bearing key. Keyed on the latter, every stall-and-re-stage and every resume that re-mints branch ids would re-dispatch. nkey is also the idempotency token handed to the tool adapter. — 2026-09-15
[1.1] Effect classes, ToolRegistry, MCP annotation mapping. Undeclared tools classify as WRITE and never raise on get(); running one raises UnknownTool. — 2026-09-15
[1.1] Decision: forward_keys templates are literal text plus {args.<name>} and nothing else — no str.format, no eval. str.format would expose {args.__class__.__init__.__globals__} and would silently quote under !r, so a write declaring ticket:T1 would never intersect a read declaring ticket:'T1' and the overlap check would report a miss. — 2026-09-15
[1.2] Fake world with branch-attributed mutation log, witnessed reads, four injectable faults. A parametrised test disables the mutation funnel and asserts no tool can move state around it. — 2026-09-15
[1.2] Decision: World tools carry their TRUE semantics separately from the developer's DECLARED ToolSpec. The gap between them is the runtime's trust boundary and is where attacks 7.1 and 7.10 live; enqueue_reindex models it directly. — 2026-09-15
[1.2] Decision: reads are recorded when the request is sent, not when it returns, so a speculative read cancelled by a squash still counts as having reached upstream (attack 7.2). — 2026-09-15
[1.2] Decision: added a `charges` table beyond the four the spec names, so Demo 1's double-charge is visible in state and not only in the mutation log. — 2026-09-15
[0.3] Journal: WAL, synchronous=FULL, one fsync per entry, hash-chained per run, dense per-run offsets assigned in Python. Kill test green: 1,000 appends with os._exit at a random point, 15 runs. — 2026-09-15
[0.3] Signature change: Journal.read's `after` defaults to -1, not section 7's 0. Offsets are dense from zero and `after` is exclusive, so the specified default silently skips every run's first entry — which for a replay means losing run_started, the entry the config and registry are checked against. — 2026-09-15
[0.3] Decision: entry_hash is a pure function of the six non-payload columns rather than a stored column, so the chain covers kind/offset/run_id/ts while the table keeps exactly the columns the spec names. — 2026-09-15
[0.3] Defect found and closed: the first draft of verify_chain compared canonical(payload) against canonical(payload) — a check that could never fail. Entry now carries the stored payload_json so byte fidelity is checked against the text actually on disk, which catches a hand-edited journal whose JSON is valid but not canonical. — 2026-09-15
[0.3] Decision: one journal writer per database file per process on a single-threaded executor. SQLite WAL admits one writer; this serialises 20 concurrent runs (task 5.3) in FIFO order rather than scattering SQLITE_BUSY retries through the runtime. — 2026-09-15
[0.4] ModelClient protocol, RequestEnvelope + request_hash (Rule 13), JournaledModel, ReplayModel/ReplayDivergence, ScriptedModel/RecordingModel, Anthropic adapter behind the optional extra. — 2026-09-15
[0.4] Decision: the request hash is taken inside JournaledModel at the wire boundary, not by the caller. If anything between caller and socket mutates the request, a caller-side hash records a clean value for a dirty request and Rule 13 becomes a no-op that still reports zero divergences. — 2026-09-15
[0.4] Decision: provider correlation ids are rewritten to positional tokens (tu:<turn>:<ordinal>) rather than dropped. Dropping them would let a tool result attached to the wrong call hash equal a correct one; hashing them raw would diverge every replay for no reason. — 2026-09-15
[0.4] Decision: Rule 13 tracks role='target' requests only. A draft model's request legitimately differs, and folding it in would raise ContextDivergence on every tier-2 run's first call — whose obvious fix is to loosen the comparison until it stops catching real target-side divergence too. — 2026-09-15
[0.4] Decision: the Anthropic adapter lives in integrations/, not core/, so the Hard Rule 1 control-path scan needs no exemption. Not a reorganisation of the section 6 tree; the spec does not place the provider adapter. — 2026-09-15
[AUDIT] Two adversarial audits of the architecture note returned 38 violations (14 blocking) and 24 gaps. Two touched already-shipped code and were fixed immediately; the rest are being reconciled into a corrections addendum before Phase 2. — 2026-09-15
[AUDIT] Fix: effect_dispatched now requires dispatch_index as well as stage_index. A ledger ordered by stage_index would sort a reversed drain back into stage order and silently pass task 5.1's mandated "dispatch order reversed" planted bug. — 2026-09-15
[AUDIT] Fix: World gained an append-only on-disk mutation log, fsynced per call, and rebuilds state on reopen. Tasks 2.5 and 5.2 SIGKILL a subprocess and compare world.mutations against the uninterrupted run; an in-memory world dies with that process, so the resumed run could not have seen a duplicate dispatch the dead one already made. Chosen over the audit's proposed out-of-process World server: the log achieves the same thing with far less machinery. — 2026-09-15
[AUDIT] Fix: state changes now run inside World._mutate rather than beside it, so the funnel is structural rather than conventional and each record carries the resulting row — which is what lets recovery replay the log without re-running any tool. — 2026-09-15
[AUDIT] Already closed by shipped code: the audit asked that ReplayModel be keyed by step rather than ordered, so it is idempotent and safe when concurrent branches request the same step. It was implemented that way in 0.4. — 2026-09-15
[1.3] Idempotency keys (three derivations from one framed preimage) and the durable dispatch claim protocol. Keys verified stable across processes. — 2026-09-15
[1.3] Decision: the preimage is a canonical JSON object, never a concatenation, because ("ab","c") and ("a","bc") concatenate to identical bytes and two different effects must never share a key. A test pins it. — 2026-09-15
[1.3] Decision: the claim row is the intent record, so no sixteenth journal entry kind is invented for it — a send intent is neither a model output nor a tool result, so Hard Rule 5 does not reach it. — 2026-09-15
[1.3] Decision: an in-flight claim found after a crash is AMBIGUOUS, not retry-safe. Only a failure that demonstrably never left the process is downgraded to retry-safe. Treating the ambiguous case as safe is how a card gets charged twice. — 2026-09-15
[1.4] Store buffer: stage/forward/drain/discard, plus core/branch.py and core/hazards.py. Drain preconditions are re-asserted inside drain(), not trusted from the caller, because 1.6 plants a bug that walks around a call-site check. — 2026-09-15
[1.4] Signature change: StoreBuffer.stage is async. The effect_staged entry is fsynced before the branch is told the effect exists; staging without a durable record would let a resume lose an effect the branch believes it holds and then compute later arguments from it. — 2026-09-15
[1.4] Decision: handle-accepting tools are not implemented. Rules 4, 8 and 9 each independently forbid them, so depends_on is always empty and kept only for the section 7 shape. Recorded so a later reader does not mistake it for an oversight. — 2026-09-15
[1.4] Decision: added a ninth Hazard member, NODE_NOT_SPECULABLE. A predicted route into a node the adapter cannot run speculatively must be named rather than silently not attempted, or 6.2(c)'s histogram is incomplete and the honest answer about available speculation is understated. — 2026-09-15
[1.4] Defect found and closed (from the audit): IRREVERSIBLE_ON_PATH must fire only on a SPECULATIVE branch. Firing it unconditionally makes the canonical path refuse to stage an irreversible call it has already been told to make, so the run livelocks and the shipped send_email example could never send anything. — 2026-09-15
[1.5] Dispatcher: at-least-once, bounded exponential backoff, unjittered by default so a chaos failure reproduces. World faults now subclass ToolDispatchError so a partition reports sent='no' and a timeout sent='maybe'; without that distinction every partition would be treated as ambiguous and the crash-window tests would pass for the wrong reason. — 2026-09-15
[1.5] Decision: a drain halts at the first dead letter rather than skipping it. The effects after it were staged on the assumption it happened, so sending them anyway would put the world in a state no run ever produced. — 2026-09-15
[1.6] THE LEAK TEST green: 500 hypothesis-generated branch trees, random outcomes, random faults, ~8s. — 2026-09-15
[1.6] Decision: the leak test asserts two invariants, not one. The spec's invariant ({mutating branches} subset of {retired branches}) cannot see the planted bug the spec names, because a drain that runs before its confirming entry is fsynced still happens on a branch that is CONFIRMED and does retire. A second invariant checks every dispatched effect against a confirming entry that was durable at dispatch time, and that one does catch it. — 2026-09-15
[1.6] Decision: the vocabulary test caught an unqualified "exactly-once" in the dispatcher's own docstring during this task and the docstring was rewritten. Recording it because it is evidence the honesty checks work on this project's own code, not only on planted examples. — 2026-09-15
[AUDIT2] The corrections addendum was itself verified adversarially; the verification found a real defect in shipped code that no mandatory test could see. Nothing advanced the step cursor per tool call, so two identical calls in one node body derived one idempotency key, dedupe suppressed the second, and an effect the sequential run performed never reached the world — invisible because the leak invariant cannot see a MISSING effect and both equivalence arms collide identically. Branch.advance_step() now takes one position per call and staging a duplicate key is refused. — 2026-09-16
[AUDIT2] Fix: drain no longer snapshots the staged list, and DrainReport reports any effect that got no outcome. A node released by one drain can stage the next write from the value it just received. — 2026-09-16
[AUDIT2] Fix: drain's durability check reads the entry at the claimed offset and checks it is this branch's branch_resolved(confirmed). Comparing the offset to the journal head is satisfied by any later entry, so task 1.6's planted bug walked through it. — 2026-09-16
[AUDIT2] Fix: Journal.max_step takes a branch filter. Counting every entry includes steps consumed by branches whose work was thrown away, so a resume would restart above the committed program position. — 2026-09-16
[2.1] Committed state, COW forks, a hand-written deterministic RFC 6902 differ, four named reducers, and an immutable ContextChain. Differ round-trip and determinism at 1,500 hypothesis examples each, plus a subprocess test that a fresh interpreter derives the same patch hash. — 2026-09-16
[2.1] Decision: no public function in core/state.py takes two branch states, so a merge of two speculative siblings cannot be expressed. A test asserts it structurally rather than leaving it to convention. — 2026-09-16
[2.2] Sequential scheduler over one code path. The canonical branch is minted before the turn (correction C2) so the model request has an owning branch and the ledger stamps context_identity 'unchecked' rather than claiming a check nobody made. — 2026-09-16
[2.2] Defect found and closed: the first scheduler awaited the node task directly, which deadlocks on the ordinary shape ack = await charge_card(...); send_receipt(ack['charge_id']) at the first write of the first sequential run. The scheduler now waits for the task to finish OR to park on a staged write's result, drains, and repeats until the node is done. A single drain pass is not enough: resuming and re-staging takes several event-loop turns because staging awaits a journal append. — 2026-09-16
[2.4] Plain-Python integration: @tool, @node, PlainAdapter. examples/support_agent runs end to end — the read executes, both writes stage and drain in order, and the receipt carries the real charge id rather than a placeholder. — 2026-09-16
[2.3] LangGraph integration by substitution, not by driving. Each compiled node's bound runnable is replaced with a shim that runs the original body inside a branch; LangGraph keeps routing, reducers and map-reduce. Verified against langgraph 1.2.11; probe() fails loudly if the seam moves rather than silently running unwrapped. — 2026-09-16
[2.3] Decision: routed() wraps a declared tool so it calls straight through outside a run and goes through the runtime inside one. That is what lets one graph file satisfy 2.3's 'runs unchanged wrapped and unwrapped' without a second copy of the example. — 2026-09-16
[2.3] Decision: the node shim runs the body as a task rather than wrapping it in a context manager. A node parked on a staged write must retire while its body is still suspended; an 'async with' around the body only reaches its exit after the body finishes, which is the deadlock again. — 2026-09-16
[2.3] Defect found and closed: the drain guard required the scheduler's own task, which is wrong for a framework that owns its run loop. The invariant is narrower and better expressed against the branch: a node may never dispatch the write it is waiting on. — 2026-09-16
[2.3] Defect found and closed: wrap() minted a run id for the store buffer while run() minted another for the scheduler, so effects were journaled under one run and the ledger built from another. The run reported ok with an empty ledger and a changed world. Scheduler.run is now the single source of the run id, with a regression test. — 2026-09-16
[2.5] Resume: recover() rebuilds committed state and the program cursor from branches the journal records as RETIRED, and the dedupe table stops anything already acked from going out again. Kill/resume green across 15 random points, 12 genuinely killed, 0 duplicate deliveries. — 2026-09-16
[2.5] Defect found and closed: the cursor was inferred from the highest step visible. Inferring lands a resumed run at a different program position, so every idempotency key it derives differs from its pre-crash value, the dedupe table misses, and effects that already went out go out again. The cursor is now journaled verbatim as cursor_after on each retirement and restored from there. — 2026-09-16
[2.5] Defect found and closed (ordering): branch_resolved{retired} was journaled BEFORE state_delta_applied. A crash between them left the branch reading as retired while its state change was lost, so the resume re-ran the node from a different position and re-charged the card. Reproduced at a specific kill delay, fixed, and verified over 40 kill points with zero duplications. A regression test asserts the ordering directly rather than leaving the chaos test to rediscover it. — 2026-09-16
[2.5] Deviation from the Verify's literal wording, with reason: the spec asks that a resumed run's effects equal the uninterrupted run's. That holds at every kill point except one class. If the process dies between a request reaching the world and its ack being recorded, nobody can tell afterwards whether it took effect; charge_card is declared idempotent=False, so the runtime dead-letters it rather than risk a second charge, and the run reaches a prefix of the clean run's effects. The test asserts the two properties that actually matter — never duplicated, never invented (always a prefix, in order) — and requires a dead letter whenever it falls short. Declaring the tool idempotent would make the literal equality hold by redelivering a charge that may already have gone through. — 2026-09-16
[2.6] CLI: init, runs, ledger (with --json and --normalised), verify-ledger, sign-ledger, verify, status. Progress Log entry for the subcommands beyond the section 6 list: runs, sign-ledger, verify and status are read-only views over the journal. — 2026-09-16
[C1] attested_origins implemented: the retired chain unioned with the branch's own lineage. Lineage resets at every retirement, so a lineage-only filter hides every earlier assistant turn and the Rule 13 rebuild would diverge on every step. The tempting repair once that is seen — admit anything not squashed — admits an unresolved sibling, which is the exact Hard Rule 6 channel, so the test is membership rather than absence from a blacklist. — 2026-09-16
[3.1] Tier-0 early issue: each tool_use block is issued the instant it parses out of the target's stream. Verified on the clock, not on the design — with a 3-block turn streaming over ~300ms, the READ completes before the stream ends, and the WRITEs are journaled staged only after the turn's model_response entry. — 2026-09-16
[3.1] Decision: writes emitted mid-stream are staged after the turn completes rather than as they parse, so a staged effect never exists for a turn the journal does not yet record. Reads are issued mid-stream, which is where the saving is. — 2026-09-16
[3.2] Tier-1 pattern index (PASTE's mechanism, credited in the module docstring and README). Order-k over tool signatures — name plus argument shape, values dropped, or every trace would be unique. Mined from RETIRED lineages only: a squashed branch's calls are what the run decided not to do. Saved as canonical bytes so two builds are byte-identical. — 2026-09-16
[3.2] Decision: argument templates fill from prior tool RESULTS as well as prior arguments. Without that the index predicts send_receipt correctly and then cannot fill charge_id — which only exists in charge_card's output — so it offers nothing and the whole tier is dead. PASTE's JSONPath generality is not implemented; a keyed lookup two levels deep is. — 2026-09-16
[3.4] The gate: exact canonical equality, with FreeText never confirming even against itself. A branch that predicted nothing is refused rather than confirmed — the canonical branch is confirmed by the turn it owns, and a gate that confirms a branch with no prediction confirms anything. — 2026-09-16
[3.5] Witness validation at retirement (E3). Only reads issued while the branch was speculative are re-checked; a read a CONFIRMED branch made happened after the decision was durable, and re-checking it livelocks the sequential arm under a competing writer. Unwitnessed reads are reported as such, never counted fresh. The re-fetch costs a real upstream call and is counted. — 2026-09-16
[3.5] Decision: read arguments live on the ReadRecord rather than in a module-level cache keyed by hash. The cache would outlive the run, grow without bound, and let one run's arguments answer another run's question. — 2026-09-16
[3.6] Budgets and the rolling acceptance rate. Tier 0 is counted separately and excluded from the gate's input: its acceptance is 1 by construction, so averaging it in would hold the rate above any floor however badly the real predictors were doing, and the gate would never fire. A partial window reports no rate at all — disabling speculation on three samples is a worse error than speculating three more times. — 2026-09-16
[3.8] Context identity: check_structural is total and never waived (no placeholder, every message attested, tool results in program order), and fold_context rebuilds from the journal rather than from the branch's own message list — which would compare that list to itself and pass every time. — 2026-09-16
[5.1] THE EQUIVALENCE TEST green over a real workload, with expect_rows so an empty comparison cannot pass and a world join so two ledgers cannot agree with each other while both disagree with the world. — 2026-09-16
[5.5] THE CONTEXT-EQUIVALENCE TEST green. It asserts on what a RecordingModel RECEIVED, never on what the runtime says it sent: the live check and the retirement rebuild share a prompt builder and can be wrong identically. Includes the two-turn write-barrier variant, which a 'the next call only' flag would pass and be wrong about. — 2026-09-16
[DEMO1] Demo 1 runs and produces exactly the table the spec predicted, from measured values: naive-parallel 60 charges with 10 from squashed branches, specunode 50 with 0, sequential 50 with 0. Committed to bench/results/demo_leak.json, with a test asserting the file still matches a fresh run. — 2026-09-16
[6.3] Baselines: B_seq, B_readonly_spec, B_naive_parallel, B_specunode. The naive arm must be able to leak or the demo measures nothing, so bench/baselines.py imports nothing from specunode.buffer. — 2026-09-16
[9.1] README written with the demo's measured table, prior art credited by name (PASTE, Claude Code's streaming executor, langchain-nvidia, ToolAhead, SagaLLM, ATP, SCOPEGATE, Temporal/DBOS/Restate, Tomasulo), 'What beats it' before the wins, and 'What this is not'. No latency figure appears anywhere, because none has been measured. — 2026-09-16
[9.1] Two detector refinements while writing it, both the checker working: a decade ('the 1960s') is not a measurement and is now excluded; and a [cited] marker that had wrapped onto the next line was not attached to its number, so the README was rewrapped rather than the check loosened. — 2026-09-16
[3.3] Real branch speculation: while the model streams block j, the predictor guesses block j+1 and a branch is forked to run it. A confirmed guess is adopted — the call is not made twice, which is the latency win — and a wrong one is squashed with its buffer discarded unsent. Tested both ways, including that the world ends up identical whichever the drafter guessed. — 2026-09-16
[3.7] Cancellation is the squash primitive, and revocation happens BEFORE cancellation: a tool that cannot be cancelled finishes anyway, and a closed buffer is what stops its write being staged into something nothing will ever drain. — 2026-09-16
[3.9] Read budget enforced through the same hazard path, with the exhausted budget named rather than reported as a generic stop. — 2026-09-16
[7.x] Adversarial suite: 8 strategies, all of which beat the runtime, each with a measured rate and a stated claim. A strategy that cannot run becomes an error row rather than a dropped row — a suite that silently drops what it cannot execute reports a clean sheet for the wrong reason. Committed to bench/results/attacks.json. — 2026-09-16
[7.5] Measured, not asserted: 6 of 9 laundering transforms are caught, 3 evade (base64, hex, and a split *inside* the handle prefix). A split *after* the prefix leaves it intact and is caught — the first version of this case was mislabelled as a miss, and testing both splits is what surfaced it. A test asserts the hazards doc's table agrees with the measurement. — 2026-09-16
[7.x] Decision: a test asserts each known hole STILL defeats the runtime. If one stops, that is a human decision — either a real hole closed and docs/limitations.md should drop it, or the attack stopped exercising what it claims. Publishing a hole that no longer exists is as dishonest as hiding one that does. — 2026-09-16
[5.2] Chaos matrix: partitions mid-drain, duplicate deliveries, slow reads outliving their branch. 0 leaks, 0 duplicates, 4 dead letters across 12 rounds — the dead letters are the partition rounds behaving correctly, and a round count of zero would mean no fault fired. — 2026-09-16
[5.3] Concurrency: 20 runs through one journal with separate worlds produce ledgers identical to a solo run; 20 runs through one journal AND one world produce no leaks, no duplicates and no cross-branch observations. — 2026-09-16
[5.3] Defect found in the MEASUREMENT, not the runtime: the first version compared each run's branches against every run's mutations in a shared world, and compared interleaved ledgers to a solo run when the runs genuinely interact (each charge mints its own id). It reported 380 leaks and 19 equivalence failures, none of which were real. The two questions — does sharing a journal disturb a run, and is every effect attributed to its own run — are now measured separately. — 2026-09-16
[5.4] verify-ledger distinguishes integrity (rows edited, or not reproducing from the journal) from origin (valid signature, unknown key), with tests for both forgeries and for the genuine case. — 2026-09-16
[6.1] Decision Gate D1 does NOT fire: nebius/SWE-rebench-openhands-trajectories is reachable through the HF datasets-server rows API, and 300 real trajectories (19,484 tool calls) were fetched and normalised. Only the normalised form is committed — tool name, argument keys, whether an argument references a prior result, and the turn structure — with a manifest hash a test verifies. — 2026-09-16
[6.1] Decision: the corpus effect-class table is hand-written per tool name and str_replace_editor is a WRITE, although its `view` command reads. Classifying by inspecting the command argument would be exactly the argument-level heuristic Hard Rule 2 forbids, and the runtime could not do it either without a declaration. task_tracker is probably a scratchpad and is left WRITE, because 'probably' is the inference the table exists to avoid. — 2026-09-16
[6.2] THE HEADLINE ANTI-RESULT, measured: on this corpus, running ahead past a write buys NOTHING. The span is 0.0000 with a bootstrap interval that does not move off zero, because every tool call opens a new model turn (measured 1.0000) and a staged write blocks the next turn. 95.5% of the corpus is the model->write->model shape spec section 1 predicts gains nothing. — 2026-09-16
[6.2] The non-zero half, reported beside it: PASTE can speculate 4.5% of steps (the reads); SpecuNode can stage the other 95.5%. That is an upper bound on opportunity, not a speedup, realisable only where the predictor is right — measured at 53.5% top-1 and 83.7% top-3. Neither number is ever quoted alone. — 2026-09-16
[6.6] RESULTS.md is generated by bench/report.py and regenerates identically; a missing results file produces an explicit 'not measured yet' section rather than a silent omission, because an absent measurement and a measurement of zero are different things. — 2026-09-16
[9.1] README leads with the negative half of the measured result, before any claim about what works. No wall-clock figure appears anywhere, because none has been measured. — 2026-09-16
[7.7] Drafter poisoning: an index trained on an adversarial strong chain that ends in a charge made 8 predictions, all 8 squashed, 2,000 wasted tokens, the alpha gate disabled speculation, and 0 effects reached the world. The cost is tokens and stalls; it is not a leak. — 2026-09-16
[7.7] The first version measured 0 predictions, because the poisoned chain's arguments were not fillable from history and the drafter correctly declined to offer it. A strong chain has to carry its arguments forward or there is no attack — fixing the fixture is what made the attack real. — 2026-09-16
[7.8] Replay under model drift: a one-token system prompt change and an added tool both diverge at step 0, as the spec predicts. Held, not beaten. — 2026-09-16
[6.5] Overhead measured: 6.645 ms per step (19.934 ms per run) against the same LangGraph app running bare. That is 87.7% of wall clock here, and the percentage is the misleading half — a scripted model answers instantly, so this is the worst case for the ratio. The absolute per-step figure is the one that transfers to a real multi-second turn. Dominated by the journal's one-fsync-per-entry discipline, which is the cost of Hard Rule 5 and is not being optimised away. — 2026-09-16
[8.4] Adapter contract documented and enforced: every bundled tool is checked for cancellability, a canonical-form result, a witness when it claims one, absorbing a repeat delivery when it claims idempotence, and reporting whether a failed request left the process. None of those fail loudly on their own, which is why they are a suite rather than a doc. — 2026-09-16
[8.5] Postgres backend written: schema.sql shared verbatim, DML written once with :name parameters and rewritten once at import for psycopg. BLOCKER-ADJACENT: no Postgres or Docker is available in this environment, so the Postgres path is UNVERIFIED locally. The tests are gated on SPECUNODE_TEST_POSTGRES_DSN and CI runs a Postgres 16 service; until that CI job runs green, treat the Postgres journal as untested. — 2026-09-16
[8.3] Sub-graphs: a node whose bound runnable is itself a compiled graph is recursed into rather than wrapped. Wrapping the container would attribute every inner effect to one branch, and the leak invariant would be meaningless inside a sub-graph — a wrong inner effect could not be told from a right one. Inner nodes carry the parent's path, so two sub-graphs with a node of the same name derive different idempotency keys. — 2026-09-16
[8.2] astream yields only what reached the world, after each retirement. A speculative branch's output may be squashed, and a user's screen cannot be un-written. — 2026-09-16
[8.1] Interrupts: a speculative branch does not run node bodies unless the node opted in, so an interrupt cannot reach one by default. The opt-in is explicit and per node; a test asserts the default is empty. — 2026-09-16
[4.x] MCP proxy: the staging rules live in ProxyState and are tested without a transport (20 tests); the stdio transport is a thin layer over them. The split is deliberate — the rules carry the correctness claims and the SDK does not, and mcp went 1.x -> 2.x with a breaking server API change during this build. The proxy probes on startup and fails loudly rather than degrading into something that forwards writes it was meant to hold. — 2026-09-16
[4.2] Decision (this was flagged as needing a human, and is resolved mechanically instead): the proxy returns a staged-write handle ONLY to a client that advertised the specunode/decisions capability. Any other client blocks until a decision arrives. Handing a placeholder to a client that does not understand it puts that placeholder in the next prompt, which Hard Rule 13 forbids and the proxy cannot see to prevent. The mode is read from what the client advertised, not from a default. — 2026-09-16
[4.x] Every proxy run is stamped context_identity: unenforced. The proxy never sees a prompt, so it cannot check one, and a stamp for a property nobody checked is worse than no stamp. — 2026-09-16
[4.x] NOT VERIFIED END TO END: the transport has not been driven by a real MCP client against a real upstream server in this environment. Phase Gate 4 (a generic client producing the same ledger as the LangGraph integration) is therefore NOT met and 4.x is ticked for the rules and the wiring, not for that gate. — 2026-09-16
[9.2] All six docs written (effect-classes, hazards, limitations, replay, adapters, mcp-proxy); specunode init writes the config and .specunode/ and was run to check it. — 2026-09-16
[9.4] bench/make_report_pdf.py regenerates the report from bench/results/*.json. A test asserts the generator's own source contains no hard-coded figure, because a number typed into the generator would survive a change in what was measured and the provenance link would break silently. The wall-clock section renders 'not measured' rather than being omitted. — 2026-09-16
[T2] Tier-2 draft model shipped behind the optional extra. It holds the only prompt in the package, and it lives in drafters/ rather than the control path — a drafter may hold one because everything it produces is a candidate the gate must still confirm by exact equality. A test greps the four control packages for that prompt text. — 2026-09-16
[T2] Decision: a draft model that is down, rate-limited or slow returns no opinion rather than failing the run. The sequential path is always correct, so losing a speculation is not an error; a drafter that could fail a run would make speculation a liability rather than an optimisation. — 2026-09-16
[0.4] Decision: CallScope is carried in a ContextVar rather than passed as an argument, so JournaledModel satisfies ModelClient and can be substituted wherever the developer's graph already calls a model — which is what lets task 2.3 leave their graph file unchanged. — 2026-09-15
[5.x] examples/ops_agent and examples/research_agent built, so the three sample apps section 6 names all exist. Deliberately different shapes: support_agent is the explicit complete()+call_tool() pattern Demo 1 uses, ops_agent hands its turn to the runtime via call_turn with several calls in it, research_agent is read-heavy with an IRREVERSIBLE barrier. — 2026-09-16
[5.x] bench/workloads/ now holds the workload registry the three mandatory tests iterate, so "which workloads exist" is written down once. Task 5.1's wording ("every workload in bench/workloads/") is satisfied literally rather than by a list copied into each test. — 2026-09-16
[5.x] Decision: each workload declares expect_effects, drives_turn and tier_1_can_predict. The last two are expectations about what the runtime will and will not do on that shape, asserted rather than assumed — without them a workload that silently stopped speculating would still pass every comparison, because two runs that never speculate are trivially equivalent. — 2026-09-16
[5.x] Defect found: the three mandatory tests ran on one workload at one tier. Parameterising them over three workloads x two tiers is what surfaced the three defects below; none of them was reachable from the previous coverage. — 2026-09-16
[3.2] DEFECT (BLOCKING, fixed): a tier-1 prediction of a WRITE that the model then confirmed deadlocked the run forever. The child branch staged the effect, the canonical branch adopted the child's task and awaited its ack, and nothing ever drained the child's buffer — drain dispatches by branch id and only the canonical branch retires. Fixed with StoreBuffer.adopt(), which moves a confirmed speculation's staged effects onto the branch that will retire, re-attributing branch_id and lineage but never nkey (the token the tool sees). This is the case the whole project is named for — speculating past a write and being right — and it hung. — 2026-09-16
[3.2] Why that defect survived everything: the speculation suite's stub drafter predicts a READ for its confirm case (reads stage nothing) and a WRITE only for its squash cases (squashed buffers are discarded, never drained). No test had ever confirmed a prediction of a write. Regression test added at tests/integration/test_t1_end_to_end.py, asserted against the world rather than the ledger. — 2026-09-16
[3.2] DEFECT (fixed): park events are keyed by branch id, so the event the child set when it staged was on a key nothing waits on. Adoption now signals the canonical branch's park event; without it the effect moved to the right list and still never left. — 2026-09-16
[5.1] DEFECT (Hard Rule 9 violation, fixed): a confirmed speculation did not advance the canonical branch's step cursor, so every later call in the run derived a different idempotency key depending on whether the runtime happened to speculate. A resume with speculation off would not dedupe against a crashed run that had it on, and the effect would be delivered twice. Caught by the equivalence test the moment a workload confirmed a prediction — the ledgers differed at reserve_capacity's key and authorised_by_step. — 2026-09-16
[3.x] DEFECT (fixed): branch_forked journaled predicted: None, predicted_hash: "" and tier: None for every fork, including speculative ones, so the durable record could not distinguish a predicted branch from the canonical one and "how much did this run speculate" was unanswerable from the journal. Now recorded from the branch. — 2026-09-16
[3.x] DEFECT (fixed): a confirmed speculative branch was never journaled as resolved at all — only squashed ones were — leaving its lifecycle open in the durable record. Now journaled as branch_resolved{status: confirmed, adopted_by}. — 2026-09-16
[3.2] Finding, not a defect: a drafter cannot use the result of the call it was just asked about. It is consulted immediately after a tool_use block parses, when that block's call has only been issued, so the earliest usable result is from a block two back. This halves the reach of PASTE's data-flow idea inside this runtime. Waiting for the read before asking would serialise exactly what early issue exists to overlap. Documented in docs/limitations.md. — 2026-09-16
[3.2] Finding, not a defect: a one-call-per-turn workload offers the drafter nothing to predict from, because its history is the calls within the current turn. That is 1.0000 of the offline corpus and the same fact as the 0.0000 speculable span, seen from the runtime side. Two of the three sample apps keep that shape on purpose. — 2026-09-16
[2.3] Finding: a node that calls session.model.complete() and then session.call_tool() routes around tier-0 early issue and the drafters entirely — they live inside the turn the runtime drives, reached via session.call_turn. Nothing warns about it; the run is simply sequential. The equivalence test now asserts "never asked" and "asked and declined" separately so the two cannot be confused. — 2026-09-16
[0.x] Decision: pytest's pythonpath is set to the repo root. Without it the mandatory tests collected under a bare `pytest` and failed to import under `pytest tests/test_equivalence.py` — the invocation someone debugging one of them would reach for. — 2026-09-16
[0.x] Test-quality fix: test_there_are_exactly_fifteen_entry_kinds asserted a constant that breaks whenever a kind is added and says nothing about whether the new kind works. Replaced with the property that actually matters — no duplicate kinds, and every kind declares the fields append will demand. — 2026-09-16
[3.2] tests/unit/test_t1_drafter.py added: the tier-1 predictor had no tests of its own. The speculation machinery was exercised through a stub returning a fixed answer, which proves the runtime handles a prediction and nothing about the thing that makes them. — 2026-09-16
[5.2] The leak test now also runs the real scheduler on the real workloads at both tiers, including a deliberately mistrained index so that a genuinely wrong prediction is squashed. The hypothesis suite above it proves Rule 3 about the simulator; these prove it about what ships. — 2026-09-16
[9.1] README Status section corrected: it still said the MCP proxy, the benchmarks, the adversarial suite and RESULTS.md did not exist. All four do. Replaced with the four gaps that are actually open. — 2026-09-16
```

---

## Final Report

**Written 2026-09-16, after 64 of 71 tasks. The seven that are not done are listed below with
the reason each one is blocked, and none of them is blocked on more work I could do here.**

**666 tests pass, 27 skip. `ruff` and the configured `mypy --strict` are clean.**

### What was built, in five sentences

SpecuNode executes an agent graph the way an out-of-order CPU executes instructions: reads
issue early, writes wait in a branch-scoped store buffer, and the target model's real decision
is the only thing that can release a write. Every model output and tool result is journaled and
fsynced before the runtime acts on it, so a crashed run resumes and a finished run replays, and
the replay refuses the moment the run would ask the model a different question. Three
never-skipped tests hold the invariants: nothing reaches the world from a branch that did not
retire, the effect ledger with speculation on equals the ledger with it off, and the speculative
arm asked the model the same questions as the sequential arm. It ships a LangGraph integration
that runs an unchanged graph file, a plain-Python API, an MCP proxy, three drafter tiers, and a
benchmark suite whose numbers are all read from committed files. The most useful thing it
produced is a negative result.

### The headline numbers, with the commands that produced them

```
python bench/corpus/fetch.py
python bench/offline/run_opportunity.py --out bench/results/opportunity.json
```

Measured on 300 real OpenHands trajectories from `nebius/SWE-rebench-openhands-trajectories`,
19,484 tool calls. Means with 95% percentile-bootstrap intervals over trajectories.

| Measure | Value |
|---|---|
| Reads — the whole of what PASTE can speculate | 0.0445 [0.0423, 0.0469] |
| **SpecuNode speculable span past a write** | **0.0000 [0.0000, 0.0000]** |
| Steps PASTE must skip that SpecuNode can stage | 0.9555 [0.9531, 0.9577] |
| Calls that open a new model turn | 1.0000 [1.0000, 1.0000] |
| T1 predictability, leave-one-trajectory-out | top-1 0.5350, top-3 0.8369 |

**The second row is the headline and it is zero.** Every tool call in that corpus opens a new
model turn, and a staged write blocks the next *turn* because that turn would have to contain a
placeholder where the real result belongs. There is nothing to run ahead into. The mechanism
this project is named for buys nothing on the only real public corpus available.

The third row is the part that is not zero, and it is a different quantity: PASTE refuses to
speculate on a tool with side effects at all, while SpecuNode stages one, so a *predicted* write
can be run ahead and discarded. That is an upper bound on opportunity rather than a speedup, and
it is realisable only where the predictor is right. The two numbers are never quoted apart.

**The second headline number does not exist.** There is no wall-clock reduction figure anywhere
in this repository, because the online latency benchmark has not been run — see the manual steps.

### The anti-results

- **Past-write speculation: 0.0000 span**, as above. 95.5% of the corpus is the
  `model → write → model(reads the result)` shape that section 1 predicts gains nothing from it.
- **Break-even α: not measured.** It is defined as the α at which the speculative arm's wall
  clock equals the sequential arm's, and wall clock has not been measured. `alpha_floor` defaults
  to `None`, which means the gate is inactive rather than set to a guessed number.
- **Undetectable stale reads: 0.5** of the stale reads in attack 7.3's fixture were unwitnessed
  and therefore undetectable. That fraction, not the stale rate, is the honest number.
- **Journaling and classification overhead: 6.645 ms per step.** That is 87.7% of wall clock in
  the measurement, and the percentage is the misleading half — a scripted model answers
  instantly, so it is the worst possible ratio. The absolute per-step figure is what transfers.

### The attacks that beat it

Ten strategies run; eight defeat the runtime.

| # | Strategy | Measured |
|---|---|---|
| 7.1 | A tool declared READ that writes | 1 effect from a squashed branch; leak rate 1.0 |
| 7.2 | A read with upstream side effects | 5 of 10 reads charged to squashed branches |
| 7.3 | Stale reads under contention | 0.5 of stale reads undetectable |
| 7.4 | Duplicate delivery, non-idempotent tool | 1 intended, 2 delivered |
| 7.5 | Return-value laundering | 6 of 9 caught; miss rate 0.333 (base64, hex, split inside the prefix) |
| 7.6 | Prompt-injected tool call | dispatched when the model emits it; not an authorization layer |
| 7.9 | `stage_irreversible=true` | an effect with no undo released by machinery |
| 7.10 | Async side effect behind a READ | 1 leaked effect per squashed branch |

Held: 7.7 drafter poisoning (8 predictions, 8 squashed, 2,000 wasted tokens, 0 leaks) and 7.8
replay under model drift (both cases diverge at step 0).

### Decisions this spec did not specify

The Progress Log above has 100+ entries; these are the ones that changed the shape of the build.

1. **`Journal.read`'s `after` defaults to −1, not §7's 0.** Offsets are dense from zero and
   `after` is exclusive, so the specified default silently skips every run's first entry.
2. **`StoreBuffer.stage` is async.** The `effect_staged` entry is fsynced before the branch is
   told the effect exists.
3. **Three key derivations, not one.** `key` carries the branch lineage and stays internal;
   `nkey` drops it and is both the dedupe primary key and the token the tool sees; `ekey` drops
   the run too and is used only by the equivalence relation. Deduping on the lineage-bearing key
   re-dispatches after every stall-and-re-stage and every resume.
4. **A staged write always returns a future, never a value.** Handing back a real ack means
   dispatching inside the call, which is task 1.6's planted bug; making the caller await the
   drain deadlocks the first write of the first sequential run.
5. **The leak test asserts two invariants.** The spec's own invariant cannot see the bug the
   spec names as the planted one, because in a single process a drain before its confirming
   entry is durable still happens on a branch that retires.
6. **Hard Rule 1's check parses the AST.** A grep cannot tell a protocol signature that forwards
   the developer's messages from code that authors a prompt, and fires on every docstring Rule 13
   needs. The literal `messages=[` grep the spec names is kept alongside it.
7. **Projections are not implemented at all.** A projection retires unverified and can make Hard
   Rule 9's mandatory comparison fail in a supported configuration, and Rule 9 has no policy
   escape clause.
8. **Handle-accepting tools are not implemented.** Rules 4, 8 and 9 each independently forbid
   them.
9. **A ninth hazard, `NODE_NOT_SPECULABLE`.** A refusal that is not named is missing from the
   histogram, and the honest answer about available speculation is understated.
10. **Task 2.5's Verify is satisfied in a weaker, truer form.** A resumed run's effects equal the
    uninterrupted run's at every kill point except one class: if a process dies between a request
    reaching the world and its ack being recorded, a non-idempotent tool is dead-lettered rather
    than redelivered, so the run reaches a *prefix*. The test asserts never-duplicated and
    never-invented, and requires a dead letter whenever it falls short.
11. **The MCP proxy's mode is read from the client's advertised capability**, not from a default.
    A client that cannot be told "this has not happened" will put the placeholder in its next
    prompt, and the proxy cannot see prompts.
12. **The corpus effect-class table is hand-written and `str_replace_editor` is a WRITE**, though
    its `view` command reads. Classifying by inspecting the `command` argument is the
    argument-level heuristic Hard Rule 2 forbids.
13. **The three sample apps are deliberately three different shapes**, not three instances of
    one. `support_agent` calls the model directly and issues each tool itself (Demo 1's
    pattern); `ops_agent` hands its whole turn to the runtime via `session.call_turn` and emits
    several calls in it; `research_agent` is read-heavy and its write is irreversible and
    therefore a barrier. A suite where every workload took the same path would test one path
    three times.
14. **Each workload declares `drives_turn` and `tier_1_can_predict`**, and the tests assert
    them. These are expectations about what the runtime will and will not do on that shape.
    Without them, a workload that silently stopped speculating would still pass every
    comparison, because two runs that never speculate are trivially equivalent.
15. **A confirmed speculation's effects are adopted by the canonical branch** rather than the
    child being retired. Only one branch retires, and re-attributing the effect is what makes
    the run identical to the one that never speculated — which is Hard Rule 9 restated.

### What the expanded coverage found

The three mandatory tests originally ran on one workload at one tier. Parameterising them over
three workloads and two drafter tiers was the last substantive work done here, and it surfaced
four defects. None was reachable from the previous coverage, and one was fatal.

**A confirmed prediction of a write deadlocked the run.** The child branch staged the effect,
the canonical branch adopted the child's task and awaited its ack, and nothing drained the
child's buffer — the drain dispatches by branch id and only the canonical branch retires. The
run hung rather than failing, which is the worst of the three outcomes because nothing reports
it. This is the case the project is named for. It survived because no test had ever confirmed a
prediction of a *write*: the stub drafter predicts a read for its confirm case and a write only
for its squash cases, and reads stage nothing while squashed buffers are discarded, never
drained. Fixed with `StoreBuffer.adopt`.

**Speculating changed the idempotency keys.** A confirmed speculation did not advance the
canonical branch's step cursor, so every later call in the run derived a different key
depending on whether the runtime happened to speculate. A resume with speculation off would
not have deduped against a crashed run that had it on, and the effect would have been delivered
twice. That is a Hard Rule 9 violation, and the equivalence test caught it the instant a
workload confirmed a prediction — which had never happened before.

**Park events are keyed by branch id**, so the event the child set when it staged was on a key
nothing waits on. Found while fixing the first defect; without it the effect moved to the
correct list and still never left.

**The journal could not tell a predicted branch from the canonical one.** Every `branch_forked`
entry recorded `predicted: null` and `tier: null`, and a confirmed speculative branch was never
journaled as resolved at all. "How much did this run actually speculate" was unanswerable from
the durable record, which is the only record Hard Rule 12 permits an answer to come from.

Two further things are findings rather than defects, and are in `docs/limitations.md`:

- **A drafter cannot use the result of the call it was just asked about.** It is consulted
  immediately after a block parses, when that block's call has only been issued. The earliest
  usable result is from a block two back. This halves the reach of PASTE's data-flow idea
  inside this runtime, and waiting for the read before asking would serialise exactly what
  early issue exists to overlap.
- **A one-call-per-turn workload offers the drafter nothing**, because its history is the calls
  within the current turn. That is 1.0000 of the offline corpus, and it is the same fact as the
  0.0000 speculable span seen from the runtime side rather than from the trace.

### Definition of Done: what is not ticked, and why

| Item | Status |
|---|---|
| Wheel on 3.11/3.12/3.13, macOS **and Ubuntu** | Verified on macOS for all three; Ubuntu is CI-only and CI has not been run. |
| The three tests on every workload, every tier, **every CI job** | They run, are never skipped, and now cover all three sample apps at tiers 0 and 1. Tier 2 is a draft *model* behind an optional extra: requiring it here would make a mandatory test skippable, which is the one property these files may never have, so it is covered by its own tests instead. The open clause is **every CI job** — CI has not been run. |
| LangGraph ✓, plain ✓, **MCP proxy with a generic client** | The proxy's rules are tested (20 tests) and the stdio transport is wired, but it has not been driven by a real client against a real upstream server. Phase Gate 4 is **not** met. |
| Offline ✓, overhead ✓, adversarial ✓, **online latency** | Needs an API key. Not run. |
| Published to PyPI; demoed from the published wheel | Not done. Needs credentials. |

### Manual steps left for you

1. **An Anthropic API key and a spend cap**, for the online latency benchmark (task 6.4). Set
   `ANTHROPIC_API_KEY` and `SPECUNODE_BENCH_BUDGET_USD` (default 25). Decision Gate D2 says to
   report the reduced *n* and its wider interval rather than raising the cap, and the runner is
   written to do that. Until this runs, there is no wall-clock number and the README says so.
2. **PyPI credentials**, for task 9.3. `uv build` works and the wheel installs and runs on 3.11,
   3.12 and 3.13 locally; publishing and the clean-venv install from PyPI are yours.
3. **Run CI once.** The Ubuntu matrix, the Postgres 16 job and the extras matrix have never
   executed. The Postgres backend in particular is **written and type-checked but never run** —
   this machine has no Postgres and no Docker.
4. **Decide Phase Gate 4.** "The support example driven by a generic MCP client produces the same
   ledger as the LangGraph integration" cannot hold while `node_id` is a mandatory key input and a
   generic client reports no node. Either accept the node-insensitive comparison, or accept that
   the gate passes only for clients that report node ids.

Decision Gate D1 did **not** fire: the corpus was fetched from Hugging Face, so the opportunity
analysis is on real trajectories rather than self-generated ones.

### What I would do differently with another month

**Find a corpus where the mechanism can work.** The measured zero is a real finding, but it is a
finding about OpenHands' one-call-per-turn shape rather than about agents in general. A workload
that emits several tool calls per turn — a parallel-fanout agent, a batch-of-reads planner — is
where past-write speculation has room, and I would go looking for one and publish both.

**Measure wall clock against a real model.** Every latency claim in this design is currently an
argument. The overhead number says 6.6 ms per step; whether that is noise or a tax depends
entirely on numbers that need an API key.

**Drive the MCP proxy end to end.** The rules are tested and the transport is wired, but "wired"
and "works" are different words and only one of them has been demonstrated.

**Widen the coverage matrix again, and expect it to find more.** Going from one workload at one
tier to three workloads at two tiers found four defects in an afternoon, one of them fatal to
the project's central claim. That is not a comfortable ratio, and the honest reading is that the
next axis — more tiers, more fault injection, more turn shapes — probably has more in it. Demo 2
was still not built, because `ops_agent` only reached its final shape at the end.

**Spend the time on the predictor, not the buffer.** The store buffer works and its guarantees
hold under every fault I could inject. The number that decides whether any of it pays for itself
is the acceptance rate, measured here at 0.5350 top-1 — and that is a prediction problem, not a
runtime one.
