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
#   tool_result, branch_forked, branch_resolved, group_forked, effect_staged,
#   effect_dispatched, effect_dead_lettered, effect_discarded, state_delta_applied,
#   read_validated, policy_event, run_finished

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
- [x] **3.9 Read budget.** `max_speculative_reads` per run, charged for reads made on a forked guess; exhausted → the gate closes for the rest of the run and journals `speculation_disabled{max_speculative_reads}` (not a per-branch hazard: the scheduler consults the gate before it asks the drafter, so a `READ_BUDGET` stall was unreachable and the clause has been removed). The ledger reports `speculative_reads_upstream` — every read that reached upstream without a durable decision, early issue included — and `speculative_reads_charged` beside it.
  *Verify:* a drafter that always guesses reads is capped at the budget and the ledger count equals `world` read calls attributed to squashed branches.

**Phase Gate 3:** leak test still green with speculation on; context-identity check stalls a model call after a staged write; T0/T1 drafters produce measured (not asserted) α on the examples; stale-read squash works; budget gate disables speculation on a hostile script.

---

## PHASE 4 — MCP proxy

Goal: a developer who cannot change their agent's code still gets the store buffer.

- [x] **4.1 Proxy skeleton.** `specunode mcp-proxy --upstream "<cmd>" --config specunode.yaml`; stdio; forwards `initialize`, `tools/list` (annotations passed through, overrides applied), and everything not tool-related unchanged.
  *Verify:* a reference MCP client lists tools through the proxy and sees the upstream's list with its annotations passed through unchanged; the proxy classifies each tool from those annotations with the config's overrides winning, and the end-to-end test proves a tool the upstream marks `readOnlyHint: true`, with no override anywhere, is forwarded as a read.
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
  *Status, 2026-09-23:* run once — 270 runs, $1.64 — and two faults were found in its arms afterwards: `B_seq` already issued reads early, so it was not the sequential baseline it was labelled as, and `B_readonly_spec` was handed no predictor, so it never guessed. Both are fixed and tested (`B_strict_seq` is the new baseline; both guessing arms get the same drafter). The table is not quoted until it is re-run. The corrected comparison is 6.8's 0 ms rung, and the break-even α is 6.9.
- [x] **6.5 Overhead.** Journaling + classification overhead of `B_seq` vs the same graph on vanilla LangGraph with no SpecuNode, same `ReplayModel`. Reported as absolute ms per step and as a fraction of wall clock.
  *Verify:* `bench/results/overhead.json`.
- [x] **6.6 Report generation.** `bench/report.py` → `RESULTS.md`; `bench/plots/make_plots.py` → PNGs; `bench/check_numbers.py` enforces README traceability.
- [x] **6.7 Acceptance rate, measured.** `bench/corpus/fetch.py --values` writes the argument values the committed corpus drops to a sidecar (`values.json`, digests over 64 canonical bytes; the corpus and its hash untouched). `bench/offline/run_acceptance.py` grades `PatternDrafter` with `resolve_decision` — the runtime's own gate — leave-one-trajectory-out over every trajectory, under the runtime's within-turn policy and the across-turns policy it does not implement, next to the copying ceiling (steps whose argument values have all been seen before). Output `bench/results/acceptance.json`; `RESULTS.md` and the README quote it and never the signature figure alone.
  *Verify:* leave-one-out by subtraction equals retraining; a right signature with a wrong value is a miss; a guess at a turn boundary is a miss within a turn and a hit across turns; the drafter sees only the turn's calls within a turn; acceptance never exceeds signature accuracy or the copying ceiling; the committed numbers regenerate identically.
  *Verify:* `RESULTS.md` regenerates identically from committed JSON; CI fails on a planted untraceable number.
- [x] **6.8 Latency sweep, against a real model.** `bench/online/run_latency_sweep.py`: every tool slowed to 0, 500 and 2000 ms, reads and writes alike; four arms (`B_strict_seq`, `B_seq` = early issue only, `B_readonly_spec`, `B_specunode`); 20 tasks per arm per rung against `claude-sonnet-5`; bootstrap intervals on the difference. Output `bench/results/sweep.json`.
  *Result:* the only resolved saving is early issue on `ops_agent`, the one sample app that hands its model turn to the runtime: +11.8% [+7.8%, +15.5%] at 500 ms and +16.7% [+15.1%, +18.2%] at 2000 ms. Guessing on top of it: every interval spans zero.
- [x] **6.9 Break-even α.** `bench/offline/run_break_even.py`: a guesser of controlled accuracy — right with probability α, otherwise a valid near miss — against early issue alone, on a stand-in calibrated from 6.8. Output `bench/results/break_even.json`.
  *Result:* with 500 ms tools a guess of any accuracy costs time (−0.2% to −1.3%, even at α = 1); with 2000 ms tools it pays from α = 0.25 and tops out near +4.8%. Guessing reads and writes is indistinguishable from guessing reads only in every cell: a staged write cannot leave before its branch retires.
- [x] **6.10 Tier-2 acceptance rate, against a real draft model.** `bench/online/run_tier2_acceptance.py`: `claude-haiku-4-5` shown the last 12 calls with their real argument values (`values_full.json`, gitignored), graded by `resolve_decision` with guesses carried across turns, 1000 sampled steps. Output `bench/results/tier2.json`.
  *Result:* 0.069 [0.054, 0.086], against tier 1's 0.0002 under the same rule. Under the runtime's own within-turn rule it is zero for any predictor on this corpus (6.7).
- [x] **6.11 When the model is the slow part.** Three changes for model-bound runs, none of which touches what makes speculation safe: `specunode.core.loop.agent_loop` (every result of a reply back in one message, the reply echoed back unchanged, thinking blocks included); prompt caching on by default (`target.cache`); and parallel nodes (a router may name several nodes; they run side by side and retire in the order named). `bench/offline/run_model_bound.py` measures the runtime's side against a stand-in (`bench/results/model_bound.json`); `bench/online/run_model_bound.py` measures a real model (`bench/results/model_bound_online.json`): seven configurations round-robin, 15 rounds, stopping at its cap or at the first run that fails.
  *Result, against `claude-sonnet-5`:* the same on-call task went from 11 replies to 5 — 35.4% less time and, with caching, 77.7% less cost; three independent checks side by side took 60.2% less time than one after another; caching alone cut the bill by 72.7% and did not change the time at this prompt size. Every run correct, 15 of 15 in each configuration; no leaks. Told nothing, the model already asks for independent calls together; the one-call arm is held to one by the API's `disable_parallel_tool_use`.
- [x] **6.12 Pull the plug.** `bench/offline/run_crash_safety.py`: one billing run -- three customers charged, three receipts, one summary -- killed at each of its seven writes, once before the request reached the upstream and once after it took effect with the reply lost, and restarted the way each system restarts: a plain async loop, LangGraph with a checkpointed node per customer, LangGraph's recommended `@task` per call with `durability="sync"`, SpecuNode, and SpecuNode with a `reconcile` per tool. No model, no network. Output `bench/results/crash_safety.json`.
  *Result:* extra effects in the world over 14 crashes: plain loop 49, LangGraph nodes 13, LangGraph tasks 7 (one for every lost reply), SpecuNode 0 -- stopping for a human in all 14 -- and SpecuNode with `reconcile` 0, finishing all 14 with every receipt naming its customer's real charge.

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
- [x] **7.7 Drafter poisoning.** Train the T1 index on traces with an adversarial "strong chain" that ends in a write, and run it inside the real scheduler. Measure: guesses forked, charges staged and discarded, the alpha gate closing and its closure journaled, wasted tokens (zero, because a pattern-index guess costs no model tokens — reported as zero, not invented); assert leaks stay 0.
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
- [x] LangGraph integration works on an unchanged graph file; plain-Python integration works; MCP proxy works with a generic client
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
  *Fired.* Tier-1 α is 0.0000 under the runtime's policy (6.7); a guess of any accuracy is worth −1.3% to +4.8% (6.9) and nothing any interval resolves against a real model (6.8). The headline is early issue plus the store buffer's safety, as this gate says, and no workload was tuned to flip it. — 2026-09-23
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
[7.7] Drafter poisoning: an index trained on an adversarial strong chain that ends in a charge made 8 predictions, all 8 squashed, 2,000 wasted tokens, the alpha gate disabled speculation, and 0 effects reached the world. The cost is tokens and stalls; it is not a leak. — 2026-09-16 [SUPERSEDED 2026-09-17: those figures came from a hand-driven Budget with an invented 250 tokens per squash, not from the runtime. See the 2026-09-17 entry below and the committed row.]
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
[2.2] Demo 2 (--demo past-write) built: the section 2.2 ops workload under sequential, readonly-spec and specunode, timed at the tool boundary. Measured, and the saving is one read's latency. The demo prints the overlap as a measured number rather than inferring it from the wall-clock difference, and says in its own output that model turn 2 is not hidden. — 2026-09-16
[2.2] Decision: Demo 2 compares what reached the *world* across arms, not the effect ledgers. The two baselines are separate implementations that issue each call in emitted order while specunode issues the independent read as its block parses, so the read takes the earlier program position and every later step index shifts. That is a difference between three programs, not a Rule 9 violation — Rule 9 compares speculation on against off for one graph, and there both arms early-issue. The demo prints the explanation rather than hiding the mismatch. — 2026-09-16
[2.2] Decision: no wall-clock figure from Demo 2 appears in the README. The demo measures on the machine that runs it; committing one would be flaky or would be a number nobody measured. The README describes the shape, and the acceptance test asserts everything that is not a timing. — 2026-09-16
[3.1] DEFECT (fixed, found by building Demo 2): _timed_read marked an early-issued read speculative by saving the branch's status, setting SPECULATIVE, and restoring the saved value in a finally. A read still running when the scheduler confirmed the branch put the old status back afterwards, demoting a CONFIRMED branch and making the next drain refuse with a Hard Rule 3 message about a branch that had been confirmed correctly. Overlapping a read with the drain is the point of early issue, so this was reachable on the ordinary path. The same shape could have promoted a squashed branch to CONFIRMED, which is Rule 3 itself. Fixed by separating the branch's lifecycle from a per-call durability counter. — 2026-09-16
[3.1] The first version of that regression test passed against the unfixed code, because a read emitted as the turn's *first* block finishes long before retirement and the race never opens. Rewritten with the read as the last block, and verified to fail without the fix before being kept. — 2026-09-16
[2.x] Demo 3 (--demo replay) built: the ops run SIGKILLed mid-branch, resumed from the journal, then replayed with a changed system prompt and again with speculation off, ending with the effect ledger. The kill delay is drawn against the run's measured *work* time rather than the process lifetime, because interpreter startup dominates the latter and a delay drawn against it almost never lands in the run. — 2026-09-16
[2.x] DEFECT (fixed, found by building Demo 3): a node failure was reported as 'node <name> failed', discarding branch.reason. For a replay that refuses because the prompt changed, that reason is the step index and the field-level diff — the single most useful diagnostic this system produces — and it was being thrown away. The drive loop now carries it out. The demo's acceptance test asserts the diff is present, not merely that the replay refused. — 2026-09-16
[2.x] ScriptedModel gained `consumed`. A resumed run is a new process, so the script would otherwise start over while the run does not, and the node that is the run's second model call is handed the script's first turn. That is an artefact of scripting rather than of resuming — a real model is simply asked the new question. The demo helper counts the journaled model_response entries and skips them. This masqueraded as a resume defect (restart_job delivered twice under different keys) until the journal was read: the runtime had resumed correctly at the right node and the helper fed it the wrong turn. — 2026-09-16
[2.5] Decision: Demo 3 compares effects across the clean and resumed runs by (tool, canonical args hash), deduping by effect_key. The idempotency token carries the run id, so comparing tokens across two runs reports every effect as different. What must match between them is the call, not the label the runtime gave it. — 2026-09-16
[2.5/2.6] GAP CLOSED: both tasks were ticked and neither CLI command existed. `specunode resume` and `specunode replay` are now implemented. The CLI module's own docstring had described `replay` at length for the whole build — the kind of gap a test catches and a reading does not, which is why there are now CLI tests where there were none. — 2026-09-16
[2.6] Decision (not specified): Config gains `graph: "module:attribute"`, naming the callable that returns (graph_adapter, tool_registry). A journal records what a graph did, not what it is, so resume and replay cannot find one without being told. Both commands exit 2 with the line to add rather than importing something plausible and re-driving the wrong program. The module:attribute form is the convention already used for a custom reducer. — 2026-09-16
[2.6] Decision: `specunode replay` dispatches nothing unless --dispatch is passed. A replay re-drives the graph, and a graph that re-drives dispatches, so replaying a run that charged a card would charge it again from a command whose whole purpose is to answer a question about the past. Dispatcher gained dry_run for this. — 2026-09-16
[2.6] The dry run is recorded on the ledger row, not only in the command's banner. A rendering is something people paste into a ticket, and one reading DISPATCHED for an effect nobody sent is a lie that travels further than the banner. The status reads 'DISPATCHED (dry run: not sent)'. — 2026-09-16
[2.6] Defect in my own first CLI test: it kept the system prompt and the world in module globals. pytest imports a test module as `integration.test_cli_replay` while the CLI's module:attribute reference imports `tests.integration.test_cli_replay`, so the two held separate copies and nothing propagated. The graph moved to a stateless helper that reads the environment and a durable world log — the two things both copies genuinely share. — 2026-09-16
[2.6] A replay writes to replay-<run_id>.db beside the journal it reads, never into it: interleaving a new run's entries with the record it is checking against would corrupt the only evidence there is. And it re-drives from the journaled `inputs`, not from recovery state — starting from the end state routes straight to the graph's terminal node, replays nothing, and looks like a clean replay while checking not one thing. Both found by running the command. — 2026-09-16
[4.x] PHASE GATE 4: the MCP proxy is now driven end to end by a generic mcp client against a real stdio upstream server, in tests/integration/test_mcp_end_to_end.py. Reads forward immediately, writes are held, a matching decision dispatches, a contradicting one discards — all asserted against a log the upstream server writes itself, not against what the proxy reports forwarding. — 2026-09-16
[4.x] DEFECT (BLOCKING, fixed): the proxy could not register a single tool against a real server. The forwarding function was annotated with the recursive JsonValue alias, and the SDK builds each tool's schema by making a pydantic model from the signature, so registration raised PydanticUserError and the proxy died before serving one request. Its own six control tools had the same problem. It had been shipped, documented and 'tested' in this state for the whole build. — 2026-09-16
[4.x] DEFECT (fixed): mcp 2.x renamed inputSchema to input_schema, and getattr(tool, 'inputSchema', None) went on quietly returning None — so the proxy advertised a schema it had inferred rather than the upstream's. A missing schema field is now a startup failure naming the SDK version, because a silent None on a field this load-bearing is exactly what the probe exists to prevent. — 2026-09-16
[4.x] DEFECT (fixed): the SDK parses a call's arguments against a model built from the function *signature*, not from the advertised schema, so a bare **kwargs forwarder advertised the upstream's schema and then rejected every call made against it. The forwarder is now given the upstream's parameter names, typed Any — the upstream is the authority on its own argument types and re-deriving Python types from JSON Schema would invent disagreements. — 2026-09-16
[4.x] DEFECT (fixed): the first forwarded read hung. MCPServer.run('stdio') opens its own event loop, so the served tools ran on one loop and the upstream ClientSession on another, and awaiting the session from the wrong loop deadlocked. run_stdio_async keeps them on one. The startup probe now checks for each SDK surface these fixes depend on. — 2026-09-16
[4.x] Four defects in a row, none visible to a rules-only test suite. The split — rules tested without a transport, because the rules carry the correctness claims and the SDK does not — is defensible right up until it is the only thing tested. — 2026-09-16
[3.6] DEFECT (fixed): Hard Rule 10 was dead code. Budget.record_resolution, record_speculative_read, inflight_branches, disable and the alpha window were written, unit-tested and documented, and the scheduler called none of them — the gate opened and closed on a window that never received a sample, wasted_tokens was 0 on every run, and the "speculation disabled" policy_event could not be emitted. Wired at fork, confirm, squash and execute_read; every resolution journals alpha_observed; the gate closing journals speculation_disabled once and the counters say why; a Prediction carries cost_tokens and tier 2 fills it from the draft model's usage; run_started carries the whole alpha configuration; the ledger renders "unjudged (h/n graded)" while the window fills instead of "n/a". Each fix verified by reverting it and watching its test fail first. — 2026-09-17
[4.1] DEFECT (fixed): the proxy passed upstream annotations through to the client and never consulted them, so a tool the upstream marked readOnlyHint: true was synthesised as an unknown WRITE and held until a decision arrived. ToolRegistry.from_mcp_tools existed for exactly this and was called only by its own tests; the spec's "or rely on MCP tool annotations" was false. serve() now classifies from the annotations with the config's overrides winning; the reference upstream gained an annotated read-only tool with no override and the end-to-end test proves it is forwarded. — 2026-09-17
[6.7] The acceptance rate, measured. fetch.py --values fetches the same 300 trajectories and writes the argument values the committed corpus drops to a sidecar (values over 64 canonical bytes stored as digests, which preserve equality and nothing else; the corpus and its hash untouched, and the sidecar's manifest records that the rows served today still hash to the committed corpus). run_acceptance.py grades PatternDrafter with resolve_decision, leave-one-trajectory-out by subtraction — exact and linear, a test proves it equals retraining — under the runtime's within-turn policy and the across-turns policy it does not implement. Result: 0.0000 and 0.0002 (3 of 19,184); signature top-1 0.5339 on the same steps; every argument value already seen at 0.0984 of steps, the ceiling for any copying predictor. — 2026-09-17
[6.7] Decision: the drafter's history at run time is the current turn's calls, and this corpus emits one call per turn, so the runtime's own acceptance rate here is zero by construction rather than by prediction quality. Carrying a guess across turns was measured rather than built: it would raise the figure to 0.0002, which does not justify a change to the scheduler. — 2026-09-17
[6.7] Decision: a first draft defined the copying ceiling as "the whole call already occurred" and the fixture refuted it — the drafter correctly assembled a call nobody had made yet from a value it had seen. The ceiling is "every argument value already seen"; whole-call repeats are reported beside it as the stricter figure. — 2026-09-17
[6.7] DEFECT (fixed): the ceiling was published as a bound on "any predictor that copies values out of history", which is more than it measures. PatternDrafter also copies from tool *results*, which this corpus does not keep, and an argument of 20.5% of graded steps came from one. It bounds tier 1 as graded here — no results — and every document now says so. — 2026-09-17
[6.7] DEFECT (fixed): the within-turn signature column read 0.0000 because signature_hit was gated on the same resolvability flag as the gate's verdict, so on a corpus where every call opens a turn it could not be true. It measured the policy, not the index, beside a row that already reported the policy. Ungated it is 0.5219: the index ranks the right signature about half the time under either policy, and the acceptance rate is still zero. — 2026-09-17
[7.7] Attack 7.7 now runs the poisoned drafter inside the real scheduler instead of hand-driving a Budget with an invented 250 tokens per squash. What is measured is what the runtime did: 4 guesses forked, 4 charges staged and discarded, the alpha gate closed after its 4-sample window filled with misses and the closure journaled 1 time, wasted_tokens 0 because a pattern-index guess costs no model tokens, leaked effects 0. The previous row's wasted_tokens=2000 was arithmetic, not a measurement. A first cut counted every branch_forked and every branch_resolved{confirmed}, which include the canonical branch of each node visit; guesses are now counted from resolutions that name an adopter. — 2026-09-17
[6.4] DEFECT (fixed): B_seq was not sequential. Tier-0 early issue ran in every arm, so the baseline already contained the only mechanism that saves time and every saving was measured against it. Policy.early_issue switches it off; B_strict_seq is the new baseline. — 2026-09-23
[6.4] DEFECT (fixed): B_readonly_spec was handed no predictor, so the PASTE arm never guessed and "PASTE's policy, credited" compared nothing. Both guessing arms now get the same drafter; Policy.speculate_writes=False is PASTE's rule, enforced by the WRITE_ON_PATH hazard. — 2026-09-23
[6.4] DEFECT (fixed), and a retraction: guesses were counted as branches forked minus squashed, and every node visit forks a canonical branch, so ordinary steps counted as correct guesses. A workload whose drafter offered nothing reported alpha 1.0, and support_agent's -15.5% was described to the user as "a perfect predictor, and still slower". No guess had been made. Guesses are now counted where they are made. — 2026-09-23
[6.4] DEFECT (fixed): the retirement re-check probed witnessed reads one at a time, so re-checking a turn cost the sum of its reads' latencies instead of the longest. Probes run concurrently, verdicts kept in read-set order. — 2026-09-23
[6.4] DEFECT (fixed): a turn that failed mid-stream left its early-issued reads running; they finished after run_finished and journaled into a closed run. The turn now cancels and awaits them before re-raising. — 2026-09-23
[6.10] DEFECT (fixed): tier-2 spend was priced at Sonnet rates for a Haiku draft model, and the draft model was shown hashed argument values it could never reproduce. Per-model prices; the real values live in a gitignored sidecar. 1000 steps: 0.069 [0.054, 0.086] for $3.04; the 428-step run before it gave 0.0607 [0.0397, 0.0841]. — 2026-09-23
[6.8] Sweep against claude-sonnet-5, 20 tasks per arm per rung: early issue saves 11.8% at 500 ms and 16.7% at 2000 ms on ops_agent; every guessing interval spans zero. $4.43. — 2026-09-23
[6.9] Break-even alpha measured with an oracle of controlled accuracy: guessing costs time with 500 ms tools at every alpha, pays from 0.25 with 2000 ms tools, and guessing writes adds nothing over guessing reads. — 2026-09-23
[6.11] Decision: when the model is the slow part, guessing tool calls cannot help, so the next work is the model-facing levers — fewer replies, prompt caching, parallel nodes. None of them touches the store buffer or the gate. — 2026-09-23
[6.11] DEFECT (fixed): replay served recorded turns by step alone. An agent loop makes every turn at its node's one program position, and parallel nodes make several nodes' turns at one position, so replay served the wrong turn and diverged. It keys on (node, step) and serves each node's turns in order; a test records replies in the reverse of the order replay asks for them. — 2026-09-23
[6.11] DEFECT (fixed): the parallel-group write-conflict check ran only when every body had finished, which is exactly when nothing staged is left to protect. It runs when every body is at rest, again before each node's writes leave, and at commit; the tests pin what each point can and cannot see. — 2026-09-23
[6.11] DEFECT (fixed): a node refused at retirement for a stale witnessed read was reported as "failed after its effects were dispatched", on the sequential path and the parallel one. Nothing of it had been sent. — 2026-09-23
[6.11] DEFECT (fixed): the shipped example config set temperature: 0.0 for claude-sonnet-5, which answers any temperature with a 400, so everyone who copied it would have failed on their first request. — 2026-09-23
[6.11] DEFECT (fixed, before it cost the budget): the first real round of the online run came back with every incident cell identical -- five replies, ten calls, whatever the prompt said. Told to make one call per reply, the model batched anyway, so the "before" arm was running the "after" behaviour. Stopped in round two after nine runs ($0.19); the one-call style now sets the API's disable_parallel_tool_use. A per-run progress line is what made it visible in time. — 2026-09-23
[6.11] Real-model result, claude-sonnet-5, 15 rounds, $2.33: 11 replies to 5, 35.4% less time, 77.7% less cost with caching; parallel nodes 60.2% less time; caching alone -72.7% cost and no resolved change in time; 105 runs, all correct, no leaks. Told nothing, the model batches independent calls by itself: what held runs at one call per reply was the app. — 2026-09-23
[6.11] REVIEW: an independent adversarial review of parallel nodes, the agent loop and replay keying found 9 defects, each with a failing test, all reproduced here first and all fixed with a revert check -- two critical (a crash after or in the middle of a group made a resume re-send effects), and one more found while fixing (a cancelled run left its node bodies running). Final Report, "What the fifth review found". — 2026-09-23
[6.12] Decision: a checkpoint cannot close the lost-reply window -- the call either finished or it did not -- and SpecuNode's answer to it was always "stop for a human", even when the request had never left. A tool may now declare `reconcile(key, args)`, asked on resume whether the call under that key took effect; the runtime acts on the answer, and dead-letters as before when there is no answer or the asking fails. It must read a record the upstream writes atomically with the effect (docs/adapters.md, docs/limitations.md). — 2026-09-24
[6.12] Pull the plug, measured: 49 / 13 / 7 / 0 / 0 extra effects over 14 crashes for plain loop / LangGraph nodes / LangGraph tasks / SpecuNode / SpecuNode + reconcile; LangGraph tasks is exact on every lost request and duplicates on every lost reply, exactly as its documentation describes. — 2026-09-24
[9.1] The front door: `import specunode` exposes `tool`, `node`, `graph`, `Runtime` (journal, buffer, dispatcher and scheduler wired with safe defaults), `current_idempotency_key()` and the rest, lazily, so importing stays cheap; docs had been writing `@specunode.tool` against a package that exported only `__version__`. `examples/quickstart.py` shows a crash with the charge made and its reply lost, and a resume that asks and carries on; a test runs it. — 2026-09-24
[9.1] DEFECT (fixed), found by the quickstart: recovery rebuilt committed state from the nodes' deltas alone, dropping the run's inputs from every resumed run -- a resumed node that read one failed -- and leaving a delta that changed an input key nothing to apply to. It starts from `run_started.inputs` now. — 2026-09-24
[6.11] What the guarantees cost, measured: against the fastest loop a developer would write by hand -- stream the reply, then run every call it asked for at once, with no journal and nothing to resume from -- the runtime is 0.6% slower with instant tools and 3.3% with 300 ms tools on the alert, and a read-only fan-out 1.7% and 13.1%. The fan-out's figure was 35% before this change: each lane's witnessed reads were re-checked at its own retirement, and lanes retire in order, so the re-checks queued one tool latency per lane. They are now made together while every lane is at rest, and a lane is re-checked at its own retirement only if an earlier lane in the group sent an effect -- the one thing that can make the up-front verdict out of date. A test proves the re-checks overlap; the "read what a sibling changed" test proves the second rule is load-bearing. — 2026-09-24
[6.6] DEFECT (fixed): the traceability check read only a results file's values, so the settings a file is keyed by -- latency rungs, accuracy levels -- could not be traced. Keys that are numbers outright count now; digits inside a key's name do not. — 2026-09-23
[2.5, 0.3] The kill tests now kill at counted points rather than timed ones. `test_kill_resume` dies before each of the clean run's 27 durable journal writes and either side of each of its 2 world mutations -- 31 points, every one a real mid-run kill on every machine, run four at a time in about 11 s -- and resumes each with a model that would decide differently if asked. Each point's outcome is fixed by what the kill left on disk: 24 finish with exactly the effects of the run they continue, 6 stop for a human on a claim with no outcome, 1 (before the first entry) is refused with nothing to resume; 0 duplicate deliveries. Planted bugs -- a resume that sends nothing new, one that re-sends after a lost reply, one that asks the model again -- each fail it at the point that exposes them. The journal kill test arms its killer after a drawn number of appends and asserts every append that returned survived the kill. Fifteen random delays, calibrated three times over, had failed on CI on both sides of the window. — 2026-09-25
[2.5] DEFECT (fixed), found by the sixth review: a resume asked the model again for a turn the journal already held, whenever the node's branch had not retired -- so a model that answered differently the second time could make a second, different call after a crash that fell after an effect was sent. A resumed node is now served the journaled answer when its question is identical (`RecordedTurns`), journaled again under the resumed branch with `recorded_from`. The condition left is a question that changed across the crash (docs/limitations.md). — 2026-09-25
[2.5] DEFECTS (fixed), found by the seventh review: (1) a node reading `session.model.stream()` that wrote as a block parsed had the write dispatched before the turn was journaled -- a write on a branch with an unjournaled target turn is now refused (`Branch.unjournaled_turns`, `CallScope.track_turn`; tests/integration/test_writes_wait_for_the_turn.py); (2) serving kept the longest attempt, so after a question changed a second resume charged a third time -- it now matches turn by turn and serves the latest attempt that asked the same questions; (3) serving pinned answers that had sent nothing -- only an attempt that may have sent something is served, and `effect_dead_lettered` records `sent`; (4) serving was one slot on a shared model -- it is per run; (5) the docs overstated; (6) the kill/resume sweep could not see it -- it adds a billing node that asks and charges in one step: 20 points, 13 finish, 6 stop for a human, 1 refused, 11 resumes served the standing decision, 0 duplicates. — 2026-09-25
[2.5] DEFECTS (fixed), found by the eighth review: (1, critical) the dispatcher retried a non-idempotent write after a failure that may have landed -- a lost reply was charged twice with default settings and no crash; now it is retried only if nothing left or the tool is idempotent, else asked about (reconcile) or dead-lettered; (2, critical) a dead letter's `sent` was its last attempt's -- it is now "maybe" if any attempt, here or in an earlier process, may have landed; (3) a resume retried every dead letter -- only one that never left is retried, and `specunode resolve` records an operator's word on the rest (Journal.resolve_dispatch; the ledger shows one row per key); (4) concurrent model calls were matched in answer order -- now in asking order; (5) a guess adopted before a stream failed was sent -- a failed turn discards what it adopted; (6) a failed call_turn blocked the node's later writes -- it no longer does; (7) serving pinned whole attempts -- now only the turns up to the last possible send; (8) the refusal's scope is the whole node run -- kept, documented, message corrected; (9) release notes and docs rewritten. tests/integration/test_lost_replies.py, test_failed_turns.py, unit/test_idempotency.py, unit/test_recorded_turns.py; each rule fails under its own planted bug. Pull the plug re-run: unchanged, 49/13/7/0/0. — 2026-09-25
[2.5] DEFECTS (fixed), found by the ninth review: (1, critical) a claim taken up for a retry after it never left stayed marked not_sent, so a crash during the retry made the next resume send it again -- `_claim` now re-arms it to in_flight/unknown in one compare-and-set; (2) a failed call_turn restored the open-turn count from a snapshot -- the turn now closes itself (`partial_turns_discarded`); (3) it also discarded other tasks' writes, and a concurrent drain could send a confirmed guess mid-stream -- adoption now waits for the journaled turn, and a failed turn discards only its confirmed guesses; (4) resolve matched only the dedupe key; (5) two dead letters for one key showed as two rows; (6) resolve read and settled separately, and a settle could overwrite a claim settled as sent -- `_settle` now refuses it; (7) any `sent` but "no" counts as maybe. tests/integration/test_turns_at_once.py is new; each fix fails under its own planted bug. — 2026-09-25
[2.5] DEFECTS (fixed), found by the tenth review: two resumes of one run at once sent a charge twice -- `Journal.hold_run` (in-process registry; flock, or a Postgres advisory lock, released when the holder dies) now holds a run for the length of `run`/`resume`/`resolve`, and a second driver gets RunBusy; a read after a write in the same reply was issued early and saw the pre-write value -- it now waits (`_reads_a_pending_write`); a stream without TurnComplete is a failed turn; call_turn closes its stream deterministically; adoption sits inside the failure cleanup; the ledger checks dispatch against program order; resolve accepts the printed key. tests/integration/test_one_driver_per_run.py is new; each fix fails under its own planted bug. — 2026-09-25
[2.5] DEFECTS (fixed), found by the eleventh review: (1, critical) the Anthropic adapter built a finished turn from a stream that dropped mid-reply, and nothing read `stop_reason`, so a tool call cut off mid-argument -- "amount": 150.0 cut after the 1 -- was sent as a charge of 1; a reply that did not finish, or stopped at `max_tokens` or by a refusal with a call in it, is now refused before it is journaled; (2) a guess confirmed at the end of a reply joined the branch when the turn ended, so the drain that sent the reply's first write sent it too, ahead of a read the model had asked for between them -- each guess now joins at its place in the reply; (3) the LangGraph wrapper shared one Scheduler across every call, and two requests at once ran as one -- each call gets its own, a Scheduler drives one run, a buffer serves one run at a time, and `run` refuses a run id the journal already holds; (4) the SQLite run lock was keyed by the path as given and named after the run id -- it is keyed by the real path and named by a hash, and an OSError is a JournalError; (5) the Postgres run lock used a 32-bit key on the shared writer connection -- 64 bits, on its own connection, checked before every send; (6) a dead letter took its stage index as its place in the send order -- it records where it was tried, and the order check counts sent effects only; (7) an adoption moved the guess's write before journaling the move -- a failed append left it to be sent; the record now comes first; (8) `--journal` was a Path, which mangled a Postgres DSN -- as did `specunode.Runtime`; and the config's `journal` section, read by nothing, is now the CLI's default. Each fix fails under its own planted bug. — 2026-09-25
[2.5] DEFECTS (fixed), found by the twelfth review: (1, critical) a turn that failed -- a reply cut off and refused, an overloaded model -- left no outcome on disk, so a node that caught the failure and asked again, then crashed after its charge went out, was not served on resume: every question after the failed one went to the live model, which could charge a second time, and replay refused the same run; a failed turn is recorded as the failure it was and served and replayed as that failure, and the Anthropic adapter raises its SDK's API errors as ModelError; (2, critical) the Postgres run lock was checked once, before the claim, and a process that lost it during a retry's backoff sent the retry while another sent the same charge -- it is checked before every attempt, and a settle that finds its offset taken is a JournalConcurrencyError; (3, critical) a crash after a policy_event and before run_started left a run `run` refused and `resume` drove from empty state, charging the wrong customer -- run_started is first, and a run that never recorded its start is started, not resumed; (4) the eleventh review's per-slot adoption moved a guess's reads with its writes, after the stale-read check -- reads move when the turn ends; (5) the documented recovery of a crashed LangGraph run did not exist -- documented as unsupported, `resume` says so, and LangGraph's own config is passed through; (6) agent_loop reported a reply cut off without a call as end_turn; (7) ledgers signed before the dispatch-order change were reported as edited -- format 1 is recognised; (8) the CLI read a config's journal path from the wrong folder and without `~` -- fixed, and every command takes --config; (9) wrap(run_id=) made a wrapped graph single-use -- removed; (10) resuming an unknown run printed a traceback. And three it suspected but could not reproduce, all real: a confirmed guess discarded with its turn made a failed run look resumable; a run cancelled from outside journaled its nodes' cleanup after letting go of the run; the Postgres run lock was taken on the event loop. Each fix fails under its own planted bug. — 2026-09-25
[2.5] DEFECTS (fixed), found by the thirteenth review: (1, critical) a model call a node's own timeout cancelled left no outcome, so a node that asked again, charged and crashed was sent to the live model on resume and could charge twice -- a cancelled turn is journaled as one and served as one that never answers, until the node stops waiting again; (2, critical) an answer written after the node's timeout fired was on disk as delivered, and a resume acted on it -- the write is let finish and a second outcome says it was never handed over, and the last outcome of a question is the one served and replayed; (3) a served failure was a ModelError while the live one was the client's own error, so a node catching the client's type could never be resumed -- the live path raises ModelError too, caused by the client's error, and ModelError is exported; (4) witnessed reads were re-checked once per node, at its first park, so a write after a later read went out on a stale value -- reads made since are re-checked before every later drain, and what the node staged since a stale one is discarded unsent; (5) a connection dropped mid-reply reached nodes as the HTTP client's own error -- the adapter raises it as ModelError; (6) a user-level config without a journal section sent every command to an empty journal beside it. And two it suspected: a second cancel while the run lock was being taken left its release to the garbage collector, and a run that lost its Postgres lock still wrote run_finished into a run another process might be driving. Each fix fails under its own planted bug. — 2026-09-25
[2.5] DEFECTS (fixed), found by the fourteenth review, which found no way to send a write twice: (1) a replay of a run interrupted during a model call waited forever on the turn recorded as cancelled, and replay did not re-emit text, so a node that stopped reading on text waited too -- a served "never answers" turn now waits as long as the recorded caller did, and a margin, then raises TurnAbandoned, and replay re-emits text; (2) replay merged an answer with the marker that its node never saw it only when the two were adjacent, so a question answered in between made replay hand the answer over and `replay --dispatch` send a charge the run never made -- the last outcome per request wins; (3) replay listed turns in answer order, not asked order, and refused a faithful re-run of concurrent calls; (4) a resume hung without a word on a pinned cancelled turn whose cancellation did not recur -- the same bounded wait; (5) a stream closed by the garbage collector after the run ended journaled "cancelled" after run_finished -- not outside a run this process holds; (6) a third cancel while "cancelled" was written cancelled the write; (7) a stale read after a node's last write was reported as a discarded write. Also from its list: a journal failure mid-drain left the parked node running, and a run that lost its Postgres lock after its last drain reported success with no run_finished -- it raises now. And the slow-disk run found one more of finding 6's kind: a caller cancelled while its question was being written left the question on disk with no outcome -- the write is let finish and given a cancelled outcome. Each fix fails under its own planted bug. — 2026-09-25
[2.5] DEFECTS (fixed), found by the fifteenth review: (1, critical, in the fourteenth's repair) TurnAbandoned was a RuntimeError, so a node's ordinary `except Exception` fallback caught it and asked a question the recorded run never asked -- live -- and a card was charged a second time; nor was the abandoned question given an outcome, so the next resume asked live again. It is a BaseException now, which ends the node, and it is recorded; (2) a served turn cancelled again was journaled with a wait of 0, so the resume after gave up after the margin alone; (3) the bound measured only the model call, and a node whose deadline also covered earlier work -- faster on resume -- was abandoned: it is twice the recorded wait and 30 s more; (4) the stale-read message was wrong for a node that had finished; (5) `status` said a failed run was not resumable while `resume` re-ran it; (6) read-only commands reported on a mistyped run id, and created the journal they were pointed at; (7) `--help` showed raw ``markup``. Each fix fails under its own planted bug. — 2026-09-25
[2.5] DEFECTS (fixed), found by the sixteenth review: (1, critical) served answers came back at once, in the order asked, so a node that acted on whichever of two answers came first -- or fell back when one was not back in time, without cancelling it -- decided otherwise on resume and in replay, and a different charge went out with nothing changed; they come back at their recorded pace, per block of a stream, and in the order they first came, and a replay takes as long as the model did; (2) a resume that left an earlier attempt's claim unsettled reported success, and the ledger hid it -- the run is not ok, naming the tools for `specunode resolve`, and the ledger marks MAY HAVE BEEN SENT; (3) a node ended by TurnAbandoned still wrote from its `finally`, and was sent -- the node is closed to writes and model asks first; (4) the abandonment bound measured only the call, so a node whose deadline covered earlier work, slower in the recorded run, was abandoned every time -- the bound is also node-relative, and `resume --ask-abandoned` lets an operator ask such a turn again; (5) read-only commands wrote the journal's tables into a database that was not a journal, a folder gave a traceback, and a failed LangGraph run was called resumable. Then the slow-disk run, before any of it was committed, found the fix for (1) hanging resumes: (6) an answer waited for every earlier answer of its node to be handed over, and one the node stopped waiting for never was, so every resume of a node that gives up on a question and asks again waited for ever -- an answer is now released however its call ends. Fixing it turned up two more: (7) answers were ordered by where each was first recorded, which for a resume of a resume put an answer served again ahead of one asked live that came back before it, and a node reading the first only after the second waited for itself for ever -- they are ordered by where each came back in the attempt served, and never held past the earlier one's own bound; and (8), older than both, a node that caught `TurnAbandoned` and returned was committed, and the run reported success -- it is refused. The slow-disk CI job, which had not run the resume tests, runs them now. Each fix fails under its own planted bug. — 2026-09-26
[2.5] DEFECTS (fixed), found by the seventeenth review -- twelve, four critical, and all four in the sixteenth review's repairs: (1, critical) a streamed text block was timed by its last piece and served whole then, so a node that gives up on a model slow to start saw the first word late on resume and in replay, and charged its fallback as well -- every piece is recorded with its time now, and served as it came; (2, critical) a served stream the node gave up on was recorded with every block and no timing, so the next resume handed the node all of it at once, an early read took the fallback's position, and the fallback went out twice -- a turn is recorded as its caller had it; (3, critical) `--ask-abandoned` asked every turn the recorded node had given up on again at once, including one it would give up on again, and the node charged a price no run had decided -- it asks only at the point the node would be stopped; (4, critical) a dead letter that may have been sent did not keep a resume from reporting success, as a claim still in flight does; (5) a node's `finally` refused on its way out replaced why it was stopped in the run's error; (6) a replay was told to resume; (7) a claim demonstrably never sent failed a resume as maybe sent; (8) `ledger --json` and `--normalised` left out what may have been sent; (9) a journal whose path held `#`, `?` or `%` was refused; (10) read-only commands created the journal's tables in a Postgres database that was not a journal; (11) two tests failed on a disk slower than CI's; (12) replay.md described repeated requests at one step wrongly. A thirteenth, suspected, is closed too: a guess adopted by a branch closed to writes could still stage into it. Each fix fails under its own planted bug. — 2026-09-26
[2.5] DEFECTS (fixed), found by the eighteenth review -- seven, two critical, one of them in the seventeenth review's repair: (1, critical) a turn given up on or failed was recorded with its blocks packed together while its pieces kept the stream's positions, so after a thinking block -- which streams nothing -- a piece was served as the wrong block or dropped, early reads went missing or tripled, and a charge moved to a new key -- a partial reply keeps every block's position, a piece records whether it is a tool call, and a piece that does not fit is an error, never served as something else; (2, critical, older) a piece was timed by when the node took it, not when the model sent it, so a node busy with a slow lookup made the model look slow to a resume whose lookup was quick -- the model's stream is read ahead as it arrives (`_ReadAhead`); (3) `--ask-abandoned` lost a race to a later answer's own wait, and asked the model live on a node already stopped; (4) a turn asked again live read as served; (5) the error recommended `--ask-abandoned` for a turn it would refuse; (6) an empty ledger said nothing had reached the world just above what may have; (7, older) a call that outlived its node wrote into the journal after `run_finished`. Fixing (1) turned up a replay that raised a recorded failure before the read it had just handed over got to run, so the read's position went untaken -- an early read takes its position as its block arrives -- and the review's suspicion that a replay's clock started a question-write too early was right, and closed. Each fix fails under its own planted bug. — 2026-09-26
[2.5] DEFECTS (fixed), found by the nineteenth review -- five, one critical, and it in the eighteenth's repair again: (1, critical) the model's stream was opened before the turn's recording began, so a client whose `stream()` refused at once -- a rate limiter -- left its question with no outcome and the node the client's raw error, and a resume asked the model again -- it is opened inside the read-ahead, where its failure is recorded like any other; (2) a reply refused as cut off was timed by when the node got to it -- by when it arrived now; (3) a record that did not hold together raised an error a node could catch, and refused a journal from the commit before, whose tool pieces were marked 0 -- the old mark is read, and a record that does not fit stops the node as `TurnAbandoned`, recorded; (4) calls left running still wrote after `run_finished`: while it was being written, into a resume of the same run in the same process, or as the first question of their drive -- every call carries its drive, and the model is told the drive is over before `run_finished` is written; (5) a question whose write straddled its node being stopped went to the live model -- looked at again once it is on disk. And its suspicion was right: a turn that failed after a guessed block left the next call one position earlier with speculation on than off -- every block now takes its position as it arrives. The model client's contract is written down (adapters.md). Each fix fails under its own planted bug. — 2026-09-26
[2.5] DEFECTS (fixed), found by the twentieth review -- seven, two critical, both in the nineteenth's repair of where a failed turn's calls sit: (1, critical) a journal from an earlier commit, resumed here, put a retried charge at another position -- a new key, and it went out twice with the run reporting success; (2, critical) whether a turn that did not complete kept the positions its blocks had taken hung on timing -- a node's deadline firing while a guess was settled, a resume slower to write its question -- so a fallback charge moved and went out twice, and a finished run failed to replay. A turn that does not complete now takes no positions (`Branch.rewind_to`; position rule 2, recorded in `run_started`), and a journal from another rule with such a turn in it is refused by resume and replay; (3) a Scheduler that could not hold its run ended its drive anyway, and its retry could ask the model nothing; (4) the stop check ran a step of the event loop before the stream was opened -- it runs where it is opened too; (5) only the target model was told a drive was over: a drafter's model and a left-over tool call still wrote after `run_finished` -- the drive's end is now kept for every call of it; (6) a new test timed a late question with a sleep, and failed on a slow disk; (7, older) a cancel inside a guess's squash squashed and counted it twice -- it is let go of first, and its squash finished. Each fix fails under its own planted bug. — 2026-09-26
[3.3] Four tests settled a guess inside a 15–25 ms block delay, which a journal append on a slow CI disk could miss. Found all at once by running the suite with every append 40 ms slower (`tests/slow_journal.py`); each now holds the settling block until the event it needs has happened, and passes at 40, 100 and 250 ms of added latency. The `slow-disk` CI job runs the tests that guess that way on every push. — 2026-09-25
```

---

## Final Report

**Written 2026-09-16, revised 2026-09-17 after an independent adversarial audit, and on
2026-09-23 after the first measurements against a real model.**

**1069 tests pass, 24 skip — 18 of the passes against a real Postgres 16 server.** `ruff check`,
`ruff format --check` and `mypy --strict` are clean with every extra installed, which is a
stronger statement than it was: the `anthropic` package sits in mypy's `ignore_missing_imports`
list and was not installed, so a whole adapter had been type-checking against `Any`.

**Twenty independent adversarial reviews have found 197 defects here, 33 of them critical.** The
counts, in order, were **23, 17, 13, 24, 9, 6, 6, 9, 8, 7, 8, 10, 6, 7, 7, 6, 12, 7, 5, 7.** All are fixed but one, kept on purpose
and documented: a node's writes are refused while any model turn it started is unjournaled, even
one that did not decide the write. That number is the most useful thing in this report, so it
is at the top rather than buried: the version of this document written a day earlier described
a finished project. The twentieth to sixth reviews (2026-09-25 and -26) and the fifth (2026-09-23) are
summarised below; the fourth is in commit `c8802ec`.

The second audit is the one worth reading twice. It was told to assume the first round's fixes
were incomplete, and **two of its four criticals were inside those fixes** — one of them was two
fixes from the same commit cancelling each other out. The lesson is not that the fixes were
careless; it is that a defect and its repair are written by the same judgment, and that judgment
is exactly what failed the first time.

### What was built, in five sentences

SpecuNode executes an agent graph the way an out-of-order CPU executes instructions: reads issue
early, writes wait in a branch-scoped store buffer, and the target model's real decision is the
only thing that can release a write. Every model output and tool result is journaled and fsynced
before the runtime acts on it, so a crashed run resumes and a finished run replays, and the
replay refuses the moment the run would ask the model a different question. Three never-skipped
tests hold the invariants across three sample workloads and two drafter tiers: nothing reaches
the world from a branch that did not retire, and the effect ledger with speculation on equals
the ledger with it off. The third — that the speculative arm asked the model the same questions
as the sequential arm — holds in the only form these workloads can exercise, which is that both
arms send byte-identical prompts; **no shipped path makes a speculative branch send a request at
all**, so the harder half of Hard Rule 13 is vacuously true rather than checked, and the
retirement-time rebuild the rule describes is not implemented. The runtime refuses such a branch
at retirement instead of stamping it. It ships a LangGraph integration that runs an unchanged graph file, a plain-Python API, an
MCP proxy driven end to end by a generic client, three drafter tiers, three demos, and a
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
| T1 **signature** accuracy (upper bound on acceptance), n=25 | top-1 0.5350, top-3 0.8369 |
| **T1 acceptance rate**, the gate's own verdict, n=300 (added 2026-09-17) | **0.0000** under the runtime's within-turn policy; **0.0002** with guesses carried across turns (3 of 19,184 graded steps); signature top-1 on the same steps 0.5339 across turns, 0.5219 within; ceiling for this grading 0.0984 |

**The second row is the headline and it is zero.** Every tool call in that corpus opens a new
model turn, and a staged write blocks the next *turn* because that turn would have to contain a
placeholder where the real result belongs. There is nothing to run ahead into.

The third row is the part that is not zero, and it is a different quantity: PASTE refuses to
speculate on a tool with side effects at all, while SpecuNode stages one, so a *predicted* write
can be run ahead and discarded. That is an upper bound on opportunity rather than a speedup, and
realisable only where the predictor is right. The two numbers are never quoted apart.

The fourth row turned out to be the same fact seen from the runtime side. The drafter's history
is the calls within the current turn, so a one-call-per-turn workload offers it nothing to
predict from at all — it is not that speculation is unprofitable there, it is that no prediction
can be formed.

**The sixth row was added on 2026-09-17, and it closes the question the Final Report below left
open.** The acceptance rate is measured now — the real drafter graded by the real gate on the
same corpus joined to its argument values (`bench/offline/run_acceptance.py`) — and it is
0.0002 at best and 0.0000 under the runtime's own policy. Signature accuracy
was never the number: the gate needs the exact command string, path or thought, and every
argument value of a call has already appeared earlier at only 9.8% of steps, which is the
ceiling for tier 1 as graded here — copying from earlier calls, with no tool results, because the
corpus keeps none. On this corpus the tier-1 predictor as built is worth nothing, and the case
for the store buffer rests on a predictor that generates values, which needs an API key to
measure.

**The second headline number exists now, and it is small** (revised 2026-09-23). Against
`claude-sonnet-5`, with every tool slowed to 500 ms or 2000 ms, issuing each read the moment its
block parses saves 11.8% and 16.7% on the one sample app that hands its model turn to the
runtime, and guessing on top of it adds nothing any interval resolves (6.8). A guesser of
controlled accuracy is worth at most +4.8% even when always right, because a guess runs at most
one block ahead of the model (6.9), and a real draft model is right 6.9% of the time (6.10). So
speculation's measured contribution is safety, not speed. When the model is the slow part, the
levers are fewer replies and replies side by side (6.11), and those are measured against a
real model: the same on-call task in 5 replies instead of 11, 35.4% less time and 77.7% less
cost; three independent checks side by side, 60.2% less time.

### The anti-results

- **Past-write speculation: 0.0000 span.** 95.5% of the corpus is the
  `model → write → model(reads the result)` shape that gains nothing from it.
- **Tier-1 acceptance rate: 0.0002 across turns, 0.0000 within.** 3 of 19,184
  graded steps would have retired; the ceiling for that grading is 0.0984. Signature accuracy
  was an upper bound, and a loose one.
- **Break-even α: never, with 500 ms tools; about 0.25 with 2000 ms tools, for at most +4.8%**
  (6.9). With fast tools a guess of any accuracy costs time. `alpha_floor` still defaults to
  `None` — a floor would only encode which of those regimes a deployment is in.
- **Undetectable stale reads: 0.5** of the stale reads in attack 7.3's fixture were unwitnessed
  and therefore undetectable. That fraction, not the stale rate, is the honest number.
- **Journaling and classification overhead: 6.645 ms per step**, 87.7% of wall clock in that
  measurement. The percentage is the misleading half — a scripted model answers instantly, so it
  is the worst possible ratio. The absolute per-step figure is what transfers.
- **Demo 2's saving is one read's latency.** Speculating past a write hides tool latency, not
  model latency, and on that workload it hides exactly one independent read. The demo says so in
  its own output rather than letting the shape of the table suggest more.

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

Held: 7.7 drafter poisoning (4 guesses forked, 4 squashed, 4 charges staged and discarded, the
alpha gate closed once, 0 wasted tokens — a pattern-index guess costs no model tokens — and 0
leaks) and 7.8 replay under model drift (both cases diverge at step 0).

### What the twentieth review found

Seven findings, two critical -- both in the nineteenth review's repair, the rule for where a
turn's calls sit when the turn does not complete.

- **Critical: where a failed turn's calls sat depended on timing.** The nineteenth review's
  repair had every block of a turn take its program position as it arrived, and kept those
  positions when the turn failed or its node gave up on it. How many had arrived by then is
  timing -- a guess being settled on a slow disk, a question slower to write on resume -- and a
  resume reproduces it only roughly: a fallback charge moved to another position, under a new
  key, and went out twice; a run that finished fine failed to replay. A turn that does not
  complete now takes no positions at all.
- **Critical: a journal from an earlier commit charged twice.** The same change, resumed over a
  journal written before it, put a retried charge at another position than the one already sent.
  The rule is recorded in `run_started` now, and a journal from another rule that has such a
  turn in it is refused by resume and replay, naming the version to use.
- And five more: a Scheduler that could not hold its run ended its drive, so its retry could ask
  nothing; the stop check ran a step before the stream opened; only the target model was told a
  drive was over -- a drafter's model and a left-over tool call still wrote after
  `run_finished`; a new test failed on a slow disk; and, older, a cancel inside a guess's squash
  counted it twice. All fixed.

### What the nineteenth review found

Five findings, one critical -- in the eighteenth review's repair, as a critical finding was in
the repair before it in each of the two reviews before.

- **Critical: a stream refused at once was never recorded.** Reading the model's stream ahead as
  it arrives, the eighteenth review's repair opened it before the turn's recording began. A
  client whose `stream()` refuses at once -- its own rate limiter saying "overloaded" -- left
  its question on disk with no outcome and handed the node its raw error; a resume asked the
  model again, and charged a second time. It is opened inside the read-ahead, and its failure
  recorded like any other.
- And four more: a reply refused as cut off was timed by the node rather than by when it came; a
  record that did not hold together raised an error a node could catch -- and refused a journal
  written a commit earlier; calls left running still wrote after `run_finished` in three ways,
  now closed by telling the model when each drive is over; and a question written while its node
  was being stopped went to the live model. The review suspected, without building it, that a
  failed turn left the next call at another position with speculation on than off; it did, and
  every block now takes its position as it arrives. All fixed.

### What the eighteenth review found

Seven findings, two critical -- one in the seventeenth review's repair, and one older than the
sixteenth's.

- **Critical: a thinking block moved every piece after it.** A turn the node gave up on, or that
  failed, was recorded with its blocks packed together, while its pieces kept the positions the
  stream gave them. A thinking block streams nothing, so every piece after one pointed at the
  wrong block, or none: served, a read issued early went missing or was made three times, and a
  charge moved to a new key and went out twice. A partial reply keeps every block's position
  now, each piece says whether it is a tool call, and one that does not fit is an error.
- **Critical: a busy node made a quick model look slow.** A piece was timed by when the node
  took it, not when the model sent it. A node that did a slow lookup between two pieces had the
  second recorded as arriving after the lookup; a resume whose lookup was quick was held back
  until then, and fell back. The model's stream is read ahead as it arrives, and each piece is
  timed by its arrival.
- And five more: `--ask-abandoned` lost a race to a later answer's wait and was stopped anyway,
  then asked the model on a node already stopped; a turn asked again live read as served; the
  error recommended the flag for a turn it would refuse; an empty ledger said nothing had
  reached the world above what may have; and a call that outlived its node wrote after
  `run_finished`. Fixing the first turned up a replay whose early read lost its position to a
  failure raised before the read got to run; and the review's suspicion that a replay started
  its clock a question-write early was right. All fixed.

### What the seventeenth review found

Twelve findings, four critical -- and all four were in the sixteenth review's repairs: the pace
answers are served at, the flag that asks an abandoned turn again, and the check that a resume
leaves nothing unsettled.

- **Critical: served text arrived as late as its last word.** A streamed text block was timed by
  its last piece, and a served one was handed over whole at that time. A node that gives up on a
  model slow to start -- no first word within 0.3 s -- had its first word 0.6 s late on resume
  and in replay, and charged its fallback on top of what the run charged. Every piece is
  recorded now, with when it reached the caller, and served the same way.
- **Critical: a turn the node gave up on was recorded as if it had seen all of it.** Served,
  then given up on before its first block, it was recorded with both blocks and no timing; the
  next resume handed the node both at once, a read issued early took the fallback charge's
  position, and the fallback went out a second time under a new key. A turn is recorded as its
  caller had it, live or served, answered, failed or given up on.
- **Critical: `--ask-abandoned` asked too much.** It asked again every turn the crashed run had
  given up on, at once -- one the node would give up on again too, which now answered in time,
  and the node charged a price no run had decided. It asks only at the point the node would be
  stopped, and not a streamed turn part of which was handed over.
- **Critical: a dead letter that may have been sent did not stop a resume reporting success.**
  The sixteenth review's fix counted claims still in flight, not dead letters whose request may
  have left; one whose node no longer made that call was passed over.
- And eight more: why a node was stopped was replaced by whatever its `finally` then ran into; a
  replay was told to resume; a claim demonstrably never sent was reported as maybe sent; the
  JSON and normalised ledgers left out what may have been sent; a journal path with `#`, `?` or
  `%` was refused; read-only commands changed a Postgres database that was not a journal; two
  tests failed on a slower disk than CI's; and the replay docs described one step wrongly. The
  review suspected, without building it, that a guess adopted by a branch closed to writes could
  still stage into it; it could, and is refused now. All fixed.

### What the sixteenth review found

Six findings, one critical -- and this time not in a repair: in the idea of serving answers.

- **Critical: a resume that changed nothing could still decide differently.** Served answers
  came back instantly, in the order the questions were asked. A node that acts on whichever of
  two answers arrives first, or falls back when one is not back in time without cancelling it,
  decided otherwise on resume and in replay -- a different charge under a new key, with the
  world, the code and the questions all unchanged. Served answers now come back at the pace
  they first did, block by block for a stream, and in the order they first arrived; a resume
  and a replay take as long as the model took.
- A resume that left an earlier attempt's claim unsettled reported success, and the ledger
  listed only what settled. The run now says what may be out, and the ledger marks it. A node
  ended by `TurnAbandoned` still wrote from its `finally`; it is closed first now. The wait
  before abandoning measured only the call; it is node-relative as well, and an operator can
  resume with `--ask-abandoned`. And read-only commands changed databases that were not
  journals. All fixed.
- **And the fix for the critical one hung resumes** -- found by the slow-disk test run before it
  was committed: once more, the repair was where the next defect lived. An answer waited for
  every earlier answer of its node to be handed over, and one the node had stopped waiting for
  never was: a node that gives up on a question and asks again waited for ever on resume. Fixing
  that turned up two more. Answers were ordered by where each was first recorded, which for a
  resume of a resume could put an answer behind one that had come back after it -- and a node
  that needed the first before it would read the other waited for itself. And, older than both,
  a node that caught `TurnAbandoned` and returned was committed, and the run reported success.
  An answer is now released however its call ends, ordered by where it came back in the attempt
  served, and never held past the earlier answer's own bound -- a node still holding that one
  open by then is stopped -- and a node that catches `TurnAbandoned` is refused.

### What the fifteenth review found

Seven findings, one critical -- and the critical one was in the fourteenth review's repair, the
fourth time in five reviews that a fix was where the next defect lived.

- **Critical: a node could catch the abandonment and charge again.** The fourteenth review's fix
  ended a resumed node's wait on a question its recorded run had given up on with
  `TurnAbandoned`, a `RuntimeError`. A node with an ordinary `except Exception` fallback caught
  it, asked a question the recorded run never asked -- live -- and charged a second time; and
  the question it gave up on had no outcome, so the next resume asked live again. It is a
  `BaseException` now, like a cancellation, so it ends the node; and it is recorded.
- The wait before giving up was recorded as 0 by a resume that was itself served the turn, and
  measured only the model call when a node's deadline also covered earlier work. It is carried
  over from the recorded run now, and is twice the recorded wait and 30 seconds more.
- The stale-read message was wrong for a node that had finished; `status` called a failed run
  unresumable while `resume` re-ran it; read-only commands reported on mistyped run ids and
  created the journals they were pointed at; `--help` showed raw markup. All fixed.

### What the fourteenth review found

No critical finding: it found no way to send a write twice, and said so. Seven others, five
major, in replay, in hangs and in late writes -- two of them in the thirteenth's repairs.

- **A turn served as "never answers" could wait forever.** A replay of a run interrupted during
  a model call, and a resume whose node no longer cancelled the call its recorded run had given
  up on, waited without end and said nothing. Such a turn now waits as long as the recorded
  caller did, and a few seconds more, then raises `TurnAbandoned`: the node is not asking what
  it asked before. Replay also re-emits text now, as a served turn does.
- **Replay could hand a node an answer it never saw.** The marker that a caller had stopped
  waiting was merged with its answer only when the two sat next to each other; a question
  answered in between made replay serve the answer, and `replay --dispatch` sent a charge the
  run never made. And replay matched concurrent questions in the order their answers came back.
  Each question's last outcome is the one used now, in the order the questions were asked.
- A stream the garbage collector closed after the run had ended wrote "cancelled" after
  `run_finished`; a third cancel cancelled the write recording the second; and a stale read
  after a node's last write was reported as a discarded write. All fixed -- and the slow-disk
  run of the new tests found one more: a caller cancelled while its *question* was being written
  left it on disk with no outcome. It is let finish, and recorded as cancelled.

### What the thirteenth review found

Six findings, two critical -- both the same shape as the twelfth's first, in a place its fix did
not reach: a node that stops waiting for the model.

- **Critical: a timed-out turn had no outcome.** The twelfth review's fix recorded a turn that
  failed with an exception. A node that wraps its model call in a timeout cancels it instead,
  and nothing was written for that question: a node that asked again and charged, then crashed,
  was sent to the live model on resume, which could charge a second time. A cancelled turn is
  recorded now, and served as one that never answers until the node stops waiting again -- so
  the node's own timeout fires the same way, and it asks its second question again.
- **Critical: an answer the node never saw was served to it.** A timeout that fired while the
  answer was being written left it on disk as delivered; a resume served it, and the node acted
  on it as well as on the answer it had asked for next. The write is let finish, a second
  outcome says it was never handed over, and a question's last outcome is the one that counts.
- A failure served on resume was a `ModelError` while the live one had been the client's own
  error, so a node that caught the client's type could not be resumed; both are `ModelError`
  now. Witnessed reads were re-checked only at a node's first write, so a later write went out
  on a read that had gone stale; they are re-checked before every write. A connection dropped
  mid-reply reached nodes as the HTTP client's error; and a user-level config sent every command
  to an empty journal. All fixed, with two smaller findings it suspected.

### What the twelfth review found

Ten findings, three critical -- two of them in the eleventh review's repairs, again.

- **Critical: asking again after a failed turn broke the resume.** The eleventh review's fix
  refused a cut-off reply before it was journaled, and its message said to ask again. A node
  that did, and crashed after the charge the second answer asked for, left a question with no
  outcome on disk; on resume that question matched nothing, and every question after it went
  to the live model, which could decide differently and charge again. The same was true of any
  failed turn a node retried -- an overloaded model -- and replay refused the run. A failed
  turn is now recorded as the failure it was, and served and replayed as that failure.
- **Critical: a process that had lost its Postgres run lock sent a retry.** The lock was
  checked once, before the claim; a process that lost it during a retry's backoff sent the
  retry while the process that had taken the run up sent the same charge. It is checked before
  every attempt now.
- **Critical: a crash in the run's first moment lost its inputs.** With the default policy a
  `policy_event` was written before `run_started`; a crash between them left a run that `run`
  refused and `resume` drove from empty state, and the support agent charged the wrong
  customer. The start is the first entry now, and a run that never recorded it is started, not
  resumed.
- A guess confirmed after a write in its reply moved its reads with its writes -- after the
  stale-read check had run -- so a stale guessed read let a charge out that the same run
  without guessing refused. Its reads move when the turn ends.
- The docs described recovering a crashed LangGraph run from its checkpointer, and no such path
  worked. It is documented as unsupported in this version, `resume` says so, and `wrap()`
  passes LangGraph's own config through. `agent_loop` reported a reply cut off with no call in it
  as the model finishing; ledgers signed before the dispatch-order change verified as edited;
  the CLI read a config's journal path relative to the shell, not the config; `wrap(run_id=)`
  made a wrapped graph single-use; and resuming an unknown run printed a traceback. All fixed.

It also listed four things it suspected and could not reproduce. Three were real: a guess
confirmed in a turn that then failed stayed "confirmed", and a failed run that sent nothing was
reported resumable; a run cancelled from outside journaled its nodes' cleanup after it had let go
of the run; and the Postgres run lock was taken on the event loop. All fixed and tested.

### What the eleventh review found

Eight findings, one critical -- and the critical one was in the adapter to the model the
benchmarks run against.

- **Critical: a tool call cut off mid-argument was sent.** The Anthropic SDK assembles a
  message from whatever arrived, and a call cut off inside its arguments still parses:
  `"amount": 150.0` stopped after the `1` is a charge of 1. The adapter built a finished turn
  from a stream whose connection dropped, and rewrote a missing stop reason to `end_turn`; and
  nothing read `stop_reason`, so a reply that ran out of `max_tokens` in the middle of a call
  was acted on as it stood. A reply that did not finish is now a failed turn, and one that
  stopped with a call in it -- out of tokens or context, or by a refusal -- is refused before it
  is journaled.
- **A read between two writes saw both.** A reply that charges, looks the customer up, and
  charges again, with the second charge guessed: the guess joined the branch when the turn
  ended, so the drain that sent the first charge sent the second with it, and the lookup --
  deferred until after the first -- ran after both. Each confirmed guess now joins the branch
  only when the loop over the reply reaches it.
- **Two requests at once to one wrapped LangGraph ran as one.** `wrap()` built one Scheduler
  per graph, and a Scheduler keeps its run on itself: the second run took over the first, both
  runs' charges were journaled under the second, and the first never finished. Each call now
  gets its own Scheduler; a Scheduler refuses a second run, a buffer a second run at once, and
  `run` a run id the journal already holds -- which would have started it over.
- The run lock: SQLite's was keyed by the path as given, so a symlink was a second lock, and
  named after the run id, so on a filesystem that ignores case two ids were one; Postgres's
  used a 32-bit key, on the writer's shared connection, which `evict_writer` closed with the
  runs still going. It is now keyed by the real path and a hash, and on Postgres 64 bits on a
  connection of its own that is checked before every send -- a run whose lock has gone stops
  rather than send. A dead letter took its stage index as its place in the send order, and
  a correct run whose last effect failed was reported out of order; an adoption moved the
  guess's write before journaling the move, and a failed append left it to be sent; and
  `--journal` was a `Path`, which folded a Postgres DSN into a SQLite file. All fixed.

Fixing them found two more of the last kind: `specunode.Runtime` mangled a DSN the same way,
and the config's `journal` section was read by nothing, so a journal configured there was not
the one any command used. It is now the CLI's default.

### What the tenth review found

No critical finding, for the first time; seven others, three major.

- **Two resumes of one run at once sent a charge twice** -- one sent it, the other asked the
  upstream while it was in flight and sent it again -- and the new refusal to overwrite a
  settled claim then dropped the second send from the journal. A run is now driven by one
  process, and one task in it, at a time (`Journal.hold_run`); the lock dies with its holder.
- **A read after a write in the same reply was handed the value from before the write** when
  reads were issued early, which is the default. The hazard was detected and counted, and the
  read ran anyway. It now waits and runs after the write. A test that called its read
  "independent" was reading the very row the write changed; it now reads something else.
- **A stream that ended without completing its turn** was read as complete, and a guess it had
  confirmed was sent with no decision on disk. It is a failed turn now.
- A call_turn that failed on the runtime's side left its stream to the garbage collector, and
  the turn open until then; the adoption loop sat outside the failure cleanup; the ledger
  checked the order effects left against stage index, not position, and called a correct run
  out of order; and `specunode resolve` rejected the key as printed, ellipsis included. All
  fixed. The real-model results in RESULTS.md predate the read fix, and say so.

### What the ninth review found, in the eighth review's fixes

Another last look. Eight findings; the critical one was older than the week again.

- **Critical: a retry was never marked in flight.** A claim whose last attempt never left --
  a dead letter healed and resumed, one resolved as never sent, or crash window W3 -- was sent
  with its row still reading "never sent". A crash after the upstream took it made the next
  resume send it again. Taking up such a claim now re-arms it in the same statement that wins
  it, so a crash while it is out is the lost reply it may be.
- The failed-turn fix restored the branch's open-turn count from a snapshot, which was wrong the
  moment a node had another turn in flight: the count stuck at one, refusing every later write,
  or fell to minus one, turning the guard off. A turn read by `call_turn` now closes itself when
  it fails, because nothing of it reached the node.
- The same fix discarded every write that appeared on the branch during the turn, including one
  another task of the node had decided and sent -- and a concurrent drain could still send a
  confirmed guess mid-stream. A confirmed guess's writes now stay on its own branch, which
  nothing drains, until the turn is journaled; a failed turn discards exactly those.
- `specunode resolve` did not accept the key `specunode ledger` prints; a resume that
  dead-lettered again showed two rows for one effect; a resolve read the claim and settled it
  in separate steps, and a settle could overwrite a claim already settled as sent; and a tool
  that reported `sent` as anything but "no" or "maybe" was retried. All fixed.

Fixing them found one more. Holding a confirmed guess's writes off the branch until the turn
is journaled hid them from the hazard check at the next fork, which had seen them only because
they were adopted early -- so a guess could read what the model had just asked to write. The
check at a fork, and on a guessed call, now sees every write the turn has emitted, which also
closes the same gap for writes the model emitted itself.

### What the eighth review found, before the first release

One reviewer, told this was the last look before a version that cannot be withdrawn, and to
break the seventh review's fixes. Nine findings, two of them critical -- and the worst was
older than any of this week's work.

- **Critical: a charge whose reply was lost was sent again, with no crash.** The dispatcher
  retried every failure, whatever the tool declared. A gateway that took a charge and then timed
  out on the reply was charged a second time on the next attempt, under the default settings.
  A retry now happens only when the failed attempt demonstrably sent nothing, or the tool
  declared a repeat harmless; otherwise the runtime asks the tool's `reconcile`, and without
  one dead-letters it.
- **Critical: a dead letter recorded only its last attempt.** An attempt that may have landed,
  followed by one that was refused, was written down as never sent, and a resume then asked the
  model again. A dead letter is now "maybe" if any attempt, in this process or an earlier one,
  may have landed.
- A resume retried every dead letter, including one that may have landed, and applied it twice.
  It now retries only one that provably never left; for the rest, someone who has checked the
  upstream records what happened with the new `specunode resolve`, and the resume goes on.
- Model calls made at once were matched against the journal in the order their answers came
  back rather than the order they were asked, so none was served.
- A guess the model confirmed just before its stream failed was adopted and then sent by a node
  that caught the failure -- on a turn never journaled. A failed turn now discards what it
  adopted.
- A failed `call_turn` left its turn open, so a node that asked again and wrote on the new
  answer was refused. It no longer does; a stream the node read itself still does, on purpose.
- Serving was all or nothing per attempt, so a bad answer after a charge went out was pinned
  forever. Only the turns up to the last thing that may have been sent are served now.
- The refusal covers the whole node run, not the turn that decided the write -- kept, because
  the runtime cannot tell which turn decided it, and now documented with an accurate message.
- The release notes and the docs made claims the code did not keep. They were rewritten.

### What the seventh review found, in the sixth review's fix and beneath it

One reviewer, given the change that made a resume serve journaled turns, and told to break
its claims. Six findings, every one reproduced here with the reviewer's script before it was
fixed.

- **A write could leave before the turn that decided it was journaled.** A node that read
  `session.model.stream()` itself and wrote as a block parsed parked on the staged write, and
  the scheduler drained it: the journal read `effect_dispatched` before `model_response`, and a
  node that stopped reading early never journaled the turn at all. Older than the serving
  change, and under it: the claim "nothing is dispatched before its turn is journaled" was
  false on that path. A branch now counts the target turns it has asked for and not yet
  journaled, and a write on a branch with one open is refused.
- **After a question changed across one crash, a second crash's resume charged a third time.**
  Serving kept each position's longest attempt, whose questions the node no longer asked. It now
  matches the resumed node's questions turn by turn and serves the latest attempt that asked
  the same ones.
- **An answer that had sent nothing was pinned forever.** A model that named a tool that does not
  exist dead-lettered the call, and every resume was served the same answer, the model never
  asked. Only an attempt that may have sent something is served now; a dead letter records
  whether its request left.
- Serving lived in one slot on a model object that runs share, so a fresh run on another
  scheduler switched off a resume in flight; it is kept per run.
- The documentation overstated: serving does not apply on the LangGraph path, a call's arguments
  can change without its question changing, and an idempotent tool may be handed the same call
  twice.
- The kill/resume sweep could not see any of it: in the support agent the node that asks sends
  nothing and the node that sends asks nothing. It now also runs a node that asks and charges in
  one step.

### What the sixth review found, in the changes made to get CI green

One reviewer, given the day's fixes to timing-dependent tests and told to find what they got
wrong. Six findings; the worst was not in the tests at all.

- **A resume asked the model again for a turn the journal already held.** A node whose branch
  had not retired was run again, and asked its question again, even when the answer was on
  disk. So "never duplicated" rested on a real model answering the same way twice, over a
  wider window than the documentation admitted -- `journal.py` said "Resume never re-asks".
  A resumed node is now served the journaled answer when it asks the same question
  (`RecordedTurns`), which makes the resumed run decide what the dead one decided. Found by
  killing a run just after a turn was journaled; the demo had been adjusted to the old
  behaviour the day before, by a fix that treated the symptom.
- **The new kill/resume test passed with a resume that sent nothing.** It accepted any run that
  fell short as long as something was dead-lettered. With the kill points now deterministic,
  each has its exact expected outcome, worked out from what the kill left on disk; the
  reviewer's planted resume, which treated every fresh claim as ambiguous, fails it.
- **No kill point fell between a claim and the world.** Dying after the world applied an effect
  was covered, and dying before the request arrived was not. Both sides of each mutation are
  points now.
- The test's summary line and this log miscounted its outcomes; a process kill cannot test that
  an append reached the disk, which a docstring claimed; and the slow-journal plugin slowed only
  one of the four kinds of durable write it was described as slowing. Corrected, and the plugin
  slows all four.

The three defects in code and tests each have a test that fails without the fix. The other
three were claims in comments and in this log, and are corrected.

### What the fifth review found, in parallel nodes and crash recovery

One reviewer, told to break what the last three commits added -- parallel nodes, the agent
loop, replay keying -- and to prove every finding with a failing test. Nine findings, every
one reproduced here before it was fixed, and every fix checked by undoing it and watching its
test fail. Nothing reached the world from a branch that never retired in any of them; what they
found was that **a crash could make a resume send an effect twice**, which is the other half
of what this runtime promises.

- **Critical: a crash just after a group re-sent the next node's effect.** The last lane
  journaled its own position, not the group's, so the next node resumed at a lower position,
  under a key the dedupe table had never seen.
- **Critical: a crash in the middle of a group re-sent an unretired lane's effect.** A resume
  asked the router again, which saw the state some lanes had already committed and named the
  rest under new visit counts. A group is now journaled whole before any lane forks, and a
  resume finishes it -- same node ids, same position, same starting state.
- A reducer combined parallel lanes by replaying one lane's positional patch on top of
  another's commit: ``append`` duplicated items, ``last_write`` produced a value neither lane
  wrote. Lanes now commit by value, as LangGraph combines a fan-out's updates.
- A failure at a lane's retirement other than a state clash escaped the group, leaving later
  lanes parked forever and their forks unresolved; a clash found at commit left its lane
  "confirmed" for good, and the error never said its effect was already out.
- Abandoning a lane cancelled it before closing its buffer, so a ``finally`` that wrote hung
  the run -- and the wait swallowed a cancellation of the run itself.
- A turn torn down while settling left its early-issued reads running past ``run_finished``.
- Replay of a resumed run served the dead process's answer instead of the one the run kept.
- ``redacted_thinking`` and every other block type without a class was dropped from the reply
  the agent loop echoes back, which the API refuses.
- A router could return a set, whose order changes from process to process.

Fixing them found one more: a run cancelled from outside left its node bodies running,
reading upstream for a run that was over. It is fixed on both the one-node and the group path.

### What the third audit found, in the second audit's fixes

Three criticals, all inside the lattice-rule-E3 wiring committed the day before.

**Turning speculation on disabled the check that makes speculation safe.** `adopt()` moved a
confirmed speculation's staged *effects* to the branch that retires and left its `read_set`
behind on the child. A confirmed speculation never retires, so `validate_reads` never ran over
it — and the reads it made are by definition the ones issued on a guess, which are the only
reads E3 exists to re-check. Measured: with speculation off the run refused and nothing reached
the world; with it on the run returned `ok=True` and the write went out.

**`_run_in_node` discarded `_retire`'s verdict** — the twin of a bug fixed three lines above it
in the same commit. **A re-probe that raised retired the branch as if fresh**, and `unreadable`
was excluded from `witnessed`, so a run where every probe errored rendered as `0/0 fresh`.

One agent was asked only to plant bugs in `src/` and report which ones the three mandatory tests
failed to catch. **Three plants survived**: an effect dispatched with different arguments than
were staged (the world join compared idempotency keys, which are labels, and never the call);
Rule 3's ordering invariant never being applied to a real scheduler run; and the Rule 13
fail-closed refusal being deletable outright. All three are now caught.

It also found `specunode resume <unknown-run-id>` starting a fresh run and dispatching every
write in the workload — under brand-new keys the dedupe table could not match — and
`specunode verify-ledger` being structurally incapable of verifying anything, because
`sign-ledger` printed the signature and stored it nowhere.

### What the second audit found, in the first audit's fixes

**`stage()` ignored the program position the caller had just reserved.** `Branch.reserve_step`
exists precisely so a call's position cannot depend on whether the runtime speculated — its own
docstring says so — and `StoreBuffer.stage` went on deriving the step, and both idempotency
keys, from a cursor that `reserve_step` leaves as a running *maximum*. Three harms, all
reproduced: a resume with a different speculation outcome charges a non-idempotent card a second
time; Hard Rule 9 fails; and a turn of `[write, write, read]` collapses both writes onto one
position and loses one of them after the other has already reached the world — with no crash and
no speculation at all. Every shipped workload puts reads before writes, which is why the suite
was green.

**Two fixes in one commit cancelled each other.** `_resolve_prediction` stopped signalling a
park at adoption, with a comment explaining exactly why that ordering mattered — and
`mark_parked`, changed eight lines away, began resolving through adoption, so the adopted child's
own staging woke the parent mid-stream anyway.

**A squashed branch could be promoted to CONFIRMED and drained**, on the self-driving path, which
threw away the outcome from `_quiesce` and retired regardless. `Branch.confirm` had no lifecycle
guard while `Branch.retire` did.

**Lattice rule E3 had no caller.** `validate_reads` — the retirement-time witness re-check the
spec calls "the ordering that carries most of the integrity" — was defined, unit-tested, and
invoked only by a test and an attack script that hand-build a `Branch`. `policy.on_stale_read`
was dead configuration that was nonetheless journaled and rendered as though it applied.

**`BranchStatus.STALLED` was unreachable**, so the ledger's stall list was empty on every run
ever produced, on runs where hazards demonstrably fired. **`READ_AFTER_STAGED_WRITE` could not
fire for the one shape it is named for**, because a turn stages its writes after the stream ends
while reads are issued as their blocks parse. And **three MCP defects composed into a realistic
double-write**: a confirmed write returned nothing to its caller, a timed-out write was reported
as discarded while still being held, and one decision dispatched every matching held write.

### What the first audit found, after I had called it done

Six reviewers, told to assume the author had fooled himself, each finding then adversarially
verified. 23 confirmed, 1 dissolved. Two were critical and both were in code I had committed
that same day as a fix:

**The deadlock was not fixed.** `adopt()` is a point-in-time move, and the model can emit the
confirming block while the speculation is still awaiting its own journal append — so it moved
zero effects and the child staged into a list no drain visits. `_adopted_into` existed, was
recorded, had a public accessor and a comment describing exactly this guard, and was read by
nothing. Reproduced hanging at the shipped bench's own block delay.

**The step-cursor fix was a coincidence.** A child burned two positions for one call and the
parent's compensating advance cancelled it only when one early read happened to be pending at
fork time. The real fault was that positions were handed out at *execution* time while early
issue and speculation both execute out of program order. Fixing it properly exposed a third
defect: effects reached the world in a different **order** with speculation on.

**The MCP proxy could not register a single tool** — shipped, documented and "tested" in a state
where it died before serving one request, because only its rules were tested and never its
transport. Three more defects sat behind that one.

**The Postgres backend was unreachable and its test proved nothing**: `Journal(dsn)` coerced the
DSN to a file path and opened SQLite. Postgres 16 turned out to be installed on this machine, so
it is now genuinely run — and three further defects had to be fixed before one row landed.

**`bench/online/run_latency.py` did not exist** while four documents named that path and this
report described it as written and blocked only on a credential.

**Two checks cancelled each other out.** Rule 13's stamp was hard-coded true and its counter was
hard-coded non-zero; a flag that never varies and a counter that never varies read as a working
check while neither half does anything. The same shape appeared twice more: `CallScope.speculative`
was never set on a model request, and the leak test's second invariant could not detect the fault
class its own docstring named.

The pattern underneath all of it, in one reviewer's words: *"the mechanisms are written,
unit-tested and eloquently justified, but four of them are never wired into `Scheduler`, and a
grep for call sites disproves what the docstrings assert."* And about the honesty apparatus I
was proudest of: *"it verifies numbers and vocabulary rigorously, and never once asks whether a
named file exists."*

### What the earlier coverage work found, and why none of it was visible before

The three mandatory tests originally ran on one workload at one tier, the demos were one of
three, neither CLI command the spec names existed, and the MCP proxy had never been driven by a
client. Closing those four gaps found **nine defects**. Two were fatal to the project's central
claim and one had been shipped, documented and "tested" in a state where it could not start.

**A confirmed prediction of a write deadlocked the run.** The child branch staged the effect, the
canonical branch adopted its task and awaited the ack, and nothing drained the child's buffer —
the drain dispatches by branch id and only the canonical branch retires. The run hung rather than
failing, which is the worst of the three outcomes because nothing reports it. **This is the case
the project is named for.** It survived because no test had ever confirmed a prediction of a
*write*: the stub drafter predicts a read for its confirm case, and reads stage nothing.

**Speculating changed the idempotency keys.** A confirmed speculation did not advance the
canonical branch's step cursor, so every later call derived a different key depending on whether
the runtime happened to speculate. A resume with speculation off would not have deduped against
a crashed run that had it on, and the effect would have been delivered twice. A Hard Rule 9
violation, caught by the equivalence test the instant a workload confirmed a prediction.

**The MCP proxy could not register a single tool.** Its forwarding function was annotated with
the package's recursive `JsonValue` alias, and the SDK builds each tool's schema from the
signature, so registration raised `PydanticUserError` and the proxy died before serving one
request. Behind that: it read `inputSchema` where the SDK had renamed the field to
`input_schema` and so advertised a schema it had inferred; it then rejected every call made
against that schema, because the SDK parses arguments from the signature rather than the schema;
and the first forwarded read deadlocked, because `MCPServer.run("stdio")` opens its own event
loop and left the served tools and the upstream session on different ones.

**A read still in flight at retirement demoted its branch.** `_timed_read` marked a read
speculative by saving the branch's `status`, setting SPECULATIVE, and restoring the saved value
— so a read that outlived the turn put the old status back after the branch was confirmed, and
the next drain refused with a Hard Rule 3 message about a branch that had been confirmed
correctly. Overlapping a read with the drain is the whole point of early issue, so the race was
reachable on the ordinary path. The same shape could have promoted a squashed branch.

**Park events are keyed by branch id**, so the event a child set when it staged was on a key
nothing waits on. **The journal could not tell a predicted branch from the canonical one** —
every `branch_forked` recorded `predicted: null`, and a confirmed speculative branch was never
journaled as resolved at all. **A node failure was reported as `node <name> failed`**, discarding
the branch's reason — which for a replay refusal *is* the step index and the field-level request
diff, the single most useful thing this system can tell an operator.

Two further things are findings rather than defects, and are in `docs/limitations.md`: a drafter
cannot use the result of the call it was just asked about, because it is consulted immediately
after a block parses when that call has only been issued; and two of the three sample apps route
around speculation entirely by calling the model directly rather than through `session.call_turn`.

The honest reading of this section is that the ratio of defects to coverage was uncomfortable,
and the next axis of coverage probably has more in it.

### Decisions this spec did not specify

The Progress Log has 130+ entries; these changed the shape of the build.

1. **`Journal.read`'s `after` defaults to −1, not §7's 0.** Offsets are dense from zero and
   `after` is exclusive, so the specified default silently skips every run's first entry.
2. **`StoreBuffer.stage` is async.** The `effect_staged` entry is fsynced before the branch is
   told the effect exists.
3. **Three key derivations, not one.** `key` carries the branch lineage and stays internal;
   `nkey` drops it and is both the dedupe primary key and the token the tool sees; `ekey` drops
   the run too and is used only by the equivalence relation.
4. **A staged write always returns a future, never a value.** Handing back a real ack means
   dispatching inside the call, which is task 1.6's planted bug.
5. **The leak test asserts two invariants.** The spec's own invariant cannot see the bug the spec
   names as the planted one.
6. **Hard Rule 1's check parses the AST.** A grep cannot tell a protocol signature from code that
   authors a prompt. The literal grep the spec names is kept alongside it.
7. **Projections and handle-accepting tools are not implemented at all.** Each is independently
   forbidden by Rules 4, 8 and 9, and a projection can make Rule 9's mandatory comparison fail in
   a supported configuration.
8. **A ninth hazard, `NODE_NOT_SPECULABLE`.** A refusal that is not named is missing from the
   histogram.
9. **Task 2.5's Verify is satisfied in a weaker, truer form.** A resumed run reaches a *prefix* of
   the uninterrupted run's effects: at the two-generals window a non-idempotent tool is
   dead-lettered rather than redelivered. The test asserts never-duplicated and never-invented.
10. **The MCP proxy's mode is read from the client's advertised capability**, not from a default.
11. **The corpus effect-class table is hand-written**, because classifying by inspecting an
    argument is the heuristic Hard Rule 2 forbids.
12. **The three sample apps are three different shapes**, not three instances of one, and each
    workload declares `drives_turn` and `tier_1_can_predict` so that a workload which silently
    stopped speculating cannot still pass every comparison.
13. **A confirmed speculation's effects are adopted by the canonical branch**, re-attributed but
    never re-keyed — `nkey` is the token the tool sees.
14. **`Config` gains `graph: "module:attribute"`.** A journal records what a graph did, not what
    it is, so `resume` and `replay` cannot find one without being told; both refuse rather than
    import something plausible and re-drive the wrong program.
15. **`specunode replay` dispatches nothing unless `--dispatch` is passed**, and the dry run is
    recorded on the ledger row rather than only in a banner.
16. **The proxied MCP tool is given the upstream's parameter names, typed `Any`.** The upstream is
    the authority on its own argument types, and re-deriving Python types from JSON Schema would
    invent disagreements.

### Definition of Done: what is not ticked, and why

| Item | Status |
|---|---|
| Wheel on 3.11/3.12/3.13, macOS **and Ubuntu** | Verified by CI on all six, with the test suite and a Postgres 16 job, since the repository went public (2026-09-25). |
| The three tests on every workload, every tier, **every CI job** | They run, are never skipped, and cover all three sample apps at tiers 0 and 1. Tier 2 is a draft *model* behind an optional extra — requiring it would make a mandatory test skippable, which is the one property these files may never have, so it has its own tests. The open clause is **every CI job**. |
| Offline ✓, overhead ✓, adversarial ✓, **online latency** | The runner exists and CI exercises all of it on every push with `--model scripted`. A real measurement needs an API key; a scripted report is stamped `is_real_model: false`. |
| Published to PyPI; demoed from the published wheel | Needs credentials. |

**The Postgres journal backend now runs.** Postgres 16.15 was installed on this machine, so it
was stood up and the journal driven against it: append, read with an exclusive `after`, the
dispatch-claim table and chain verification, with every row read back through a `psycopg`
connection the `Journal` knows nothing about. Three defects had to be fixed first — it was
unreachable, it never set `row_factory=dict_row` while every query indexes by column name, and
the writer caught only `sqlite3` exceptions. The tests now prove which backend they are on
before they assert anything, because the previous version passed by writing SQLite to a file
named after the connection string.

**Four CI jobs would have failed at install.** `uv sync --all-extras --dev` pulls the
`local-draft` extra's `torch`, which has no wheel for every runner in the matrix. They now
install the extras the tests need.

Phase Gate 4's blocking half is met: the proxy works with a generic client, demonstrated against
a real upstream server. Its remaining clause — that the ledger match the LangGraph integration's
exactly — cannot hold while `node_id` is a mandatory key input and a generic client reports no
node. That is a design decision for you, not a bug: accept the node-insensitive comparison, or
accept that the clause holds only for clients that report node ids.

### Manual steps left for you

1. **Done, 2026-09-18:** a key was supplied and the real-model runs in 6.4, 6.8 and 6.10 were
   made, each under its cap and each recording its spend in its results file, and 6.11 on
   2026-09-23. Still to pay for: re-running 6.4 with the corrected arms. The original
   instructions follow.

   **An Anthropic API key and a spend cap**, for the online latency benchmark (task 6.4). Set
   `ANTHROPIC_API_KEY` and `SPECUNODE_BENCH_BUDGET_USD` (default 25), then run
   `python bench/online/run_latency.py --out bench/results/latency.json`.

   The runner exists and CI exercises the whole of it on every push via
   `--model scripted`: the three arms, the timing, the spend accounting, the budget gate and
   the bootstrap. What a key buys is the one thing a stand-in cannot provide, which is real
   model latency; a scripted report is labelled `is_real_model: false` and says in its own
   output that its numbers are not figures anyone should quote.

   Decision Gate D2 says to report the reduced *n* and its wider interval rather than raising
   the cap. The runner halts at the cap, records where it halted, and a test drives that gate.

   **An earlier version of this report said the runner was "written to do that" while
   `bench/online/` was an empty directory.** That was false, in the section of a project whose
   stated purpose is to make such a claim impossible. It is recorded here rather than quietly
   corrected.
2. **PyPI credentials**, for task 9.3. `uv build` works and the wheel installs and runs on 3.11,
   3.12 and 3.13 locally.
3. **Done, 2026-09-25: CI runs.** The repository is public at
   https://github.com/Poojan6216/specunode. Its first run failed every test job on causes no
   local run could show -- a test reading a file that is never committed, and three kill tests
   whose timing assumed a laptop -- all fixed. Later runs found two more one at a time: a demo
   that handed a resumed run the wrong turn, and a guess that had to stage its write within
   25 ms. Rather than wait for the next, the whole suite was run with every journal append
   made 40 ms slower (`tests/slow_journal.py`), which found every test of that kind at once:
   four, each of which now waits on the event it needs instead of on the clock. The `slow-disk`
   CI job keeps it that way. The kill tests, which had been calibrated three times, now kill
   at counted points in the run instead of after timed delays, so no runner can miss them.
4. **Decide Phase Gate 4's ledger clause**, as above.

Decision Gate D1 did **not** fire: the corpus was fetched from Hugging Face, so the opportunity
analysis is on real trajectories rather than self-generated ones.

### What I would do differently with another month

**Find a corpus where the mechanism can work.** The measured zero is a real finding, but it is a
finding about OpenHands' one-call-per-turn shape rather than about agents in general. A workload
that emits several tool calls per turn is where past-write speculation has room, and I would go
looking for one and publish both numbers.

**Measure wall clock against a real model.** Every latency claim in this design is currently an
argument. The overhead number says 6.6 ms per step; whether that is noise or a tax depends
entirely on numbers that need an API key.

**Test transports, not only rules.** The single most expensive mistake here was testing the MCP
proxy's rules without its transport. The reasoning was sound — the rules carry the correctness
claims — and the result was a component that had never once started. The audit found the same
shape four more times: the Postgres backend, the online runner, Rule 13's gate and Demo 1's own
headline row. Every place this codebase tests a policy without the thing that carries it
deserves the same suspicion.

**Get somebody else to look, and then do it again.** I closed four coverage gaps, found nine
defects, wrote a Final Report and called it done. An independent audit found 23 more behind 730
passing tests. I fixed all 23, verified every fix against the broken code, and wrote the report
again. A second audit found 17 more — including four criticals, two of them *inside* the fixes
I had just written, one of them two fixes from the same commit cancelling each other out.

The counts are 23, 17, 13. That is a decline and it is not a convergence, and nobody should read
the third number as "nearly done" — the second audit said the same thing about the second number
and was wrong. The useful question is not whether this codebase is finished but how much
independent scrutiny per change it turns out to need. For work of this shape — one author, deep
invariants, a test suite written by the same person who wrote the bugs — the answer measured
here is: a great deal more than feels necessary at the time, and more than one round.

**Nine of the fifty-three defects were inside repairs**, and every one of those repairs was
written immediately after finding something, which is exactly when judgment is least
trustworthy and feels most reliable.

**Distrust a fix more than the defect it repairs.** A defect is written once. Its repair is
written by the same judgment, under more time pressure, with the satisfaction of having found
something — and it touches code that is by definition subtle enough to have been got wrong
already. Nine of the 53 defects here were in repairs. Every fix in the last three rounds was
therefore verified by reverting it and watching the new test fail first, which is cheap and
caught two tests that would otherwise have passed against the unfixed code.

**Widen the coverage matrix again, and expect it to find more.** One workload at one tier became
three at two, and one demo became three, and that found nine defects. That is not a comfortable
ratio, and the honest conclusion is that the next axis — more fault injection, more turn shapes,
more adapters — probably has more in it.

**Spend the remaining time on the predictor, not the buffer.** The store buffer works and its
guarantees hold under every fault I could inject. The number that decides whether any of it pays
for itself is the acceptance rate, and it is **not measured anywhere in this repository**. The
0.5350 figure is signature accuracy — right tool, right argument names — while the gate compares
argument values exactly. Measuring the real thing is a prediction problem, not a runtime one,
and it is the first number I would go and get.

**Addendum, 2026-09-17 — that number has been got, and it is zero.**
`bench/offline/run_acceptance.py` grades the tier-1 drafter with the runtime's own gate over all
300 trajectories joined to their argument values: 0.0002 with guesses carried across
turns, 0.0000 under the policy the runtime runs, against a ceiling of 0.0984 for that
grading. The paragraph above stands as written; its recommendation has been carried out, and the
result is the strongest negative result in this repository. On this corpus the tier-1 predictor
as built is worth nothing, and the case for the store buffer now rests entirely on a predictor
that generates values, which is unmeasured.
