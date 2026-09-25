# Replay, resume and the journal

Everything the model said and everything a tool returned is written to the journal and fsynced
**before** the runtime acts on it. That is Hard Rule 5, and three things depend on it: a crashed
run can be resumed, a finished run can be replayed, and the effect ledger can say which model
decision authorised each effect.

## Pointing the commands at a graph

`specunode resume` and `specunode replay` both re-drive a graph, and a journal does not contain
one — it records what a graph *did*, not what it is. Both commands read `specunode.yaml` for a
`graph:` entry naming the callable that builds it:

```yaml
schema_version: 1
graph: "your_app.agent:build"     # -> (graph_adapter, tool_registry)
```

The callable takes no arguments and returns the adapter and the tool registry. The registry is
not optional: effect classes are declared out of band and never inferred, so a builder that
returns only a graph is refused rather than run with everything defaulted to `WRITE`.

Without the entry, both commands exit 2 and say what to add. That is deliberate — importing
something plausible and re-driving the wrong program is worse than stopping.

## What a replay sends

Nothing, unless you ask. A replay re-drives the graph, and a graph that re-drives dispatches, so
a replay of a run that charged a card would charge it again — from a command whose entire
purpose is to answer a question about the past.

```
specunode replay <run_id>              # dispatches nothing
specunode replay <run_id> --dispatch   # sends effects for real
```

In the default mode the effects are still journaled and still appear in the rendered ledger,
because the run really did decide to make them. Their status reads `DISPATCHED (dry run: not
sent)` — on the row, not only in a banner, because a rendering is something people paste into
a ticket and one that reads `DISPATCHED` for an unsent effect is a lie that travels.

A replay writes to `replay-<run_id>.db` beside the journal it is reading, never into it.
Interleaving a new run's entries with the record it is checking against would corrupt the only
evidence there is.


## The journal

One entry is one row is one statement is one commit is one fsync. Not batched. About a hundred
fsyncs on a thirty-step run costs tens of milliseconds against multi-second model turns, and
batching would open a crash window in exchange for nothing.

Because the unit of commit is exactly one row, "the last committed transaction" and "the last
entry" are the same thing. The entry being written when the power goes out is wholly present or
wholly absent; there is no half-written entry to reason about.

Entries are hash-chained per run. `prev_hash` of entry *n* is `entry_hash(n-1)`, where
`entry_hash` covers every stored column but the payload text — `run_id`, `offset`, `kind`, `ts`,
`payload_hash` and `prev_hash`. Covering `kind` and `offset` is what stops an entry being
retyped or reordered undetected. The first entry's `prev_hash` is a domain-separated constant
derived from the run id, so "this is the start of run R" is itself a claim the chain makes.

Verification also compares the bytes on disk against the canonical encoding of what they parse
to. That catches a hand-edited journal whose JSON is valid but not canonical — the payload hash
alone would not, because it is taken over the parse.

Ordering is by `offset`, never by `ts`. No query in the codebase contains `ORDER BY ts`.

## Replay

`ReplayModel` serves the journaled responses and calls no model. It is **keyed by step index**,
not sequential, so repeated requests for one step return the same turn and concurrent branches
asking for the same step cannot interleave into each other's answers.

It refuses the moment the run would ask a *different question*. The comparison is
`request_hash` — the same envelope projection Hard Rule 13 uses live — so changing one token of
the system prompt, or adding a tool to the registry, diverges at the first step rather than
somewhere downstream where the consequence finally shows. The exception carries the step and a
field-level diff.

Only the **retired chain** is served. A speculative run journals model responses for branches
that were later squashed; feeding one back as an input would replay a decision the run never
actually made.

### What the request hash covers

Included: the model id, the system blocks, the **entire** message list, the tool definitions in
registry order, `tool_choice`, and every sampling parameter — `temperature`, `top_p`, `top_k`,
`max_tokens`, `stop_sequences`, thinking configuration. Each of the three sampling parameters is
`None` unless the developer set one, and an unset parameter is recorded as unset rather than as
a value nobody chose; the current models reject all three anyway.

Sampling parameters are in deliberately. A branch that quietly sets a smaller `max_tokens` to
make its speculative turn cheap is asking a different question, and Rule 13 exists to make that
a fault rather than an optimisation.

Excluded, exhaustively: a message's runtime-only `id`, `origin_branch` and `journal_offset`;
`cache_control` blocks, which are a billing hint and not content; `stream` and the other
transport fields; and raw provider correlation ids, which are **rewritten** to positional tokens
rather than dropped. Rewriting rather than dropping matters: renaming ids consistently must not
move the hash, while a tool result attached to the *wrong* call must.

## Resume

`recover()` reads the journal and works out what a resumed run may build on:

- **committed state**, rebuilt by applying the state deltas of branches the journal records as
  RETIRED, in offset order
- **the program cursor**, restored verbatim from the `cursor_after` recorded on the last
  retirement — not inferred from the highest step index visible. Inferring lands the resumed run
  at a different program position, so every idempotency key it derives differs from its
  pre-crash value, the dedupe table misses, and effects that already went out go out again
- **dispatch claims still in flight**, which are the reconciliation list

Only the retired chain contributes. A branch that was confirmed but never retired had its drain
in flight when the process died; its state delta is not applied and its cursor is not adopted,
because resuming from it would dispatch effects that were never context-checked or
witness-validated.

### Ordering that matters

`state_delta_applied` is journaled **before** the `branch_resolved{retired}` entry that depends
on it. The other order loses a crash window with teeth: if the process dies between them, the
branch reads as retired while its state change is gone, so a resume re-runs that node from a
different program position and re-dispatches its effects. It looks like a resume bug and is an
ordering bug.

### What a resume does not promise

Dispatch is at-least-once. If the process dies between a request reaching the world and its
acknowledgement being recorded, nobody can tell afterwards whether it took effect. A tool that
declared `idempotent=True` is redelivered; one that did not is **dead-lettered** and the run
halts for a human.

So a resumed run can reach a *prefix* of the effects an uninterrupted run reached. That pair —
never duplicated, never invented — is what the kill/resume tests assert, and the dead letter is
required whenever the run falls short.

**Both halves of that pair are conditional on a resumed node making the calls it made
before.** An idempotency key is derived from the run, the node, the program position, the tool
and the *arguments*. Nothing is dispatched before the model turn that decided it is journaled,
and a resumed node whose earlier answer may already have sent something is served that answer
when it asks the same question -- so the resumed run makes the same calls, derives the same keys,
and the dedupe table catches the earlier attempt, whatever the model would have said the second
time. An answer that sent nothing is asked for again; there is nothing to protect. What a resume
cannot keep the same is anything else that shapes a call: a read made again that returns
something new, a timestamp, code that changed. Then a different call at the same position gets
a different key, and the world receives it as well.

The kill/resume test resumes every kill point with a model that would decide differently if it
were asked, on a node that asks and charges in one step as well as on one that only decides, and
holds each point to the outcome its on-disk state requires. See `docs/limitations.md`.

### On the LangGraph path

State belongs to the checkpointer. LangGraph owns its own reducers and channel semantics, and a
second copy in the journal would be a second answer to what the run's state is. So a LangGraph
resume needs a LangGraph checkpointer, and the journal's hash chain does not cover the state it
holds. The plain-Python path journals its deltas and needs nothing else.

## Where the journal lives

SQLite by default, in WAL mode with `synchronous=FULL`, at `./.specunode/journal.db`. Postgres
16 loads the same DDL unmodified — only `TEXT`, `BIGINT`, `PRIMARY KEY` and
`CREATE ... IF NOT EXISTS` appear in it, and offsets are assigned in Python rather than by the
database so the two backends cannot drift.

One writer per database file per process, on a single-threaded executor. SQLite in WAL mode
admits one writer at a time; twenty concurrent runs sharing a journal queue on it in FIFO order
rather than racing for the write lock. The fsync never runs on the event loop.
