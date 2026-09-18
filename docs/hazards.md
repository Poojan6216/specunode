# Hazards

A hazard is a place where the runtime refuses to guess. It does not degrade into a heuristic,
a projected value or a best effort — the branch stops speculating and the work is done the
ordinary way.

The list is closed, and each member fires at exactly one site, so the benchmark's histogram has
disjoint buckets and a run's stalls can be attributed rather than lumped together.

| Hazard | Fires when | Where |
|---|---|---|
| `UNDECLARED_TOOL` | the tool has no declared effect class | before a call |
| `RETURN_VALUE_DEPENDENCY` | a placeholder appears anywhere in the call's arguments | before a call |
| `IRREVERSIBLE_ON_PATH` | an `IRREVERSIBLE` effect would have to be staged, and `stage_irreversible` is off | before a call, on a speculative branch only |
| `READ_AFTER_STAGED_WRITE` | a read touches a resource key a staged write touches | before a call |
| `BUDGET` | speculation depth or the in-flight limit is reached | before a call |
| `FREE_TEXT_NODE` | the predicted decision is prose | when filtering a candidate |
| `NODE_NOT_SPECULABLE` | a predicted route enters a node that did not opt in | when filtering a candidate |
| `MODEL_TURN_AFTER_STAGED_WRITE` | a request would carry a placeholder, or include a slot that will never fill | before a model request |

## Precedence

First match wins, and correctness-bearing predicates come before cost-bearing ones:

```
UNDECLARED_TOOL → RETURN_VALUE_DEPENDENCY → IRREVERSIBLE_ON_PATH
                → READ_AFTER_STAGED_WRITE → BUDGET
```

`UNDECLARED_TOOL` is first because every later predicate reads `ToolSpec` fields the registry
*synthesised* — an undeclared tool's `forward_keys` and `idempotent` are fiction, so deciding
anything from them would be deciding from nothing.

Data hazards come before budget hazards so that the offline opportunity analysis, which has no
scheduler and therefore no budget state, reproduces the same histogram the online run does.

Budget hazards never apply to the canonical path. A confirmed branch that stopped advancing
because the speculation budget was spent would make the run hang rather than simply stop
speculating.

## The two detectors, and what each one cannot see

### Handle scanning — value dependencies

A staged write returns `$specunode.handle:<id>` instead of a value. Any later call whose
arguments contain that prefix depends on a result that does not exist yet.

The predicate is a **loose, case-insensitive byte scan over the exact bytes `canonical()`
produced**. Loose rather than strict because a mangled or re-cased prefix is not evidence that
no dependency exists. A structural walk runs alongside it to attribute a hit to a path and an
effect, but the scan is what decides.

| Laundering attempt | Caught |
|---|---|
| the handle as a whole argument value | yes |
| embedded in a longer string | yes |
| nested inside arrays or objects at any depth | yes |
| used as an object *key* | yes |
| prefix case-mangled | yes |
| split across two arguments *after* the prefix | yes |
| base64-encoded | **no** |
| hex- or percent-encoded | **no** |
| split across two arguments *inside* the prefix | **no** |

These verdicts are not written from memory. `bench/adversarial/run_attacks.py` runs every case
and reports the measured miss rate; the table above and that output are checked against each
other.

A prefix that resolves to no staged effect in this branch's lineage is **still** a hazard. A
sibling's handle appearing here would be a Hard Rule 6 violation, and swallowing it would hide
that.

A legitimate argument that happens to contain the literal text `$specunode.handle:` stalls the
branch. That is a false positive costing latency, never correctness.

### `forward_keys` — key dependencies

A read touching a resource a staged write touches, with no placeholder anywhere. This is a
different failure from the one above and neither detector substitutes for the other.

**This detector has no backstop.** Witness validation re-checks reads *before* the drain, and
the branch's own staged write has not been dispatched at that moment, so it cannot have made
anything stale. Nothing else stands between a staged write and a stale in-branch read.

That is why an undeclared `forward_keys` fails closed. Concretely, if it did not:

```
stage close_ticket(T1)                  -- staged, not sent
read  list_open_tickets()               -- undeclared, so no overlap is detected
      -> sees T1 still open
stage escalate(ticket=T1)               -- computed from a list that was already wrong
```

The model confirms, both writes drain, and the world receives an escalation the sequential run
would never have issued — with the leak test green and the read counted as fresh.

So: an undeclared `forward_keys` means "unknown", which conflicts with everything. A branch
holding any staged write stalls at its first undeclared read. Past-write speculation therefore
requires `forward_keys` on both the write and the read, with disjoint key sets, to buy anything
at all.

An *under*-declared one sits on the same trust boundary as a misdeclared effect class, and
[limitations.md](limitations.md) lists them together.

## The write barrier

`MODEL_TURN_AFTER_STAGED_WRITE` is the rule that keeps a model from being asked a question no
real run would ask. It fires when the request would carry a placeholder, or when any slot the
request must include is one a staged write will fill — which it never will, because the write
cannot be dispatched before the branch retires.

It is **not** "the next call only". The message list is append-only, and a branch cannot drain
before it retires, so a write staged at turn one leaves its unfillable slot in every later turn.
A flag meaning "the next call" passes the obvious test and is wrong at turn three.

The consequence is the honest limit on what speculating past a write buys: a branch can run
further *tool* calls after staging, but never another *model* call. See
[limitations.md](limitations.md).
