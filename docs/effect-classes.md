# Effect classes

Every tool carries exactly one effect class, and **you** declare it. Nothing infers it.

> **A tool whose upstream enqueues, schedules or triggers work is a `WRITE`**, regardless of
> its synchronous response or its HTTP verb. `{"status": "queued"}` is not a read result.

## The four classes

| Class | What it means | When it runs |
|---|---|---|
| `READ` | Returns information. Changes nothing. | Immediately, including on a speculative branch. |
| `WRITE` | Changes something. Could be undone by hand if you had to. | Staged. Dispatched only when the branch retires. |
| `COMPENSABLE` | A `WRITE` that names the tool which undoes it. | Staged like a `WRITE`. The compensator is a *second* effect, used after retirement. |
| `IRREVERSIBLE` | A `WRITE` with no undo at all — an email sent, a card charged, a webhook delivered. | A speculation barrier by default. `policy.stage_irreversible` can stage it; the default is off. |

## Why an undeclared tool is a WRITE

`ToolRegistry.get` never raises for an unknown name. It synthesises `WRITE(idempotent=False)`,
warns once, and carries on.

The asymmetry is the point. A tool wrongly classified `WRITE` costs a speculation the runtime
could have made. A tool wrongly classified `READ` is *executed* on a branch that may never
retire — an effect in the world that no decision authorised. So the default is the direction
whose worst case is lost performance rather than a wrong charge.

## Declaring them

In code:

```python
from specunode.core.effects import EffectClass
from specunode.integrations.plain import tool

@tool(effect=EffectClass.READ, witness=True, forward_keys="customer:{args.customer_id}")
async def lookup_customer(customer_id: str) -> dict:
    ...

@tool(effect=EffectClass.WRITE, idempotent=False, forward_keys="customer:{args.customer_id}")
async def charge_card(customer_id: str, amount: float) -> dict:
    ...
```

In `specunode.yaml`, for tools you did not write:

```yaml
tools:
  send_email: { effect: irreversible }
  create_ticket:
    effect: write
    idempotent: true
    forward_keys: "ticket:{args.customer_id}"
```

The config table wins over everything, including an MCP server's own annotations. You run the
agent; you carry the consequences of a wrong class, not the server's author.

## From MCP annotations

| Annotation | Class |
|---|---|
| `readOnlyHint: true` | `READ` |
| `destructiveHint: true` | `IRREVERSIBLE` |
| `destructiveHint: false, idempotentHint: true` | `WRITE(idempotent=True)` |
| no annotations | `WRITE(idempotent=False)` |

If the servers you use ship no annotations, everything defaults to `WRITE` and the per-tool
override table is the only route to any speculation at all.

## The three flags

**`idempotent`** — whether a second delivery of the same call is harmless upstream. It is a
claim, not something the runtime verifies, and it decides one thing: what happens when a crash
leaves a dispatch ambiguous. `True` means redeliver; `False` means dead-letter and stop for a
human. Choose it by asking "would I rather this ran twice, or stopped and waited for me?"

**`witness`** — whether a `READ` returns `{"value": ..., "witness": ...}`, where the witness is a
version, ETag or row counter. With one, the runtime re-checks before the branch retires and
squashes if it changed. Without one, the read is reported *unwitnessed* and is never counted as
fresh.

**`forward_keys`** — the resource keys a call touches, as literal text plus `{args.<name>}`
placeholders and nothing else. It is how the runtime notices that a read touches something a
staged write touches, and it is the only detector for that case. Undeclared means "unknown",
which conflicts with everything: a branch with any staged write stalls at its first undeclared
read. That is lost speculation, not a wrong answer — but it does mean past-write speculation
needs `forward_keys` on both sides to buy anything.

## The trust boundary

The runtime cannot check any of this. A `READ` that writes, a `WRITE` that claims to be
idempotent and is not, an under-declared `forward_keys` — each defeats a different mechanism,
silently. `bench/adversarial` measures what each one costs rather than arguing they are
unlikely, and [limitations.md](limitations.md) lists them together.
