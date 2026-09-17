# The MCP proxy

For a developer who cannot change their agent's code. Point your MCP client at the proxy
instead of at your server, declare your tools' effect classes in `specunode.yaml`, and writes
are held until the model's real decision confirms them.

```
specunode mcp-proxy --upstream "python my_server.py" --config specunode.yaml
```

## What was actually wrong for most of this build

The proxy's rules were tested from the start and its transport was not, on the reasoning that
the rules carry the correctness claims and the SDK does not. That is defensible right up until
it is the only thing tested. Driving the whole chain — a generic `mcp` client, the proxy, and a
real stdio upstream server — found four defects in a row, none of which a rules test could see:

1. **It could not register a single tool.** The forwarding function was annotated with the
   package's recursive `JsonValue` alias, and the SDK builds each tool's schema by making a
   pydantic model from the signature. Registration raised `PydanticUserError` and the proxy
   died before serving one request. The proxy's own six control tools had the same problem.
2. **It advertised a schema the upstream never declared.** `mcp` 2.x renamed `inputSchema` to
   `input_schema`, and a `getattr(tool, "inputSchema", None)` went on quietly returning `None`.
   A missing schema field is now a startup failure with the SDK version in the message.
3. **It rejected every call made against the schema it advertised.** The SDK parses arguments
   against a model built from the *signature*, not from the advertised schema, so a bare
   `**kwargs` forwarder asked clients for a literal `kwargs` field. The forwarder is now given
   the upstream's parameter names.
4. **The first forwarded read hung.** `MCPServer.run("stdio")` opens its own event loop, so the
   served tools ran on one loop and the upstream `ClientSession` on another. `run_stdio_async`
   keeps them on one.

All four are now covered by `tests/integration/test_mcp_end_to_end.py`, which asserts on what
the upstream server *received* — from a log it writes itself — rather than on what the proxy
reports having forwarded. The startup probe checks for each SDK surface these fixes depend on,
so drift fails loudly rather than degrading into something that forwards writes it should hold.


## The difficulty, stated first

**The proxy cannot see the model.** It sees tool calls arriving and results going back; it never
sees the prompt or the model's decision. That has two consequences, and neither is worked
around — they are what this integration *is*.

**Branch resolution has to arrive out of band.** Something must tell the proxy what the model
actually decided. The LangGraph and plain-Python integrations do it automatically because they
are inside the loop; over MCP, either the client reports it or a person does:

```
# The proxy's own MCP tool, called by the client -- not a CLI command.
# There is no `specunode retire` binary; this used to be documented as one in three places.
specunode.retire {"tool": "charge_card", "args": {...}}
```

Until a decision arrives, nothing held is sent. `specunode.status` says what is being held and
that it has not happened.

**Hard Rule 13 cannot be enforced here.** The rule says a placeholder must never enter a model
prompt. The proxy has no visibility of prompts, so it cannot check. Every run through the proxy
is stamped `context_identity: unenforced`, and that stamp is the honest answer rather than a
disclaimer.

The in-process integrations do **not** stamp `enforced`; they stamp `unchecked`, because the
retirement-time rebuild Rule 13 describes is not implemented there either. They fail closed
instead — a branch that sent a request while speculating is refused at retirement rather than
approved. See `docs/adapters.md`. This paragraph previously claimed those integrations
"actually checked", which was not true of any of them.

## Two modes, and why the client chooses

A staged write has no result to return. What the proxy hands back instead depends on what the
client said it understands.

**A client that advertises `specunode/decisions`** gets the real thing: the call returns a
placeholder handle with a `_specunode` block saying the write has not happened, and the buffer
drains when a decision arrives. This is the mode that buys latency.

**Any other client blocks.** The call does not return until a decision arrives, or until the
decision deadline expires (`--deadline`, 300s by default). That is slower and it is the correct
default, because a client that does not understand a handle will put it straight into its next
prompt — which is exactly what Rule 13 forbids, and the proxy cannot see the prompt to stop it.
Returning a handle to such a client would trade a correctness property for latency without
telling anyone.

**On deadline expiry the write is still held and still unsent**, and the client is told exactly
that. It is *not* reported as discarded, because it is not: `specunode.status` still lists it and
a later matching `specunode.retire` will forward it. Telling a client its write was discarded
when a later decision would still send it is how one intended write becomes two — the client
reissues, and then both go out. Call `specunode.discard` to drop it unsent.

**A single decision authorises a single held write.** Two structurally identical staged calls do
not both go out on one `specunode.retire`; the next one can be confirmed by the next decision.
The proxy computes no idempotency key and keeps no dedupe table, so nothing downstream would
absorb a repeat.

The mode is read from the client's advertised capabilities, not from a flag with a convenient
default. `--handles` forces the first mode; use it only if you know your client.

## What the proxy classifies

Effect classes come from `specunode.yaml`'s `tools:` table, merged with the upstream server's
own MCP annotations, with the config winning. See [effect-classes.md](effect-classes.md).

A tool with no annotation and no override is a `WRITE`. If the servers you use ship no
annotations — which Decision Gate D5 anticipated — then everything defaults to `WRITE` and the
override table is the only route to any speculation at all.

## The proxy's own tools

| Tool | What it does |
|---|---|
| `specunode.status` | What is held, in what mode, and whether context identity is enforced |
| `specunode.ledger` | Effects dispatched and discarded this session |
| `specunode.stall` | Stop returning handles; block on writes instead |
| `specunode.discard` | Drop every held write unsent. Requires `confirm=true` |
| `specunode.retire` | Report the model's actual decision |
| `specunode.replay_check` | Whether held writes match a journaled run |

## Security

The stdio transport is a child process and inherits its parent's environment; nothing is
listening on a socket. The optional streamable-HTTP mode binds loopback only, requires a
per-run token, sets no cookies and checks `Origin`.

## Versions

Probed against `mcp` 2.2.0. The SDK moved from 1.x to 2.x with a breaking change to the server
API, so the proxy probes on startup and **fails loudly** if the surface it drives is missing,
rather than degrading into something that forwards writes it was meant to hold.

The staging rules live in `ProxyState` and are tested without a transport at all. That split is
deliberate: the rules carry the correctness claims and the SDK does not, and a design that put
them inside protocol handlers would need re-verifying every time the SDK moves.

## What this mode does not give you

- **Context identity.** Stamped `unenforced`, always.
- **A journal of the model's turns.** The proxy never sees them, so `replay` needs a run driven
  through one of the in-process integrations.
- **Automatic branch resolution.** Something outside the proxy has to say what the model decided.

If you can change your agent's code, use the LangGraph or plain-Python integration instead. This
one exists because sometimes you cannot.
