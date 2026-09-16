# The MCP proxy

For a developer who cannot change their agent's code. Point your MCP client at the proxy
instead of at your server, declare your tools' effect classes in `specunode.yaml`, and writes
are held until the model's real decision confirms them.

```
specunode mcp-proxy --upstream "python my_server.py" --config specunode.yaml
```

## The difficulty, stated first

**The proxy cannot see the model.** It sees tool calls arriving and results going back; it never
sees the prompt or the model's decision. That has two consequences, and neither is worked
around — they are what this integration *is*.

**Branch resolution has to arrive out of band.** Something must tell the proxy what the model
actually decided. The LangGraph and plain-Python integrations do it automatically because they
are inside the loop; over MCP, either the client reports it or a person does:

```
specunode retire <run-id> --step 4 --decision '{"tool": "charge_card", "args": {...}}'
```

Until a decision arrives, nothing held is sent. `specunode.status` says what is being held and
that it has not happened.

**Hard Rule 13 cannot be enforced here.** The rule says a placeholder must never enter a model
prompt. The proxy has no visibility of prompts, so it cannot check. Every run through the proxy
is stamped `context_identity: unenforced`, and that stamp is the honest answer rather than a
disclaimer — the in-process integrations stamp `enforced` because they actually checked.

## Two modes, and why the client chooses

A staged write has no result to return. What the proxy hands back instead depends on what the
client said it understands.

**A client that advertises `specunode/decisions`** gets the real thing: the call returns a
placeholder handle with a `_specunode` block saying the write has not happened, and the buffer
drains when a decision arrives. This is the mode that buys latency.

**Any other client blocks.** The call does not return until a decision arrives. That is slower
and it is the correct default, because a client that does not understand a handle will put it
straight into its next prompt — which is exactly what Rule 13 forbids, and the proxy cannot see
the prompt to stop it. Returning a handle to such a client would trade a correctness property
for latency without telling anyone.

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
