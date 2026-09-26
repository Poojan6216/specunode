# Command-line reference

`specunode` answers questions about a run that already happened, or continues one that did not
finish. No subcommand decides anything: the journal decides, and these read it.

Every command that reads a run takes `--journal`: a SQLite file, or a `postgresql://` DSN.
Without it, the journal is the one the config's `journal` section names -- its `path`, relative
to the config file's folder and with `~` expanded, or for `kind: postgres` its `dsn` (or
`SPECUNODE_JOURNAL_DSN`) -- and without a config, `./.specunode/journal.db`. Every such command
also takes `--config`, so each can read the journal a `resume --config` used; a config that does
not load is an error, exit 2, and `--journal` names the journal directly. A journal that does
not exist, and a run id it does not hold, are errors too, exit 2 -- never an empty report, and
never a journal created to report on. `--version` (or `-V`)
prints the version; `specunode` alone prints this list.

Commands that need a config (`resume`, `replay`, `mcp-proxy`) take `--config PATH`. Without the
option the search is `./specunode.yaml`, then `$XDG_CONFIG_HOME/specunode/config.yaml` (or
`~/.config/specunode/config.yaml`), and finding neither means built-in defaults. A `--config`
that names a file which does not exist is an error, exit 2 — never a silent fallback to a
different file.

### `specunode init [DIRECTORY]`

Write `specunode.yaml` from the example packaged with the library, and create `.specunode/`,
under `DIRECTORY` (default: the current directory). An existing config is left alone and said
so.

### `specunode runs`

List the run ids the journal holds, one per line. Options: `--journal`.

### `specunode status RUN_ID`

Say what a run left behind and what a resume would build on: whether it finished, the journal
offset it reached, how many branches retired, which branches were confirmed but never retired,
how many dispatch claims are still in flight, the step index a resume would continue above,
and whether it is resumable at all -- not, and why, for a run `resume` would refuse, such as one
recorded under another rule for where a node's calls sit. Options: `--journal`.

### `specunode ledger RUN_ID`

Print a run's effect ledger: what reached the world, and what authorised it -- and, marked MAY
HAVE BEEN SENT, any effect that may have reached the upstream and that nothing has settled: one
claimed for sending whose reply never came, or one dead-lettered without proof it never left.
Every view shows them. Options: `--journal`; `--short` abbreviates ids; `--normalised` renders
only what the equivalence relation compares, so two runs can be diffed, and below it any effect
that may have been sent -- the relation refuses to compare a run that has one; `--json` emits
the rows (effect id, tool, arguments, idempotency key, branch, authorising step, status, and
`unsettled`: whether it may have been sent with nothing settling it) as JSON instead, with an
`IN_FLIGHT` row for each effect claimed and never answered.

### `specunode verify RUN_ID`

Walk a run's hash chain and report the first break, if any. Exits 0 with the entry count when
the chain verifies, and 1 naming the reason and the offset otherwise. Options: `--journal`.

### `specunode sign-ledger RUN_ID`

Sign a run's ledger with the local key, creating the key on first use. The signature envelope
is written to a file, by default `<journal dir>/ledgers/<run>.sig`, and echoed. Options:
`--journal`; `--keystore DIR` (default `./.specunode/keys`); `--out PATH`.

### `specunode verify-ledger RUN_ID`

Rebuild the ledger, read its signature file, and check both the signature and the journal
chain, saying which failed. An edited ledger and a ledger signed by a key the keystore does not
trust are different problems with different remedies, so they are reported as different
reasons, each with its own exit code: 0 the ledger verifies; 2 the rows are not the bytes the
signature covers; 3 the signing key is not one the keystore trusts; 4 the journal chain is
broken, or the rows do not reproduce from the journal; 5 the signature envelope is malformed;
6 there is no signature. Options: `--journal`; `--keystore DIR`; `--signature PATH` (default:
where `sign-ledger` writes it).

### `specunode resume RUN_ID`

Continue an interrupted run without re-sending what already went out. Committed state is
rebuilt from the branches the journal records as retired, the step counter continues above the
position they consumed, and the graph is driven on from there. An effect that was acked before
the crash is claimed and skipped; one whose request demonstrably never left is re-sent; one
that may or may not have taken effect is asked about through its tool's `reconcile`,
redelivered if its tool declared a repeat harmless, or dead-lettered until someone records what
happened with `specunode resolve`. A node that runs again is served any model answer that may
already have sent something, rather than asked for it again. This needs a live target, because
an answer the journal does not hold, or one that sent nothing, is asked for. A model turn that
failed -- cut off, refused, overloaded -- is recorded as the failure it was and served again as
that failure, so a node that caught it and asked again is matched with its second question. A
run is driven by one process at a time: resuming one that another process is running exits 2, as
does one whose Postgres run lock was lost while it ran, an unknown run, one that never recorded
its start, a LangGraph run, which cannot be resumed in this version, and a run any part of which
was recorded by another version under another rule for where a node's calls sit.
Prints the ledger; exits 1 if the run did not complete and 2 if the config is missing,
unreadable, or cannot build the graph or the target. A turn the crashed run had stopped waiting
for is served as one that never answers, and a node that keeps waiting well past that ends with
`TurnAbandoned` -- every time, until someone decides: `--ask-abandoned` asks the model again
instead, live, at the point the node would be stopped, knowing its answer may differ from what
was acted on. Only that turn: one the node stops waiting for again, as the crashed run did, is
served as it was; and a streamed turn part of which was already handed over is not asked again
part-way -- the node is stopped. A resume that finishes while an effect may have been sent and
nothing settled it -- claimed by an earlier attempt and never answered, or dead-lettered without
proof it never left -- does not report success: it names the tools, for `specunode resolve`.
Options: `--journal`; `--config PATH`; `--ask-abandoned`.

### `specunode resolve RUN_ID KEY`

Record what happened to an effect the runtime could not settle on its own: a write that may
have reached the upstream before its reply was lost, and that a resume therefore will not send
again. Check the upstream first, then say which. `--landed` settles it as sent -- a resume
skips it and hands the node `--ack JSON`, the upstream's own result, if given. `--not-sent`
records that it never took effect -- a resume sends it, once, under the same key. The claim and
a journal entry naming who said so are written in one transaction. `KEY` is the effect's key,
or a unique prefix of it, as `specunode ledger` prints it. Exits 2 if the key matches no
effect or several, if the effect is already settled, or if a process is running the run.
Options: `--landed`; `--not-sent`; `--ack JSON`; `--journal`.

### `specunode replay RUN_ID`

Re-run a journaled run against its own recorded model output, from the inputs the journal
recorded, into a separate journal (`replay-<run>.db` beside the source, so the record being
checked is never written to; a second replay of the same run appends another run to that same
file rather than starting empty). Refuses at the first turn whose request does not match the
journal's, naming the step and the fields that differ, rather than continuing down a trajectory
the recorded run never took (exit 1); and refuses a run any part of which was recorded by another
version under another rule for where a node's calls sit (exit 2). Dispatches
nothing unless told to. Options: `--journal`; `--config PATH`; `--speculation on|off` (default
`on`; anything else exits 2); `--dispatch` actually sends effects, which is off by default
because a replay that re-sent every effect would charge every card again. See
[replay.md](replay.md).

### `specunode mcp-proxy --upstream "<command>"`

Proxy an MCP server over stdio: reads are forwarded immediately, everything else is held until
the model's decision confirms it. Tools are classified from the upstream's own annotations,
with the config's per-tool overrides winning. The tool list is read once at startup: a server
that paginates its `tools/list`, or adds tools later, has those tools neither classified nor
served. Options: `--upstream CMD` (required; the command that starts the upstream server);
`--config PATH`; `--deadline SECONDS` (default 300), how long a blocking write waits for a decision
before giving up, after which it is still held and still unsent and the client is told so;
`--handles`, which returns a staged-write handle instead of blocking, for a client that
understands one and will not put it in a prompt. See [mcp-proxy.md](mcp-proxy.md).
