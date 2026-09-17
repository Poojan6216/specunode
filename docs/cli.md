# Command-line reference

`specunode` answers questions about a run that already happened, or continues one that did not
finish. No subcommand decides anything: the journal decides, and these read it.

Every command that reads a run takes `--journal PATH`, defaulting to `./.specunode/journal.db`.
`--version` (or `-V`) prints the version; `specunode` alone prints this list.

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
and whether it is resumable at all. Options: `--journal`.

### `specunode ledger RUN_ID`

Print a run's effect ledger: what reached the world, and what authorised it. Options:
`--journal`; `--short` abbreviates ids; `--normalised` renders only what the equivalence
relation compares, so two runs can be diffed; `--json` emits the rows (effect id, tool,
arguments, idempotency key, branch, authorising step, status) as JSON instead.

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
reasons; the exit code is the category's. Options: `--journal`; `--keystore DIR`;
`--signature PATH` (default: where `sign-ledger` writes it).

### `specunode resume RUN_ID`

Continue an interrupted run without re-sending what already went out. Committed state is
rebuilt from the branches the journal records as retired, the step counter continues above the
position they consumed, and the graph is driven on from there. An effect that was acked before
the crash is claimed and skipped; one whose request demonstrably never left is re-sent; one
that may or may not have taken effect is dead-lettered unless its tool declared a repeat
harmless. This needs a live target, because the turns the journal does not already hold have
to be asked for. Prints the ledger; exits 1 if the run did not complete and 2 if the config
cannot build the graph or the target. Options: `--journal`; `--config PATH` (default
`./specunode.yaml`).

### `specunode replay RUN_ID`

Re-run a journaled run against its own recorded model output, from the inputs the journal
recorded, into a fresh journal (`replay-<run>.db` beside the source, so the record being checked
is never written to). Refuses at the first turn whose request does not match the journal's,
naming the step and the fields that differ, rather than continuing down a trajectory the
recorded run never took (exit 1). Dispatches nothing unless told to. Options: `--journal`;
`--config PATH`; `--speculation on|off` (default `on`; anything else exits 2); `--dispatch`
actually sends effects, which is off by default because a replay that re-sent every effect
would charge every card again. See [replay.md](replay.md).

### `specunode mcp-proxy --upstream "<command>"`

Proxy an MCP server over stdio: reads are forwarded immediately, everything else is held until
the model's decision confirms it. Tools are classified from the upstream's own annotations,
with the config's per-tool overrides winning. Options: `--upstream CMD` (required; the command
that starts the upstream server); `--config PATH` (default `./specunode.yaml`, used when
present); `--deadline SECONDS` (default 300), how long a blocking write waits for a decision
before giving up, after which it is still held and still unsent and the client is told so;
`--handles`, which returns a staged-write handle instead of blocking, for a client that
understands one and will not put it in a prompt. See [mcp-proxy.md](mcp-proxy.md).
