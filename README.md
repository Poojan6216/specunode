# SpecuNode

Speculative out-of-order execution for agent graphs, with a store buffer.

An agent graph is executed the way an out-of-order CPU executes instructions. Reads issue
early. Writes go into a **store buffer** keyed to the speculative branch that produced them.
The **target model's actual output** is the branch-resolution signal: when it matches the
speculation, the branch **retires** and the store buffer drains to the world with deterministic
idempotency keys; when it does not, the branch is **squashed** and its buffer is discarded,
never dispatched. Every model output and tool result is **journaled** before use, so the run
replays exactly, speculation on or off.

> **Status: under construction.** This README is a placeholder. Every number that appears in
> the finished README will trace to a committed command and a committed results file
> (`bench/results/*.json`), checked by `bench/check_numbers.py` in CI. No figure is written
> here until a run has produced it.

See [`BUILD_SPEC.md`](BUILD_SPEC.md) for the full design, the hard rules, and the build plan.

## License

Apache-2.0. See [LICENSE](LICENSE).
