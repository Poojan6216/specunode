"""Read validation at retirement: did the world move under a speculation?

Lattice rule E3, and its position in the ordering carries most of the integrity. A branch
whose reads went stale between speculating and being confirmed must not retire *just because
the model's decision matched*. The decision being right does not make the data the branch
computed from right.

Two honesty constraints shape this module:

**Only speculative reads are validated.** A read a CONFIRMED branch issued happened after the
model's decision was already durable; there is no speculation to invalidate, and re-checking it
would turn a workload with a competing writer into a livelock on the *sequential* arm, where
validation means nothing.

**A read without a witness is reported unwitnessed, never fresh.** The runtime cannot tell
whether it went stale. Counting it as fresh would inflate the one number attack 7.3 exists to
publish honestly -- the fraction of stale reads that were undetectable.

Re-fetching costs a real upstream call, and those calls are counted in the ledger's speculative
read total rather than left out because they are the runtime's own.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from specunode.canonical import JsonValue, chash
from specunode.core.branch import Branch
from specunode.core.effects import ToolRegistry
from specunode.core.model import CallScope, call_scope

__all__ = ["ReadValidation", "ReadVerdict", "validate_reads"]

Verdict = Literal["fresh", "stale", "unwitnessed", "unreadable"]


@dataclass(frozen=True, slots=True)
class ReadVerdict:
    """What re-checking one read found."""

    tool: str
    args_hash: str
    verdict: Verdict
    witness_before: JsonValue = None
    witness_after: JsonValue = None
    raced_drain: bool = False


@dataclass(frozen=True)
class ReadValidation:
    """The whole read set's verdict, and the honest breakdown of it."""

    verdicts: tuple[ReadVerdict, ...] = ()
    #: Upstream calls this validation itself made. Counted, not hidden.
    probes: int = 0

    @property
    def fresh(self) -> int:
        return sum(1 for v in self.verdicts if v.verdict == "fresh")

    @property
    def stale(self) -> int:
        return sum(1 for v in self.verdicts if v.verdict == "stale")

    @property
    def unwitnessed(self) -> int:
        return sum(1 for v in self.verdicts if v.verdict == "unwitnessed")

    @property
    def unreadable(self) -> int:
        """Reads the re-probe could not complete: it raised, or the tool cannot be re-fetched.

        Counted, because an unchecked read is not a checked one. This had no counter at all, so
        it was absent from the journal payload and from the rendered tally, and ``_retire``
        gated only on ``stale`` -- meaning a probe that raised was treated exactly like a probe
        that came back fresh, and the write dispatched. Any flaky or hostile upstream turned E3
        off silently, and the ledger printed 100% fresh while doing it.
        """
        return sum(1 for v in self.verdicts if v.verdict == "unreadable")

    @property
    def unverified(self) -> int:
        """Everything that was *not* positively confirmed fresh and is not a known unknown.

        Stale and unreadable both mean "this branch's writes are not backed by a value the
        world still agrees with". ``unwitnessed`` is deliberately excluded: it is a third
        verdict the design reports honestly rather than a failure, because a read with no
        witness was never checkable and refusing every one of them would stop most workloads.
        """
        return self.stale + self.unreadable

    @property
    def raced_drain(self) -> int:
        return sum(1 for v in self.verdicts if v.raced_drain)

    @property
    def total(self) -> int:
        return len(self.verdicts)

    @property
    def any_stale(self) -> bool:
        return self.stale > 0

    def payload(self) -> Mapping[str, JsonValue]:
        """The ``read_validated`` journal payload."""
        return {
            "reads": [
                {
                    "tool": v.tool,
                    "args_hash": v.args_hash,
                    "verdict": v.verdict,
                    "witness_before": v.witness_before,
                    "witness_after": v.witness_after,
                    "raced_drain": v.raced_drain,
                }
                for v in self.verdicts
            ],
            "fresh": self.fresh,
            "stale": self.stale,
            "unwitnessed": self.unwitnessed,
            "total": self.total,
            "probes": self.probes,
        }


async def validate_reads(branch: Branch, registry: ToolRegistry) -> ReadValidation:
    """Re-check every witnessed read this branch made while it was still a guess."""
    verdicts: list[ReadVerdict] = []
    probes = 0

    for record in branch.reads_to_validate():
        if not record.witnessed:
            verdicts.append(
                ReadVerdict(
                    tool=record.tool,
                    args_hash=record.args_hash,
                    verdict="unwitnessed",
                    raced_drain=record.raced_drain,
                )
            )
            continue

        spec = registry.get(record.tool)
        args = record.args
        if spec.synthesised:
            # Nothing to re-fetch with, or a tool nobody declared. Reported as unreadable
            # rather than assumed fresh: an unchecked read is not a checked one.
            verdicts.append(
                ReadVerdict(
                    tool=record.tool,
                    args_hash=record.args_hash,
                    verdict="unreadable",
                    witness_before=record.witness,
                    raced_drain=record.raced_drain,
                )
            )
            continue

        token = call_scope.set(
            CallScope(branch_id=branch.id, lineage=branch.lineage, speculative=False)
        )
        try:
            probes += 1
            current = await spec.fn(**args)
        except Exception:
            verdicts.append(
                ReadVerdict(
                    tool=record.tool,
                    args_hash=record.args_hash,
                    verdict="unreadable",
                    witness_before=record.witness,
                    raced_drain=record.raced_drain,
                )
            )
            continue
        finally:
            call_scope.reset(token)

        after = current.get("witness") if isinstance(current, Mapping) else None
        same = chash(after) == chash(record.witness)
        verdicts.append(
            ReadVerdict(
                tool=record.tool,
                args_hash=record.args_hash,
                verdict="fresh" if same else "stale",
                witness_before=record.witness,
                witness_after=after,
                raced_drain=record.raced_drain,
            )
        )

    return ReadValidation(verdicts=tuple(verdicts), probes=probes)
