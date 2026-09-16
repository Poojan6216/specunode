"""The equivalence relation: what the speculative run did to the world, compared (Hard Rule 9).

Rule 9 says the effect ledger of a run with speculation on equals the ledger of the same
journaled run with speculation off, "ignoring timestamps and keys' branch components". This
module is the only place that decides what "ignoring" covers, and that decision is the whole
risk. Strip too much and the relation proves nothing -- a ledger of empty rows always matches
another ledger of empty rows, the mandatory test goes green, and nothing is being checked.
Strip too little and the relation fails on bookkeeping that exists *only* because speculation
happened -- branch ids, fresh ULIDs, journal offsets -- at which point the honest test is
unsatisfiable by construction and the cheapest way out is to delete it, which the spec forbids.

The principle behind every row of the table below: **keep what the effect is to the world and
the authority it claims; strip what is a property of how the runtime got there, or what the
world itself generated.**

``LedgerRow``
------------

=========================  ==========  ===========================================================
Field                      Action      Why
=========================  ==========  ===========================================================
``call.name``              KEEP        what the world was told to do
``call.args``              KEEP        as canonical bytes: ``{"amount": 1}`` is not
                                       ``{"amount": 1.0}`` to a billing API, and Hard Rule 4
                                       already refuses to treat them as one decision
``key`` (Rule 8)           REWRITE     to ``ekey`` -- :func:`~specunode.buffer.idempotency.
                                       equivalence_key`, the same preimage with ``run_id=""``
                                       and an empty lineage. Rule 9 names lineage as
                                       non-identifying; the two arms are different runs, so the
                                       run id goes with it. ``node_id`` and ``step_index``
                                       survive inside the digest, so a loop iteration attributed
                                       to the wrong node, or a step counter that drifts when
                                       branches fork, still fails the relation.
``nkey``                   STRIP       a prefix of what ``ekey`` re-derives; it carries the run
                                       id, so comparing it fails on the run id alone. Used by
                                       :func:`ledger_matches_world` instead, where both sides
                                       come from the same run.
``node_id``, ``step_index``  STRIP as  they are inside the rewritten key already; a field that
                           fields     appears twice adds no evidence and two places to skew
``authorised_by_step``     KEEP        as ``auth``. The step index of the model turn that
                                       confirmed the effect -- the only field that catches an
                                       effect staged for decision *i* dispatched under the
                                       confirmation of decision *i+1*, where call, key, status
                                       and order are all identical. It must be a **step index**
                                       and never a journal offset: a speculative arm journals
                                       ``branch_forked``, ``effect_staged`` and
                                       ``effect_discarded`` entries the sequential arm never
                                       writes, so offsets between the arms differ by the volume
                                       of speculation, while step indices do not.
``status``                 KEEP        a dead letter is a real difference in what reached the
                                       world. This is why fault injection belongs to task 5.2
                                       and not here: an arm-specific ``DEAD_LETTER`` would make
                                       the mandatory test flake, and the tempting repair is to
                                       drop ``status`` -- after which a run that dead-letters
                                       every write compares equal to one that dispatches them.
``effect_id``              STRIP       a fresh ULID. It differs between two *sequential* runs of
                                       the same journal, so keeping it makes the relation
                                       unsatisfiable. A row's identity here is its position.
``branch_id``,             STRIP       Rule 9 says so in as many words
``branch_ord``, lineage
``retire_seq``,            SORT KEY,   order is compared; the ordinals themselves are
``dispatch_index``         not a field bookkeeping. See below on why it is dispatch order.
``stage_index``            STRIP       stage-time metadata, and actively dangerous as a sort key
``reason``                 STRIP       dead-letter reasons embed socket errors, hosts, retries
``ack``, ``ack_hash``      STRIP       the world's payload: server ids and timestamps, which
                                       Rule 9 exempts. Recovered by :func:`ledger_matches_world`
``compensates``            STRIP       the compensation is its own row; the effect on the world
                                       is the row, not the cross-reference
=========================  ==========  ===========================================================

``Ledger``
----------

``rows`` are kept, ordered, and their count is kept as an explicit ``n`` so that two runs that
each dispatched nothing cannot pass as "equivalent". Everything else is stripped, and the
stripped aggregates split into two groups:

* ``run_id``, ``signature``, ``journal_entries``, ``journal_head``, ``speculation``,
  ``drafters``, ``target_model``, ``alpha``, ``alpha_window``, ``context_identity`` --
  identity and provenance of one execution, not an effect.
* ``squashed_branches``, ``discarded_effects``, ``stalls``, ``wasted_tokens``,
  ``speculative_reads_upstream``, ``context_divergences``, ``reads_validated`` -- the *cost*
  of speculation. Every one of them is zero in the sequential arm by construction, so
  including any of them would make Rule 9 impossible to satisfy rather than harder to fake.

Excluding the cost counters does not weaken the rule, because each one's safety-relevant
consequence is already visible in ``rows``: a squashed branch that leaked shows up as an extra
row, a discarded effect that dispatched shows up as an extra row, a stale witnessed read that
should have squashed a branch under E3 *removes* rows, and a Rule 13 divergence squashes before
the model output is used, so it can only remove rows. The cost counters answer "what did
speculation cost", which differs between the arms by design; the relation answers "what reached
the world", which must not.

The one exclusion that is a genuine scope limit is ``speculative_reads_upstream``. Rule 9 is
about the **effect** ledger, and reads issued by branches that were later squashed do reach
upstream. That is a documented limitation, not an equivalence violation -- honest only because
the number is printed in every rendered ledger and listed in the README's limitations, never
because it is quietly absent here.

Order, and why it is dispatch order
-----------------------------------

Rows are ordered by ``(retire_seq, dispatch_index)`` and never by ack arrival, wall time, key,
or branch. Retirement is in program order, which makes ``retire_seq`` total; the sequential arm
dispatches in program order by construction, so this is the only ordering under which the two
arms can agree at all. ``dispatch_index`` records the order effects actually *left*, which is
not the same fact as ``stage_index``: sorting by ``stage_index`` would take a drain that
iterated its buffer backwards and put the rows back into stage order, so call, key, authority,
status and count would all match and task 5.1's mandated "dispatch order reversed" planted bug
would pass. Sorting here is therefore safe and deliberate -- the sort follows a reversal rather
than undoing it -- and it also means a ledger builder that emits rows in journal order is
compared on the same footing as one that does not.

Companions, asserted alongside and never instead
------------------------------------------------

:func:`normalise_world_mutations` and :func:`ledger_matches_world` exist so that "the ledger
says what the world got" is checked rather than assumed. Both are order-bearing: a join on a
key alone is order-insensitive and would let the reversed drain through a second time.

:func:`normalise_cross_adapter` is a **weaker** relation for one narrowly scoped comparison --
a run driven through the MCP proxy, which has no node concept, against the same run driven
through LangGraph. It empties ``node_id`` in the key preimage, so a node-attribution error
cannot be detected under it. Rule 9's own test never uses it, and the bytes carry a different
``rel`` tag so that a value produced by the weak relation can never be mistaken for one
produced by the strong one.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Final, Protocol

from specunode.buffer.idempotency import equivalence_key
from specunode.canonical import JsonValue, canonical, chash_bytes
from specunode.core.decision import ToolCall, decision_payload

__all__ = [
    "EquivalenceError",
    "LedgerLike",
    "LedgerRowLike",
    "MutationLike",
    "assert_equivalent",
    "equivalence_digest",
    "ledger_matches_world",
    "normalise_cross_adapter",
    "normalise_for_equivalence",
    "normalise_world_mutations",
    "normalised_rows",
]

#: Bumped when the normalised shape changes, so two digests taken under different definitions
#: can never silently compare equal.
RELATION_VERSION: Final = 1

#: Statuses that mean the effect left the runtime. ``DEAD_LETTER`` did not; a ``COMPENSATED``
#: row did (both the original, which reached the world and was later undone, and the
#: compensator, which is its own row).
REACHED_WORLD: Final = frozenset({"DISPATCHED", "COMPENSATED"})


class LedgerRowLike(Protocol):
    """The part of ``journal.ledger.LedgerRow`` this relation reads.

    A structural protocol rather than an import, so the relation is defined by the fields it
    compares rather than by one class, and so a ledger rebuilt by a different code path (the
    MCP proxy's, a replay's) is comparable without inheriting anything. Every member is a
    read-only property: a frozen dataclass satisfies it, and this module cannot write to it.
    """

    @property
    def call(self) -> ToolCall: ...
    @property
    def node_id(self) -> str: ...
    @property
    def step_index(self) -> int: ...
    @property
    def authorised_by_step(self) -> int: ...
    @property
    def status(self) -> str: ...
    @property
    def nkey(self) -> str: ...
    @property
    def retire_seq(self) -> int: ...
    @property
    def dispatch_index(self) -> int: ...


class LedgerLike(Protocol):
    """The part of ``journal.ledger.Ledger`` this relation reads.

    ``terminal`` is a precondition rather than a compared field: a run whose ``run_finished``
    entry is absent has a ledger that is a prefix of the one it would have had, and a prefix
    that happens to match is not evidence of anything.
    """

    @property
    def rows(self) -> Sequence[LedgerRowLike]: ...
    @property
    def terminal(self) -> bool: ...
    @property
    def discarded_effects(self) -> int: ...


class MutationLike(Protocol):
    """The part of :class:`specunode.testing.world.Mutation` the companions read.

    ``effect_key`` is the ``nkey`` -- the token the dispatcher hands the adapter as its
    idempotency key, and the only one of the three derivations the world ever sees. The
    lineage-bearing ``key`` and the comparison-only ``ekey`` never leave the runtime, which is
    why the join below is on ``nkey`` and the cross-arm comparison is not.
    """

    @property
    def sequence(self) -> int: ...
    @property
    def effect_key(self) -> str: ...
    @property
    def tool(self) -> str: ...
    @property
    def args_hash(self) -> str: ...


class EquivalenceError(AssertionError):
    """Hard Rule 9 did not hold, or the comparison that would have tested it was not meaningful.

    An :class:`AssertionError` subclass because the mandatory tests call
    :func:`assert_equivalent` in place of a bare ``assert`` and a Rule 9 failure is a build
    failure, not an exceptional condition to be caught and retried.
    """


def _ordered(ledger: LedgerLike) -> list[LedgerRowLike]:
    return sorted(ledger.rows, key=lambda row: (row.retire_seq, row.dispatch_index))


def _row_payload(row: LedgerRowLike, *, node_id: str) -> dict[str, JsonValue]:
    return {
        "call": decision_payload(row.call),
        "key": str(
            equivalence_key(
                node_id=node_id,
                step_index=row.step_index,
                tool_name=row.call.name,
                args=row.call.args,
            )
        ),
        "auth": row.authorised_by_step,
        "status": row.status,
    }


def _normalise(ledger: LedgerLike, *, relation: str, node_insensitive: bool) -> bytes:
    rows = _ordered(ledger)
    payload: JsonValue = {
        "rel": relation,
        "v": RELATION_VERSION,
        # The count is in the bytes, not merely implied by the array, so that a comparison of
        # two runs that dispatched nothing reads as "n: 0" in any dump of the normalised form
        # rather than as an empty structure nobody looks twice at.
        "n": len(rows),
        "rows": [
            _row_payload(row, node_id="" if node_insensitive else row.node_id) for row in rows
        ],
    }
    return canonical(payload)


def normalised_rows(ledger: LedgerLike) -> tuple[Mapping[str, JsonValue], ...]:
    """The compared rows, in compared order, as JSON values.

    Public so that a ``--normalised`` ledger rendering shows exactly what the relation compares
    instead of a second, hand-maintained table that can drift away from it.
    """
    return tuple(_row_payload(row, node_id=row.node_id) for row in _ordered(ledger))


def normalise_for_equivalence(ledger: LedgerLike) -> bytes:
    """The Rule 9 relation: two runs are equivalent iff these bytes match.

    Rows in dispatch order, branch ids, effect ids, timestamps and acks stripped, keys
    re-derived with an empty lineage and no run id. See the module docstring for the field
    table and for what each decision would cost if it went the other way.
    """
    return _normalise(ledger, relation="strict", node_insensitive=False)


def normalise_cross_adapter(ledger: LedgerLike) -> bytes:
    """A deliberately weaker relation, for comparing two adapters (Phase Gate 4 only).

    Identical to :func:`normalise_for_equivalence` except that ``node_id`` is emptied in the
    key preimage, because the MCP proxy has no node concept and cannot report one unless the
    client supplies it. That makes a node-attribution error invisible, so **Hard Rule 9's own
    test must not use this function**: a run compared only under this relation has not been
    shown to satisfy Rule 9. The differing ``rel`` tag keeps the two outputs from ever
    comparing equal by accident.
    """
    return _normalise(ledger, relation="cross_adapter", node_insensitive=True)


def equivalence_digest(ledger: LedgerLike) -> str:
    """The 16 hex characters a rendered ledger prints as its ``equivalence digest``.

    A short digest is for a human eyeballing two renderings side by side. Machine comparison
    uses the full bytes of :func:`normalise_for_equivalence`, never this.
    """
    return chash_bytes(normalise_for_equivalence(ledger))[:16]


def normalise_world_mutations(mutations: Sequence[MutationLike]) -> bytes:
    """What the *world* recorded, normalised for comparison between the two arms.

    Keeps the tool, the canonical argument hash and the order the mutations arrived in.
    Strips the branch id (Rule 9 says so), the wall time (Rule 9 exempts timestamps), the
    table and row id (world-assigned), and ``effect_key`` -- which is an ``nkey``, and so
    carries a run id that differs between the arms by construction. Order is carried by array
    position, which is the fact being compared; it is not restated as a field.
    """
    ordered = sorted(mutations, key=lambda m: m.sequence)
    payload: JsonValue = {
        "rel": "world",
        "v": RELATION_VERSION,
        "n": len(ordered),
        "mutations": [{"tool": m.tool, "args_hash": m.args_hash} for m in ordered],
    }
    return canonical(payload)


def _world_join_problem(ledger: LedgerLike, mutations: Sequence[MutationLike]) -> str | None:
    """The reason the ledger and the world disagree, or ``None`` when they do not."""
    rows = [row for row in _ordered(ledger) if row.status in REACHED_WORLD]
    ordered = sorted(mutations, key=lambda m: m.sequence)

    row_keys = [row.nkey for row in rows]
    unaccounted = [m for m in ordered if m.effect_key not in set(row_keys)]
    if unaccounted:
        first = unaccounted[0]
        return (
            f"the world recorded {len(unaccounted)} mutation(s) the ledger does not account "
            f"for; the first is {first.tool!r} under key {first.effect_key[:8]}… -- an effect "
            "reached the world without a row claiming authority for it"
        )

    world_keys = [m.effect_key for m in ordered]
    missing = [row for row in rows if row.nkey not in set(world_keys)]
    if missing:
        first_row = missing[0]
        return (
            f"{len(missing)} row(s) claim to have reached the world but no mutation carries "
            f"their key; the first is {first_row.call.name!r} under key {first_row.nkey[:8]}…"
        )

    if len(rows) != len(ordered):
        return (
            f"{len(rows)} row(s) reached the world but the world recorded {len(ordered)} "
            "mutation(s); a key was delivered more than once"
        )

    for index, (row, mutation) in enumerate(zip(rows, ordered, strict=True)):
        if row.nkey != mutation.effect_key:
            return (
                f"order differs at position {index}: the ledger has {row.call.name!r} "
                f"({row.nkey[:8]}…) where the world has {mutation.tool!r} "
                f"({mutation.effect_key[:8]}…)"
            )
    return None


def ledger_matches_world(ledger: LedgerLike, mutations: Sequence[MutationLike]) -> bool:
    """Does the ledger account for the world, in order, one mutation per row that reached it?

    Joins on ``nkey``, the one key the world sees, and then compares *sequence*: one mutation
    per ``DISPATCHED`` or ``COMPENSATED`` row, no mutation without a row, and the world's
    arrival order equal to the ledger's row order. The order clause is what makes a reversed
    drain detectable twice -- a key join on its own is order-insensitive, and an equivalence
    suite that relied on it alone would report that the ledger and the world agree about a
    drain that ran backwards.

    Both arguments must come from the **same** run: ``nkey`` is run-scoped, so this is a
    within-run corroboration and never a cross-arm comparison.
    """
    return _world_join_problem(ledger, mutations) is None


def _first_row_difference(left: LedgerLike, right: LedgerLike) -> str:
    left_rows = normalised_rows(left)
    right_rows = normalised_rows(right)
    for index, (a, b) in enumerate(zip(left_rows, right_rows, strict=False)):
        if canonical(a) != canonical(b):
            return f"first difference at row {index}:\n  off: {a}\n   on: {b}"
    if len(left_rows) != len(right_rows):
        longer, label = (
            (right_rows, "on") if len(right_rows) > len(left_rows) else (left_rows, "off")
        )
        extra = longer[min(len(left_rows), len(right_rows))]
        return (
            f"the arms agree on the first {min(len(left_rows), len(right_rows))} row(s); "
            f"the {label} arm has {abs(len(left_rows) - len(right_rows))} extra row(s), "
            f"the first being {extra}"
        )
    return "the rows are equal pairwise; the difference is in the header (row count or version)"


def assert_equivalent(
    ledger_off: LedgerLike,
    ledger_on: LedgerLike,
    mutations_off: Sequence[MutationLike],
    mutations_on: Sequence[MutationLike],
    *,
    expect_rows: int,
    min_squashed_with_staged: int = 0,
) -> None:
    """Assert Hard Rule 9 for one workload, with the anchors that stop it passing vacuously.

    ``expect_rows`` is the number of effects the workload is known to produce, committed
    alongside it. Without it, a runtime that dispatched nothing compares two empty ledgers and
    the mandatory test reports success for a run that did not happen. Too few rows means the
    workload did not run; too many means something extra reached the world, which is the leak
    shape -- both fail.

    ``min_squashed_with_staged`` is the exercise anchor for the speculative arm: it is the
    number of staged effects that must have been discarded, which is above zero only when a
    branch that had already staged something failed to retire. Zero discarded effects means
    the store buffer was never asked to hold anything back, so the "retire on SQUASHED" bug
    the relation exists to catch was unreachable in that run and the arms were never really
    compared. It defaults to ``0`` for the tier-0 arm, where the drafter only ever replays
    decisions the target has already emitted, so nothing mispredicts by construction; that
    exemption is stated at each call site rather than assumed here.

    Fault injection belongs to task 5.2, not here: an arm-specific ``DEAD_LETTER`` is a real
    difference in what reached the world, and injecting faults into this comparison would make
    a mandatory test flake for a reason that is not a bug.

    Raises :class:`EquivalenceError` on any failure; returns ``None`` otherwise.
    """
    for label, ledger in (("off", ledger_off), ("on", ledger_on)):
        if not ledger.terminal:
            raise EquivalenceError(
                f"the speculation-{label} ledger is not terminal (no run_finished entry); "
                "an unfinished run's ledger is a prefix, and a prefix that matches proves "
                "nothing"
            )

    for label, ledger in (("off", ledger_off), ("on", ledger_on)):
        count = len(ledger.rows)
        if count != expect_rows:
            direction = "only " if count < expect_rows else ""
            raise EquivalenceError(
                f"the speculation-{label} arm produced {direction}{count} ledger row(s) where "
                f"the workload declares {expect_rows}; the arms were not compared"
            )

    for label, ledger, mutations in (
        ("off", ledger_off, mutations_off),
        ("on", ledger_on, mutations_on),
    ):
        problem = _world_join_problem(ledger, mutations)
        if problem is not None:
            raise EquivalenceError(
                f"the speculation-{label} ledger does not match its world: {problem}"
            )

    if normalise_for_equivalence(ledger_off) != normalise_for_equivalence(ledger_on):
        raise EquivalenceError(
            "Hard Rule 9: the speculative ledger differs from the sequential ledger.\n"
            + _first_row_difference(ledger_off, ledger_on)
        )

    if normalise_world_mutations(mutations_off) != normalise_world_mutations(mutations_on):
        raise EquivalenceError(
            "the two ledgers agree but the two worlds do not: the arms' mutation sequences "
            f"differ (off: {len(mutations_off)} mutation(s), on: {len(mutations_on)})"
        )

    if ledger_on.discarded_effects < min_squashed_with_staged:
        raise EquivalenceError(
            f"the speculative arm discarded {ledger_on.discarded_effects} staged effect(s), "
            f"below the workload's declared minimum of {min_squashed_with_staged}: no branch "
            "carrying staged effects was ever squashed, so this comparison never exercised "
            "the store buffer"
        )
