"""Fetch and normalise the offline trace corpus (spec task 6.1).

Source: ``nebius/SWE-rebench-openhands-trajectories`` on Hugging Face (CC-BY-4.0), 67,074
OpenHands trajectories with per-step tool call names and JSON arguments. Pulled through the
datasets-server rows API so no bulk download is needed.

Only the **normalised** form is committed: per trajectory, the ordered sequence of
``(tool, argument keys, whether an argument references a prior result)``. That is what the
opportunity analysis reads, it is a few hundred kilobytes rather than gigabytes, and it makes
the analysis reproducible without a network.

Argument **values** are dropped from that form on purpose, and they are exactly what the
acceptance measurement needs: the runtime releases a write only on exact canonical equality of
values, so a corpus without them can bound the acceptance rate and cannot measure it.
``--values`` fetches the same rows again and writes the values to a sidecar, ``values.json``,
step for step alongside the committed corpus, with any value longer than ``VALUE_INLINE_BYTES``
replaced by a digest. The committed corpus, and every number derived from it, keeps its hash.

Decision Gate D1: if the dataset is unavailable, ``--fallback`` generates traces from the
sample apps instead, and the manifest records which arm produced the corpus so the README can
say so.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from specunode.canonical import CanonicalError, canonical, chash_bytes

DATASET = "nebius/SWE-rebench-openhands-trajectories"
ROWS_URL = "https://datasets-server.huggingface.co/rows"
PAGE = 100
HERE = Path(__file__).resolve().parent

#: Argument values whose canonical form is longer than this are stored as a digest. The tier-1
#: drafter copies a value verbatim from history or does not offer the call at all, so equality
#: is the only property of a value the acceptance measurement uses -- and a digest preserves
#: equality exactly while keeping 19,000 steps of shell commands and file contents at a size
#: that can be committed.
VALUE_INLINE_BYTES = 64


def reduce_value(value: Any) -> Any:
    """A value as the sidecar stores it: itself when short, ``{"$hash": ...}`` when long."""
    try:
        body = canonical(value)
    except CanonicalError:
        # A value with no canonical form (a NaN, say) could not be compared by the gate
        # either. Its digest stands in for it, so the step is kept and the value still equals
        # itself and nothing else.
        body = json.dumps(value, sort_keys=True, default=str).encode("utf-8")
    if len(body) <= VALUE_INLINE_BYTES:
        return value
    return {"$hash": chash_bytes(body)}


@dataclass
class Step:
    """One tool call, reduced to what the analysis needs."""

    tool: str
    arg_keys: list[str] = field(default_factory=list)
    #: Whether any argument value appeared in an earlier step's *result* text. PASTE's
    #: data-flow observation, detected structurally rather than by a model.
    refs_prior_output: bool = False
    #: Which model turn emitted this call. Two calls from one turn can both be issued while
    #: that turn is still streaming; a call in a later turn had to wait for the model to see
    #: the previous results. The distinction decides how much speculating past a write can
    #: possibly buy, so it is recorded rather than inferred later.
    turn: int = 0
    #: Position within its turn. Zero means this call opened a new turn.
    ordinal: int = 0


@dataclass
class Trace:
    trajectory_id: str
    repo: str
    steps: list[Step] = field(default_factory=list)


def _fetch_page(offset: int, length: int) -> list[dict[str, Any]]:
    query = urllib.parse.urlencode(
        {
            "dataset": DATASET,
            "config": "default",
            "split": "train",
            "offset": offset,
            "length": length,
        }
    )
    with urllib.request.urlopen(f"{ROWS_URL}?{query}", timeout=60) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return [row["row"] for row in payload.get("rows", [])]


def normalise(row: dict[str, Any]) -> Trace:
    """Reduce one trajectory to its tool-call sequence."""
    return normalise_with_values(row)[0]


def normalise_with_values(row: dict[str, Any]) -> tuple[Trace, list[dict[str, Any]]]:
    """The committed shape, plus the argument values it drops, step for step."""
    trace = Trace(trajectory_id=str(row.get("trajectory_id", "")), repo=str(row.get("repo", "")))
    arg_values: list[dict[str, Any]] = []
    seen_results: list[str] = []
    turn = -1
    for message in row.get("trajectory") or []:
        content = message.get("content") or ""
        calls = message.get("tool_calls") or []
        if not calls:
            if content:
                seen_results.append(str(content)[:4000])
            continue
        turn += 1
        for ordinal, call in enumerate(calls):
            function = call.get("function") or {}
            name = str(function.get("name", ""))
            try:
                args = json.loads(function.get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {}
            if not isinstance(args, dict):
                args = {}
            values = [str(v) for v in args.values() if isinstance(v, (str, int, float))]
            refs = any(
                value and len(value) > 8 and any(value in prior for prior in seen_results[-3:])
                for value in values
            )
            trace.steps.append(
                Step(
                    tool=name,
                    arg_keys=sorted(args),
                    refs_prior_output=refs,
                    turn=turn,
                    ordinal=ordinal,
                )
            )
            arg_values.append({key: reduce_value(args[key]) for key in sorted(args)})
    return trace, arg_values


def fetch(limit: int, seed_offset: int = 0) -> list[Trace]:
    return fetch_with_values(limit, seed_offset)[0]


def fetch_with_values(
    limit: int, seed_offset: int = 0
) -> tuple[list[Trace], list[list[dict[str, Any]]]]:
    """The same rows ``fetch`` reads, with each trajectory's argument values alongside."""
    traces: list[Trace] = []
    values: list[list[dict[str, Any]]] = []
    fetched = 0
    offset = seed_offset
    while fetched < limit:
        want = min(PAGE, limit - fetched)
        rows = _fetch_page(offset, want)
        if not rows:
            break
        for row in rows:
            trace, step_values = normalise_with_values(row)
            if trace.steps:
                traces.append(trace)
                values.append(step_values)
        fetched += len(rows)
        offset += want
        print(f"  fetched {fetched}/{limit}", file=sys.stderr)
    return traces, values


def fallback_traces() -> list[Trace]:
    """Decision Gate D1: traces from the sample apps, when the dataset is unavailable."""
    shape = [
        Step(tool="lookup_customer", arg_keys=["customer_id"], turn=0, ordinal=0),
        Step(tool="charge_card", arg_keys=["amount", "customer_id"], turn=1, ordinal=0),
        Step(
            tool="send_receipt",
            arg_keys=["charge_id", "customer_id"],
            refs_prior_output=True,
            turn=1,
            ordinal=1,
        ),
    ]
    return [
        Trace(trajectory_id=f"self-{i}", repo="examples/support_agent", steps=list(shape))
        for i in range(60)
    ]


def write_corpus(traces: list[Trace], source: str, out: Path) -> dict[str, Any]:
    payload = {
        "source": source,
        "dataset": DATASET if source == "huggingface" else "examples/*",
        "traces": [asdict(t) for t in traces],
    }
    body = canonical(payload)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(body)
    manifest = {
        "source": source,
        "dataset": payload["dataset"],
        "trajectories": len(traces),
        "steps": sum(len(t.steps) for t in traces),
        "tools": sorted({s.tool for t in traces for s in t.steps}),
        "corpus_hash": chash_bytes(body),
    }
    (out.parent / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return manifest


def write_values(
    traces: list[Trace], values: list[list[dict[str, Any]]], source: str, out: Path
) -> dict[str, Any]:
    """Write the values sidecar as canonical bytes, and say whether it lines up with the corpus.

    The manifest carries the hash of the corpus these values were cut from, computed the way
    ``write_corpus`` computes it. It equals the committed manifest's hash exactly when the
    dataset served the same rows in the same order -- and the acceptance measurement checks the
    alignment again, per trajectory and per step, when it loads.
    """
    dataset = DATASET if source == "huggingface" else "examples/*"
    payload = {
        "source": source,
        "dataset": dataset,
        "inline_bytes": VALUE_INLINE_BYTES,
        "trajectories": [
            {"trajectory_id": trace.trajectory_id, "steps": step_values}
            for trace, step_values in zip(traces, values, strict=True)
        ],
    }
    body = canonical(payload)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(body)

    corpus_hash = chash_bytes(
        canonical({"source": source, "dataset": dataset, "traces": [asdict(t) for t in traces]})
    )
    manifest_path = out.parent / "manifest.json"
    committed = (
        json.loads(manifest_path.read_text(encoding="utf-8")).get("corpus_hash")
        if manifest_path.is_file()
        else None
    )
    digested = sum(
        1
        for steps in values
        for step in steps
        for value in step.values()
        if isinstance(value, dict) and set(value) == {"$hash"}
    )
    manifest = {
        "source": source,
        "dataset": dataset,
        "trajectories": len(traces),
        "steps": sum(len(steps) for steps in values),
        "values": sum(len(step) for steps in values for step in steps),
        "values_digested": digested,
        "inline_bytes": VALUE_INLINE_BYTES,
        "values_hash": chash_bytes(body),
        "corpus_hash": corpus_hash,
        "matches_committed_corpus": committed is not None and committed == corpus_hash,
    }
    (out.parent / f"{out.stem}_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fetch the offline trace corpus")
    parser.add_argument("--limit", type=int, default=300)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--fallback", action="store_true", help="skip the network (D1)")
    parser.add_argument(
        "--values",
        action="store_true",
        help="fetch the same rows and write values.json beside the corpus, which is not rewritten",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help=(
            "with --values, keep every value whole instead of digesting the long ones. The "
            "result is too large to commit and is gitignored; it is what a measurement needs "
            "when the *content* of a value matters and not only its identity."
        ),
    )
    parser.add_argument("--verify-manifest", action="store_true")
    parser.add_argument("--out", type=Path, default=HERE / "traces.json")
    args = parser.parse_args(argv)

    if args.verify_manifest:
        manifest_path = args.out.parent / "manifest.json"
        if not manifest_path.is_file() or not args.out.is_file():
            print("no committed corpus; run bench/corpus/fetch.py first", file=sys.stderr)
            return 1
        manifest = json.loads(manifest_path.read_text())
        actual = chash_bytes(args.out.read_bytes())
        if actual != manifest["corpus_hash"]:
            print(f"corpus hash {actual} != manifest {manifest['corpus_hash']}", file=sys.stderr)
            return 1
        print(
            f"manifest verified: {manifest['trajectories']} trajectories, {manifest['steps']} steps"
        )
        values_manifest = args.out.parent / "values_manifest.json"
        values_path = args.out.parent / "values.json"
        if values_manifest.is_file():
            expected = json.loads(values_manifest.read_text(encoding="utf-8"))
            if not values_path.is_file():
                print("values_manifest.json is committed without values.json", file=sys.stderr)
                return 1
            got = chash_bytes(values_path.read_bytes())
            if got != expected["values_hash"]:
                print(f"values hash {got} != manifest {expected['values_hash']}", file=sys.stderr)
                return 1
            print(
                f"values manifest verified: {expected['trajectories']} trajectories, "
                f"{expected['values']} values ({expected['values_digested']} digested)"
            )
        return 0

    if args.values:
        if args.fallback:
            print("--values needs the dataset; the fallback corpus has no values", file=sys.stderr)
            return 2
        global VALUE_INLINE_BYTES
        if args.full:
            # Nothing is digested: every value is kept whole. The committed sidecar exists to
            # answer "is this value the same one?", which a digest answers exactly; a draft
            # model asked to *predict* a value needs to have seen real ones, and told us so --
            # it replied "the hashes in the previous calls obscure" what it needed.
            VALUE_INLINE_BYTES = 1 << 30
        traces, values = fetch_with_values(args.limit, args.offset)
        out = args.out.parent / ("values_full.json" if args.full else "values.json")
        manifest = write_values(traces, values, "huggingface", out)
        print(json.dumps(manifest, indent=2, sort_keys=True))
        if not manifest["matches_committed_corpus"]:
            print("the rows served today are not the committed corpus", file=sys.stderr)
            return 1
        return 0

    if args.fallback:
        traces, source = fallback_traces(), "self-generated"
    else:
        try:
            traces, source = fetch(args.limit, args.offset), "huggingface"
        except Exception as exc:  # Decision Gate D1
            print(f"dataset unavailable ({exc}); falling back to self-generated", file=sys.stderr)
            traces, source = fallback_traces(), "self-generated"

    manifest = write_corpus(traces, source, args.out)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
