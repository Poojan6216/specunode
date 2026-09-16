"""Fetch and normalise the offline trace corpus (spec task 6.1).

Source: ``nebius/SWE-rebench-openhands-trajectories`` on Hugging Face (CC-BY-4.0), 67,074
OpenHands trajectories with per-step tool call names and JSON arguments. Pulled through the
datasets-server rows API so no bulk download is needed.

Only the **normalised** form is committed: per trajectory, the ordered sequence of
``(tool, argument keys, whether an argument references a prior result)``. That is what the
opportunity analysis reads, it is a few hundred kilobytes rather than gigabytes, and it makes
the analysis reproducible without a network.

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

from specunode.canonical import canonical, chash_bytes

DATASET = "nebius/SWE-rebench-openhands-trajectories"
ROWS_URL = "https://datasets-server.huggingface.co/rows"
PAGE = 100
HERE = Path(__file__).resolve().parent


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
    trace = Trace(trajectory_id=str(row.get("trajectory_id", "")), repo=str(row.get("repo", "")))
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
    return trace


def fetch(limit: int, seed_offset: int = 0) -> list[Trace]:
    traces: list[Trace] = []
    offset = seed_offset
    while len(traces) < limit:
        want = min(PAGE, limit - len(traces))
        rows = _fetch_page(offset, want)
        if not rows:
            break
        traces.extend(normalise(row) for row in rows)
        offset += want
        print(f"  fetched {len(traces)}/{limit}", file=sys.stderr)
    return [t for t in traces if t.steps]


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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fetch the offline trace corpus")
    parser.add_argument("--limit", type=int, default=300)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--fallback", action="store_true", help="skip the network (D1)")
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
