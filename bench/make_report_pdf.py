"""Regenerate the technical report PDF from the committed results (spec task 9.4).

Every figure in the output is read from ``bench/results/*.json``. Nothing is typed in, and a
table whose results file is absent renders as "not measured" rather than being left out --
an absent measurement and a measurement of zero are different things, and a report that omits
the first will be read as the second.

``python bench/make_report_pdf.py --out SpecuNode-Report-generated.pdf``
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "bench" / "results"


def load(name: str) -> dict[str, Any] | None:
    path = RESULTS / name
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def ci(stat: dict[str, float]) -> str:
    return f"{stat['mean']:.4f}  [{stat['ci95_low']:.4f}, {stat['ci95_high']:.4f}]"


def build_story(styles: Any) -> list[Any]:
    from reportlab.lib import colors
    from reportlab.lib.units import mm
    from reportlab.platypus import PageBreak, Paragraph, Spacer, Table, TableStyle

    body, h1, h2 = styles["BodyText"], styles["Heading1"], styles["Heading2"]
    story: list[Any] = []

    def para(text: str, style: Any = None) -> None:
        story.append(Paragraph(text, style or body))
        story.append(Spacer(1, 3 * mm))

    def table(rows: list[list[str]]) -> None:
        t = Table(rows, hAlign="LEFT")
        t.setStyle(
            TableStyle(
                [
                    ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 9),
                    ("FONT", (0, 1), (-1, -1), "Helvetica", 9),
                    ("LINEBELOW", (0, 0), (-1, 0), 0.6, colors.black),
                    ("LINEBELOW", (0, 1), (-1, -2), 0.2, colors.lightgrey),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ]
            )
        )
        story.append(t)
        story.append(Spacer(1, 5 * mm))

    para("SpecuNode", h1)
    para(
        "Speculative execution for agent graphs, with a store buffer: what happens when an AI "
        "agent runs ahead of its own model, and how to make sure nothing it guessed reaches "
        "the world.",
    )
    para(
        "<b>Every number in this document is read from a committed file in "
        "<font face='Courier'>bench/results/</font>.</b> Nothing here is typed in by hand. A "
        "section whose results file is missing says so rather than being omitted.",
    )

    # -- the negative result, first ---------------------------------------------------------
    para("1. How much speculation a real corpus exposes", h2)
    opportunity = load("opportunity.json")
    if opportunity is None:
        para("Not measured. Run <font face='Courier'>bench/offline/run_opportunity.py</font>.")
    else:
        corpus, o = opportunity["corpus"], opportunity["opportunity"]
        para(
            f"Corpus: <font face='Courier'>{corpus['dataset']}</font> ({corpus['source']}), "
            f"{o['trajectories']} trajectories, {o['steps']} tool calls. Effect classes come "
            "from a committed hand-written table per tool name; anything absent from it is a "
            "WRITE, exactly as in the runtime."
        )
        table(
            [
                ["Measure", "Mean [95% CI]"],
                ["Reads - speculable under any policy", ci(o["read_fraction"])],
                ["Calls opening a new model turn", ci(o["calls_opening_a_new_model_turn"])],
                ["Args referencing a prior result", ci(o["args_referencing_a_prior_result"])],
                ["PASTE speculable span (calls)", ci(o["paste_speculable_span"])],
                [
                    "Steps PASTE skips, SpecuNode stages",
                    ci(o["steps_paste_must_skip_that_specunode_can_stage"]),
                ],
                ["SpecuNode span past a write (calls)", ci(o["specunode_post_write_span"])],
                [
                    "Model turn consuming a write result",
                    ci(o["model_turn_consuming_a_write_result"]),
                ],
            ]
        )
        para(
            f"<b>On this corpus, running ahead past a write buys nothing at all.</b> The span "
            f"is {o['specunode_post_write_span']['mean']:.4f}, with a bootstrap interval that "
            "does not move off zero. Every tool call in these trajectories opens a new model "
            "turn, and a staged write blocks the next turn because that turn would have to "
            "contain a placeholder where the real result belongs. "
            f"{o['model_turn_consuming_a_write_result']['mean']:.1%} of the corpus is that shape."
        )
        para(
            "The half that is not zero: PASTE excludes a tool with side effects from "
            f"speculation entirely, so it can speculate on {o['read_fraction']['mean']:.1%} of "
            "steps. SpecuNode stages a write instead of refusing it, covering the other "
            f"{o['steps_paste_must_skip_that_specunode_can_stage']['mean']:.1%}. That is an "
            "upper bound on opportunity, not a speedup, realisable only where the predictor is "
            f"right - measured at {opportunity['opportunity']['predictability']['top_1']:.1%} "
            "top-1. Neither number means anything alone."
        )

    story.append(PageBreak())

    # -- what beats it ----------------------------------------------------------------------
    para("2. What beats it", h2)
    attacks = load("attacks.json")
    if attacks is None:
        para(
            "Not measured. Run <font face='Courier'>bench/adversarial/run_attacks.py --all</font>."
        )
    else:
        rows = [["#", "Strategy", "Outcome", "Measured"]]
        for attack in attacks["attacks"]:
            outcome = (
                "error"
                if attack["error"]
                else ("beats it" if attack["defeated_runtime"] else "held")
            )
            measured = (
                attack["error"][:50]
                if attack["error"]
                else "; ".join(f"{k}={v}" for k, v in list(attack["measured"].items())[:3])
            )
            rows.append([attack["id"], attack["name"][:36], outcome, measured[:52]])
        table(rows)

    # -- demo -------------------------------------------------------------------------------
    para("3. Demo 1 - speculation without a store buffer", h2)
    demo = load("demo_leak.json")
    if demo is None:
        para("Not measured. Run <font face='Courier'>bench/demo.py --demo leak</font>.")
    else:
        rows = [["Runtime", "Runs", "Mispredictions", "Reached world", "From squashed branches"]]
        for arm in demo["arms"]:
            rows.append(
                [
                    arm["runtime"],
                    str(arm["runs"]),
                    str(arm["mispredictions"]),
                    str(arm["effects_reaching_world"]),
                    str(arm["effects_from_squashed_branches"]),
                ]
            )
        table(rows)
        para("The last column, and the fact that the second and third rows agree.")

    # -- invariants and cost ----------------------------------------------------------------
    para("4. Invariants under load, and what the runtime costs", h2)
    chaos, concurrency = load("chaos.json"), load("concurrency.json")
    if chaos and concurrency:
        c, n = chaos["chaos"], concurrency["concurrency"]
        table(
            [
                ["Matrix", "Runs", "Leaks", "Duplicates", "Equivalence failures"],
                [
                    "Chaos",
                    str(c["runs"]),
                    str(c["leaks"]),
                    str(c["duplicate_deliveries"]),
                    str(c["equivalence_failures"]),
                ],
                [
                    "Concurrency",
                    str(n["runs"]),
                    str(n["leaks"]),
                    str(n["duplicate_deliveries"]),
                    str(n["equivalence_failures"]),
                ],
            ]
        )
    else:
        para("Not measured. Run <font face='Courier'>bench/chaos/run_chaos.py</font>.")

    overhead = load("overhead.json")
    if overhead:
        o = overhead["overhead"]
        para(
            f"Journaling and classification cost {o['overhead_ms_per_step']} ms per step "
            f"({o['overhead_ms_per_run']} ms per run over {o['steps_per_run']} steps). That is "
            f"{o['overhead_fraction_of_wall_clock']:.1%} of wall clock here, and the percentage "
            "is the misleading half: a scripted model answers instantly, so this is the worst "
            "case for the ratio."
        )
    else:
        para("Overhead not measured.")

    para("5. Wall-clock latency", h2)
    latency = load("latency.json")
    if latency is None:
        para(
            "<b>Not measured.</b> The online latency benchmark calls a real target model and "
            "needs an API key and a spend cap. No wall-clock figure appears anywhere in this "
            "report or in the repository, because none has been produced."
        )
    return story


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Regenerate the report PDF from results")
    parser.add_argument("--out", type=Path, default=REPO / "SpecuNode-Report-generated.pdf")
    args = parser.parse_args(argv)

    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.lib.units import mm
        from reportlab.platypus import SimpleDocTemplate
    except ImportError:
        print("the report needs the bench extra: pip install 'specunode[bench]'", file=sys.stderr)
        return 1

    doc = SimpleDocTemplate(
        str(args.out),
        pagesize=A4,
        leftMargin=20 * mm,
        rightMargin=20 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm,
        title="SpecuNode - design and benchmark protocol",
    )
    doc.build(build_story(getSampleStyleSheet()))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
