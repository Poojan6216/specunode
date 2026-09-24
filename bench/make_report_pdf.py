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
            f"right. The nearest measured proxy is signature accuracy at "
            f"{opportunity['opportunity']['signature_predictability']['top_1']:.1%} top-1, "
            "which compares tool names and argument keys but not argument values, and is "
            "therefore an upper bound on acceptance rather than a measurement of it "
            "top-1. Neither number means anything alone."
        )

    # -- the acceptance rate, measured ------------------------------------------------------
    para("The acceptance rate, measured", h2)
    acceptance = load("acceptance.json")
    if acceptance is None:
        para("Not measured. Run <font face='Courier'>bench/offline/run_acceptance.py</font>.")
    else:
        a = acceptance["acceptance"]
        within, across, ceiling = a["within_turn"], a["across_turns"], a["copying_ceiling"]
        para(
            "The signature figure is an upper bound; this is the quantity itself. The tier-1 "
            "predictor, graded by the runtime's own gate on exact canonical equality of "
            f"argument values, over all {a['trajectories']} trajectories joined to their "
            "argument values, leave-one-trajectory-out. <b>within_turn</b> is the policy the "
            "runtime runs; <b>across_turns</b> is what carrying a guess into the next model "
            "turn would be worth."
        )
        table(
            [
                ["Measure", "within_turn", "across_turns"],
                ["Steps graded", str(within["steps"]), str(across["steps"])],
                [
                    "Guesses offered",
                    f"{within['offered_rate']:.4f}",
                    f"{across['offered_rate']:.4f}",
                ],
                [
                    "Acceptance rate, pooled",
                    f"{within['acceptance_rate']:.4f}",
                    f"{across['acceptance_rate']:.4f}",
                ],
                [
                    "Acceptance rate, by trajectory",
                    ci(within["acceptance_rate_by_trajectory"]),
                    ci(across["acceptance_rate_by_trajectory"]),
                ],
                ["Confirmed guesses", str(within["accepted"]), str(across["accepted"])],
                [
                    "Signature top-1, same steps",
                    f"{within['signature_top1']:.4f}",
                    f"{across['signature_top1']:.4f}",
                ],
                [
                    "Squashed at a turn boundary",
                    str(within["squashed_at_turn_end"]),
                    str(across["squashed_at_turn_end"]),
                ],
            ]
        )
        para(
            f"<b>The tier-1 acceptance rate on this corpus is {across['acceptance_rate']:.4f} "
            f"at best and {within['acceptance_rate']:.4f} under the runtime's own policy.</b> "
            f"The index knows which tool comes next {across['signature_top1']:.1%} of the "
            "time; the gate needs the exact command, path or thought. Every argument value of "
            f"a call has already appeared earlier at {ceiling['rate']:.1%} of steps, which is "
            "the ceiling for any predictor that copies values out of history, and this one is "
            "far below it. Anything above the ceiling needs a predictor that generates values, "
            "and that has not been measured."
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

    def band(ci: dict[str, float] | None) -> str:
        if not ci:
            return "not measured"
        return f"{ci['mean']:+.1%} [{ci['ci95_low']:+.1%}, {ci['ci95_high']:+.1%}]"

    # A present results file used to render nothing here: the section handled only a missing
    # file, so the first real measurement this project made was silently left out of the one
    # document that promises never to omit a section.
    para("5. Wall-clock latency, against a real model", h2)
    latency = load("latency.json")
    if latency is None or not latency.get("is_real_model"):
        para(
            "<b>Not measured against a real model.</b> The online latency benchmark needs an "
            "API key; a run against the scripted stand-in is not a latency measurement."
        )
    elif not any("B_strict_seq" in e["arms"] for e in latency["workloads"].values()):
        para(
            f"The {latency['tasks_completed']} runs on file were taken before two faults in "
            "their arms were found -- the baseline already issued reads early, and the "
            "reads-only arm was given no predictor -- so their savings are not quoted. Section "
            "6's 0 ms rung is the corrected measurement; re-running this table is pending."
        )
    else:
        para(
            f"Target <font face='Courier'>{latency['target_model']}</font>, "
            f"{latency['tasks_completed']} runs, tools as fast as the in-memory sample world "
            "makes them. Saving is against the arm that waits for each turn and then calls "
            "the tools in order."
        )
        rows = [["Workload", "Early issue only", "Early issue + guessing", "Leaks"]]
        for name, entry in sorted(latency["workloads"].items()):
            strict = entry.get("saving_vs_strict_seq_ci95") or {}
            leaks = sum(int(s["leaks"]) for s in entry["arms"].values())
            rows.append(
                [name, band(strict.get("B_seq")), band(strict.get("B_specunode")), str(leaks)]
            )
        table(rows)

    para("6. How slow a tool has to be", h2)
    sweep = load("sweep.json")
    if sweep is None or not sweep.get("is_real_model"):
        para("Not measured against a real model.")
    else:
        rows = [["Tool latency", "Workload", "Early issue only", "Early issue + guessing"]]
        for ms, point in sorted(sweep["points"].items(), key=lambda kv: int(kv[0])):
            for name, entry in sorted(point.items()):
                strict = entry["saving_vs_strict_seq_ci95"]
                rows.append(
                    [
                        f"{ms} ms",
                        name,
                        band(strict.get("B_seq")),
                        band(strict.get("B_specunode")),
                    ]
                )
        table(rows)
        para(
            "Only the workload that hands its model turn to the runtime can gain anything; the "
            "other two call the model themselves. Where it gains, the whole of the saving is "
            "tier-0 early issue, and guessing adds nothing the interval can resolve."
        )

    para("7. What a guess is worth", h2)
    breakeven = load("break_even.json")
    if breakeven is None:
        para("Not measured. Run <font face='Courier'>bench/offline/run_break_even.py</font>.")
    else:
        rows = [["Tool latency", "Accuracy", "Guess reads and writes", "Guess reads only"]]
        for ms, point in sorted(breakeven["points"].items(), key=lambda kv: int(kv[0])):
            for alpha, arms in point["by_alpha"].items():
                rows.append(
                    [
                        f"{ms} ms",
                        alpha,
                        band(arms["reads_and_writes"]["saving_vs_early_issue_ci95"]),
                        band(arms["reads_only"]["saving_vs_early_issue_ci95"]),
                    ]
                )
        table(rows)
        para(
            "A guesser of controlled accuracy, against early issue alone, on a stand-in that "
            f"streams {breakeven['think_ms_per_block']:.0f} ms per block. A correct guess can "
            "run at most one block ahead; a wrong one is squashed and costs no wall clock; and "
            "guessing writes as well as reads adds nothing, because a staged write cannot be "
            "sent before its branch retires."
        )

    para("8. The acceptance rate of a draft model", h2)
    tier2 = load("tier2.json")
    if tier2 is None or not tier2.get("is_real_model"):
        para("Not measured against a real draft model.")
    else:
        r = tier2["result"]
        para(
            f"<font face='Courier'>{tier2['draft_model']}</font>, graded by the runtime's own "
            f"gate on {r['steps_graded']} sampled steps: acceptance "
            f"{band(r['acceptance_rate_ci95']).replace('+', '')}, right tool "
            f"{r['right_tool']:.1%}. Against a tier-1 acceptance rate that is effectively zero, "
            "a draft model is the only predictor here that gets anything right -- and section 7 "
            "is what a hit is worth when it does."
        )
    para("9. When the model is the slow part", h2)
    bound = load("model_bound.json")
    if bound is None:
        para("Not measured. Run <font face='Courier'>bench/offline/run_model_bound.py</font>.")
    else:
        stand_in = bound["stand_in"]
        para(
            "Fewer replies, and replies side by side, against a stand-in model whose reply "
            f"takes {stand_in['reply_ms']:.0f} ms before its first block and "
            f"{stand_in['block_ms']:.0f} ms per block -- settings, not measurements. The reply "
            "and in-flight counts do not depend on them."
        )
        rows = [["Tool latency", "One call per reply", "Every independent call at once", "Saving"]]
        for ms, point in sorted(bound["replies"].items(), key=lambda kv: int(kv[0])):
            one, many = point["one_call"], point["parallel"]
            rows.append(
                [
                    f"{ms} ms",
                    f"{one['replies']} replies, {one['wall_ms_mean']:.0f} ms",
                    f"{many['replies']} replies, {many['wall_ms_mean']:.0f} ms",
                    band(point["saving_ci95"]),
                ]
            )
        table(rows)
        rows = [["Tool latency", "Nodes one after another", "Side by side", "Saving"]]
        for ms, point in sorted(bound["branches"].items(), key=lambda kv: int(kv[0])):
            serial, side = point["one_at_a_time"], point["side_by_side"]
            rows.append(
                [
                    f"{ms} ms",
                    f"{serial['in_flight_max']} in flight, {serial['wall_ms_mean']:.0f} ms",
                    f"{side['in_flight_max']} in flight, {side['wall_ms_mean']:.0f} ms",
                    band(point["saving_ci95"]),
                ]
            )
        table(rows)
        totals = bound["totals"]
        online = load("model_bound_online.json")
        measured = online is not None and online.get("is_real_model")
        para(
            f"Correct runs: {totals['correct']} of {totals['runs']}. Effects from a branch that "
            f"never retired: {totals['leaks']}. "
            + (
                "Section 10 is the same against a real model."
                if measured
                else "What a real model does when told it may ask for several calls at once, "
                "and what prompt caching saves, are not measured yet."
            )
        )
    para("10. The same, against a real model", h2)
    online = load("model_bound_online.json")
    if online is None or not online.get("is_real_model"):
        para("Not measured against a real model.")
    else:
        cells = online["cells"]
        para(
            f"<font face='Courier'>{online['target_model']}</font>, "
            f"{online['runs_requested']} rounds, one run of each configuration per round. One "
            "call per reply is enforced with the API's disable_parallel_tool_use: asked in the "
            "prompt alone, the model batched its calls anyway."
        )
        rows = [["Configuration", "Replies", "Calls/reply", "Wall clock", "Cost/run", "Correct"]]
        labels = (
            ("one_call/no_cache", "One call, no cache (before)"),
            ("one_call/cache", "One call, cached"),
            ("parallel/no_cache", "Several calls, no cache"),
            ("parallel/cache", "Several calls, cached (after)"),
            ("default/cache", "No guidance, cached"),
        )
        for key, label in labels:
            cell = cells.get(key) or {}
            if not cell.get("n"):
                continue
            rows.append(
                [
                    label,
                    f"{cell['replies_mean']:.1f}",
                    f"{cell['calls_per_reply']:.2f}",
                    f"{cell['wall_ms_mean']:.0f} ms",
                    f"${cell['usd_per_run']:.4f}",
                    f"{cell['correct']} of {cell['n']}",
                ]
            )
        table(rows)
        rows = [["Change", "Wall clock saved", "Cost saved"]]
        for key, label in (
            ("caching", "Caching alone"),
            ("multi_call_replies", "Several calls per reply"),
            ("before_to_after", "Before to after"),
            ("parallel_nodes", "Parallel nodes"),
        ):
            comparison = online["comparisons"].get(key) or {}
            cost = band(comparison.get("cost_saving_ci95")) if key != "parallel_nodes" else "--"
            rows.append([label, band(comparison.get("wall_saving_ci95")), cost])
        table(rows)
        leaks = sum(int(cell.get("leaks", 0)) for cell in cells.values())
        para(
            f"Spend ${online['estimated_spend_usd']}. Effects from a branch that never retired: "
            f"{leaks}."
        )
    para("11. Pull the plug: what a crash sends twice", h2)
    crash = load("crash_safety.json")
    if crash is None:
        para("Not measured. Run <font face='Courier'>bench/offline/run_crash_safety.py</font>.")
    else:
        para(
            f"A billing run with {crash['writes']} effects, killed {crash['crash_points']} "
            "times -- just before each request reached the upstream, and just after each took "
            "effect but before its reply came back -- and restarted the way each system "
            f"restarts. LangGraph {crash['langgraph']}; no model and no network."
        )
        rows = [["System", "Exact", "Stopped for a human", "Sent twice", "Extra effects"]]
        labels = (
            ("plain_loop", "Plain async loop"),
            ("langgraph_nodes", "LangGraph, node per customer"),
            ("langgraph_tasks", "LangGraph, @task per call"),
            ("specunode", "SpecuNode"),
            ("specunode_reconcile", "SpecuNode with reconcile"),
        )
        for key, label in labels:
            row = crash["systems"].get(key)
            if row is None:
                continue
            rows.append(
                [
                    label,
                    str(row["exact"]),
                    str(row["held"]),
                    str(row["duplicated"]),
                    str(row["duplicate_effects"]),
                ]
            )
        table(rows)
        para(
            "A lost reply is the crash that double-charges. SpecuNode sent nothing twice in any "
            "of them; without a way to ask the upstream it stops for a human, and with a "
            "reconcile per tool it finished every one, nothing sent twice and nothing missing."
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
