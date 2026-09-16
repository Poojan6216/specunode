"""Hard Rule 1: no LLM in the control path.

Which tool calls run, when, and whether a speculation matched are decided by deterministic
code. The only models the runtime knows about are the *target* model -- whose output is the
ground truth that resolves a branch -- and an optional *draft* model, whose output is only
ever a guess that the gate must then confirm by exact canonical equality.

So the four control packages may **pass a prompt through**, but must never **author one**,
and must never ask a model a question whose answer changes control flow.

The spec's own check is ``grep`` for ``messages=`` and ``prompt``. A literal grep cannot tell
``def complete(self, messages: Sequence[Message])`` -- a protocol signature, which is fine --
from ``messages=[{"role": "user", "content": "are these two calls the same?"}]``, which is the
thing the rule exists to stop; and it would fire on every docstring that discusses prompts,
of which Hard Rule 13 needs many. This module therefore parses the AST and checks the
property the rule actually cares about: *no prompt content originates in these packages*.
The literal ``messages=[`` grep the spec names is kept as well, so the planted-bug check the
spec describes still fires.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "src" / "specunode"

#: The packages that decide control flow. A prompt written in any of these would mean a model
#: is being asked to make a runtime decision.
CONTROL_PACKAGES = ("core", "buffer", "verify", "journal")

#: Keyword arguments whose value, if it is a literal, means this code is authoring content
#: for a model rather than forwarding the developer's.
CONTENT_KEYWORDS = {"messages", "prompt", "system", "system_prompt", "instructions", "input"}

#: Model SDKs. Importing one inside a control package means the control path can call a model.
MODEL_SDKS = {
    "anthropic",
    "openai",
    "transformers",
    "torch",
    "langchain",
    "langchain_core",
    "langchain_anthropic",
    "llama_cpp",
    "ollama",
    "google",
    "cohere",
    "mistralai",
}

#: Text that only appears in something written to be read by a model.
INSTRUCTION_PATTERNS = re.compile(
    r"\b(you are (?:a|an|the)\b|your task is|respond with|answer (?:the|this|with)|"
    r"act as|reply with|think step by step|as an ai|please (?:answer|respond|decide|choose)|"
    r"\brole\"?:\s*\"?(?:system|user|assistant))",
    re.IGNORECASE,
)

LITERAL_NODES = (ast.Constant, ast.JoinedStr, ast.List, ast.Dict, ast.Tuple)


def control_path_files() -> list[Path]:
    files = [p for pkg in CONTROL_PACKAGES for p in (SRC / pkg).rglob("*.py")]
    assert files, "no control-path source files found; the layout moved and this test went blind"
    return sorted(files)


def _is_literal(node: ast.expr) -> bool:
    """True for a value written out here, rather than one received from a caller."""
    if isinstance(node, ast.Constant):
        return isinstance(node.value, str)
    if isinstance(node, ast.JoinedStr):
        return True
    if isinstance(node, (ast.List, ast.Tuple)):
        return len(node.elts) > 0
    if isinstance(node, ast.Dict):
        return len(node.keys) > 0
    return False


def audit_source(source: str, label: str = "<memory>") -> list[str]:
    """Return one message per Hard Rule 1 violation found in ``source``."""
    problems: list[str] = []
    tree = ast.parse(source)

    # The literal grep the spec names, kept verbatim so the planted-bug check still fires.
    for lineno, line in enumerate(source.splitlines(), start=1):
        if re.search(r"\bmessages\s*=\s*\[", line):
            problems.append(f"{label}:{lineno}: builds a message list here: {line.strip()!r}")

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            for keyword in node.keywords:
                if keyword.arg in CONTENT_KEYWORDS and _is_literal(keyword.value):
                    problems.append(
                        f"{label}:{node.lineno}: passes a literal {keyword.arg}= to a call; "
                        "control-path code may forward a prompt but never author one"
                    )
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            names = [t.id.lower() for t in targets if isinstance(t, ast.Name)]
            value = node.value
            authors_content = any(k in n for n in names for k in CONTENT_KEYWORDS)
            if value is not None and authors_content and _is_literal(value):
                problems.append(
                    f"{label}:{node.lineno}: assigns literal prompt content to {names[0]!r}"
                )
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] in MODEL_SDKS:
                    problems.append(f"{label}:{node.lineno}: imports model SDK {alias.name!r}")
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if root in MODEL_SDKS:
                problems.append(f"{label}:{node.lineno}: imports model SDK {node.module!r}")

    # Instruction-shaped strings, but not inside docstrings: Hard Rule 13's design notes have
    # to be able to talk about prompts.
    docstrings = {
        id(n.body[0].value)
        for n in ast.walk(tree)
        if isinstance(n, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
        and n.body
        and isinstance(n.body[0], ast.Expr)
        and isinstance(n.body[0].value, ast.Constant)
    }
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and id(node) not in docstrings
            and INSTRUCTION_PATTERNS.search(node.value)
        ):
            problems.append(
                f"{label}:{node.lineno}: string literal reads as a model instruction: "
                f"{node.value[:60]!r}"
            )
    return problems


def test_no_prompt_originates_in_a_control_package() -> None:
    problems: list[str] = []
    for path in control_path_files():
        problems += audit_source(path.read_text(encoding="utf-8"), str(path.relative_to(REPO)))
    assert not problems, (
        "Hard Rule 1: no LLM in the control path. Which calls run, when, and whether a "
        "speculation matched are decided by deterministic code.\n\n" + "\n".join(problems)
    )


def test_the_control_packages_exist_and_are_actually_scanned() -> None:
    """A rule that scans nothing passes forever. Fail loudly if the layout moves."""
    for package in CONTROL_PACKAGES:
        assert (SRC / package).is_dir(), f"control package src/specunode/{package} is missing"
    assert len(control_path_files()) >= len(CONTROL_PACKAGES)


# --------------------------------------------------------------------------------------
# Planted-bug proofs. Spec task 0.5: "planting `messages=[` in src/specunode/core/policy.py
# fails the control-path test".
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "planted",
    [
        'messages=[{"role": "user", "content": "did the speculation match?"}]',
        'client.complete(messages=[{"role": "user", "content": "x"}])',
        'PROMPT = "You are a judge. Answer YES if these two tool calls are the same."',
        'judge_prompt = f"Are {a} and {b} equivalent?"',
        "import anthropic",
        "from openai import OpenAI",
        'reply = call(system="Act as an arbiter of tool-call equality.")',
    ],
)
def test_the_detector_fires_on_planted_violations(planted: str) -> None:
    assert audit_source(planted, "planted.py"), f"detector missed: {planted!r}"


@pytest.mark.parametrize(
    "legitimate",
    [
        # A protocol signature forwards the developer's messages; it does not author them.
        "async def complete(self, messages: Sequence[Message]) -> Response: ...",
        "return await self._target.complete(messages, tools=tools)",
        'entry = {"kind": "model_request", "prompt_hash": chash(prompt)}',
        '"""Rule 13: a placeholder must never appear in a prompt the model is sent."""',
        "messages = list(branch.context)",
        "prompt_hash = chash_bytes(canonical(messages))",
    ],
)
def test_the_detector_does_not_fire_on_forwarding(legitimate: str) -> None:
    assert not audit_source(legitimate, "ok.py"), f"detector false-positived on: {legitimate!r}"
