"""``specunode sign-ledger`` and ``verify-ledger``, end to end (spec task 7.x).

`verify-ledger` could never verify anything. `sign-ledger` echoed the signature envelope to the
terminal and wrote it nowhere; `build_ledger` never assigns `signature`; so verification always
saw an unsigned ledger and returned `unsigned` before running any of its four checks. Both
commands ran, exited, printed plausible output, and together proved nothing.

The envelope is written **beside** the journal rather than into it, and that is forced rather
than stylistic: the signed payload covers `journal_head` and `journal_entries`, so appending the
signature to the journal it signs changes the material it was computed over and it stops
verifying against a freshly built ledger.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from specunode.cli import app

REPO = Path(__file__).resolve().parents[2]
HELPER = REPO / "bench" / "_demo3_agent.py"
RUN_ID = "01SIGTESTAAAAAAAAAAAAAAAAA"

pytestmark = pytest.mark.slow


def a_real_run(directory: Path) -> Path:
    """A journal with effects in it, produced by the actual runtime."""
    done = subprocess.run(
        [sys.executable, str(HELPER), str(directory), RUN_ID, "-1"],
        capture_output=True,
        text=True,
        cwd=REPO,
        timeout=180,
    )
    assert done.returncode == 0, done.stderr[-2000:]
    return directory / "journal.db"


def invoke(*args: str) -> object:
    return CliRunner().invoke(app, list(args))


def test_signing_writes_the_envelope_somewhere_verification_can_find_it(
    tmp_path: Path,
) -> None:
    journal = a_real_run(tmp_path)
    keystore = tmp_path / "keys"

    signed = invoke("sign-ledger", RUN_ID, "--journal", str(journal), "--keystore", str(keystore))
    assert signed.exit_code == 0, signed.output  # type: ignore[attr-defined]

    envelope = journal.parent / "ledgers" / f"{RUN_ID}.sig"
    assert envelope.is_file(), "the signature was printed and stored nowhere"
    assert envelope.read_text(encoding="utf-8").startswith("ed25519:")


def test_a_signed_ledger_verifies(tmp_path: Path) -> None:
    journal = a_real_run(tmp_path)
    keystore = tmp_path / "keys"
    invoke("sign-ledger", RUN_ID, "--journal", str(journal), "--keystore", str(keystore))

    result = invoke("verify-ledger", RUN_ID, "--journal", str(journal), "--keystore", str(keystore))

    assert result.exit_code == 0, result.output  # type: ignore[attr-defined]
    assert "ok" in result.output.lower()  # type: ignore[attr-defined]
    assert "journal: verified" in result.output  # type: ignore[attr-defined]


def test_an_unsigned_ledger_is_reported_as_unsigned_not_as_ok(tmp_path: Path) -> None:
    """The old behaviour, which is correct *when nothing has been signed*."""
    journal = a_real_run(tmp_path)
    result = invoke(
        "verify-ledger", RUN_ID, "--journal", str(journal), "--keystore", str(tmp_path / "keys")
    )
    assert result.exit_code != 0  # type: ignore[attr-defined]
    assert "unsigned" in result.output.lower()  # type: ignore[attr-defined]


def test_a_tampered_signature_does_not_verify(tmp_path: Path) -> None:
    """Otherwise the command would accept anything in the file and still print ok."""
    journal = a_real_run(tmp_path)
    keystore = tmp_path / "keys"
    invoke("sign-ledger", RUN_ID, "--journal", str(journal), "--keystore", str(keystore))

    envelope = journal.parent / "ledgers" / f"{RUN_ID}.sig"
    original = envelope.read_text(encoding="utf-8").strip()
    scheme, key_id, public, signature = original.split(":")
    # Flip one base64 character of the signature itself.
    flipped = "A" if signature[0] != "A" else "B"
    envelope.write_text(f"{scheme}:{key_id}:{public}:{flipped}{signature[1:]}\n", encoding="utf-8")

    result = invoke("verify-ledger", RUN_ID, "--journal", str(journal), "--keystore", str(keystore))
    assert result.exit_code != 0, result.output  # type: ignore[attr-defined]
    assert "ok:" not in result.output.lower().split("\n")[0]  # type: ignore[attr-defined]


def test_an_edited_ledger_does_not_verify_against_its_signature(tmp_path: Path) -> None:
    """The property signing exists for: the rows cannot change after the fact."""
    journal = a_real_run(tmp_path)
    keystore = tmp_path / "keys"
    invoke("sign-ledger", RUN_ID, "--journal", str(journal), "--keystore", str(keystore))

    # Append another entry, which moves journal_head and journal_entries -- both of which the
    # signed payload covers.
    from specunode.journal.journal import Journal, close_all_writers

    store = Journal(journal)
    store.append(RUN_ID, "policy_event", {"v": 1, "event": "tick", "reason": "tamper"})
    close_all_writers()

    result = invoke("verify-ledger", RUN_ID, "--journal", str(journal), "--keystore", str(keystore))
    assert result.exit_code != 0, result.output  # type: ignore[attr-defined]
