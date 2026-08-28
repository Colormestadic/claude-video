"""setup.py --json surfaces the resolved watch detail."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

SETUP = Path(__file__).resolve().parent.parent / "skills" / "watch" / "scripts" / "setup.py"


def _run(args, *, home=None, extra_env=None):
    env = dict(os.environ)
    env.pop("WATCH_DETAIL", None)
    # Don't let a real key in the developer's shell env leak into the test.
    env.pop("GROQ_API_KEY", None)
    env.pop("OPENAI_API_KEY", None)
    env.pop("SETUP_COMPLETE", None)
    # Local whisper satisfies the same gate an API key does, so a developer
    # machine that happens to have it installed would silently turn every
    # "keyless" assertion below into a no-op. Point the override at a path that
    # cannot exist; tests covering the local backend opt back in via extra_env.
    env["WATCH_LOCAL_WHISPER_BIN"] = str(Path(os.sep) / "nonexistent" / "whisper")
    if home is not None:
        env["HOME"] = str(home)
        env["USERPROFILE"] = str(home)  # Windows
    if extra_env:
        env.update(extra_env)
    return subprocess.run(
        [sys.executable, str(SETUP), *args],
        capture_output=True, text=True, env=env,
    )


def _write_env(home: Path, body: str) -> None:
    cfg = home / ".config" / "watch"
    cfg.mkdir(parents=True, exist_ok=True)
    f = cfg / ".env"
    f.write_text(body, encoding="utf-8")
    f.chmod(0o600)


def test_json_reports_watch_detail():
    proc = _run(["--json"])
    assert proc.returncode == 0, proc.stderr
    data = json.loads(proc.stdout)
    assert data["watch_detail"] == "balanced"


def test_keyless_completed_setup_proceeds_silently(tmp_path):
    """A user who finished setup without a key must NOT be nagged forever."""
    _write_env(tmp_path, "GROQ_API_KEY=\nOPENAI_API_KEY=\nSETUP_COMPLETE=true\n")
    chk = _run(["--check"], home=tmp_path)
    assert chk.returncode == 0, f"keyless-complete should pass --check; got {chk.returncode}: {chk.stderr}"
    assert chk.stdout == "" and chk.stderr == ""

    js = json.loads(_run(["--json"], home=tmp_path).stdout)
    assert js["can_proceed"] is True
    assert js["first_run"] is False
    assert js["setup_complete"] is True
    # status still encourages a key even though we can proceed
    assert js["status"] == "needs_key"


def test_keyless_first_run_is_encouraged(tmp_path):
    """First run with no key AND no local whisper: --check reports exit 3."""
    _write_env(tmp_path, "GROQ_API_KEY=\nOPENAI_API_KEY=\n")
    chk = _run(["--check"], home=tmp_path)
    assert chk.returncode == 3, chk.stderr

    js = json.loads(_run(["--json"], home=tmp_path).stdout)
    assert js["can_proceed"] is False
    assert js["first_run"] is True


def test_key_present_is_ready(tmp_path):
    _write_env(tmp_path, "GROQ_API_KEY=sk-test-abc\n")
    chk = _run(["--check"], home=tmp_path)
    assert chk.returncode == 0, chk.stderr

    js = json.loads(_run(["--json"], home=tmp_path).stdout)
    assert js["status"] == "ready"
    assert js["can_proceed"] is True
    assert js["whisper_backend"] == "groq"


def test_local_whisper_satisfies_the_key_gate(tmp_path):
    """CMS fork: local whisper is a real backend, so a keyless first run is ready.

    Without this, a machine that can already transcribe for free offline gets
    nagged to buy an API key on every single call.
    """
    fake = tmp_path / "bin" / "whisper"
    fake.parent.mkdir(parents=True, exist_ok=True)
    fake.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    fake.chmod(0o755)

    _write_env(tmp_path, "GROQ_API_KEY=\nOPENAI_API_KEY=\n")
    chk = _run(["--check"], home=tmp_path, extra_env={"WATCH_LOCAL_WHISPER_BIN": str(fake)})
    assert chk.returncode == 0, f"local whisper should satisfy the gate; got {chk.stderr}"
    assert chk.stdout == "" and chk.stderr == ""

    js = json.loads(
        _run(["--json"], home=tmp_path, extra_env={"WATCH_LOCAL_WHISPER_BIN": str(fake)}).stdout
    )
    assert js["local_whisper"] is True
    assert js["has_api_key"] is False, "local must not be reported as an API key"
    assert js["whisper_backend"] == "local"
    assert js["status"] == "ready"
    assert js["can_proceed"] is True


def test_api_key_still_wins_over_local(tmp_path):
    """A configured key keeps upstream behaviour; local is the fallback, not an override."""
    fake = tmp_path / "bin" / "whisper"
    fake.parent.mkdir(parents=True, exist_ok=True)
    fake.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    fake.chmod(0o755)

    _write_env(tmp_path, "GROQ_API_KEY=sk-test-abc\n")
    js = json.loads(
        _run(["--json"], home=tmp_path, extra_env={"WATCH_LOCAL_WHISPER_BIN": str(fake)}).stdout
    )
    assert js["whisper_backend"] == "groq"
    assert js["has_api_key"] is True
    assert js["local_whisper"] is True


def test_installer_reports_ready_on_a_keyless_local_machine(tmp_path):
    """cmd_install must use the same readiness rule as _status().

    Checking only for an API key reports a working local-only install as a
    failed setup and exits 3, every time the installer is run.
    """
    fake = tmp_path / "bin" / "whisper"
    fake.parent.mkdir(parents=True, exist_ok=True)
    fake.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    fake.chmod(0o755)

    _write_env(tmp_path, "GROQ_API_KEY=\nOPENAI_API_KEY=\n")
    proc = _run([], home=tmp_path, extra_env={"WATCH_LOCAL_WHISPER_BIN": str(fake)})
    assert proc.returncode == 0, f"local-only install should succeed: {proc.stdout}{proc.stderr}"
    assert "whisper backend: local" in proc.stdout
    assert "one step left" not in proc.stdout

    # and it must have marked setup complete, so /watch stops re-running setup
    js = json.loads(
        _run(["--json"], home=tmp_path, extra_env={"WATCH_LOCAL_WHISPER_BIN": str(fake)}).stdout
    )
    assert js["setup_complete"] is True
