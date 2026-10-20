from __future__ import annotations

import re
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

from transcript_pipeline.cli import PROG, build_parser

ROOT = Path(__file__).resolve().parents[1]
DOCS = (ROOT / "README.md", ROOT / "skill" / "process-recording" / "SKILL.md")
EXPECTED_SUBCOMMANDS = {
    "prepare",
    "infer-event",
    "extract-frames",
    "validate",
    "render",
    "status",
    "resume",
}


def _documentation() -> str:
    return "\n".join(path.read_text(encoding="utf-8") for path in DOCS)


def _shell_lines() -> list[str]:
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    lines: list[str] = []
    for block in re.findall(r"```(?:bash|sh|console)\n(.*?)```", text, re.DOTALL):
        for raw_line in block.splitlines():
            line = raw_line.strip()
            if line and not line.startswith("#"):
                lines.append(line.removeprefix("$ "))
    return lines


def test_every_readme_command_parses_and_every_subcommand_is_documented() -> None:
    parser = build_parser()
    documented: set[str] = set()
    for line in _shell_lines():
        argv = shlex.split(line)
        if argv[:3] == ["uv", "run", PROG]:
            parser.parse_args(argv[3:])
            documented.add(argv[3])
        else:
            assert argv in (
                ["uv", "sync", "--extra", "ml"],
                ["uv", "sync", "--extra", "test"],
                ["uv", "run", "pytest"],
            ), f"unverified command in README: {line}"
    choices = next(action.choices for action in parser._actions if getattr(action, "choices", None))
    assert set(choices) == EXPECTED_SUBCOMMANDS
    assert documented == EXPECTED_SUBCOMMANDS


@pytest.mark.parametrize("subcommand", sorted(EXPECTED_SUBCOMMANDS))
def test_subcommand_help_loads_without_the_ml_stack(subcommand: str) -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "transcript_pipeline", subcommand, "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert f"{PROG} {subcommand}" in completed.stdout


def test_docs_never_show_credentials_or_pass_them_as_arguments() -> None:
    text = _documentation()
    shell = "\n".join(_shell_lines())

    assert re.search(r"\bHF_TOKEN\s*=\s*\S+", text) is None
    assert re.search(r"\bhf_[A-Za-z0-9]{20,}", text) is None
    assert re.search(r"Authorization\s*:\s*\w+\s+\S+", text, re.IGNORECASE) is None
    assert re.search(r"--(?:hf-)?token\b", shell) is None
    assert re.search(r"\b(?:curl|wget)\b", shell) is None


def test_validator_codes_named_in_the_skill_exist() -> None:
    skill = (ROOT / "skill" / "process-recording" / "SKILL.md").read_text(encoding="utf-8")
    source = (ROOT / "transcript_pipeline" / "evidence.py").read_text(encoding="utf-8")

    for code in ("quote_not_found", "speaker_mismatch", "binding_not_definitive", "answer_precedes_question"):
        assert f"`{code}`" in skill
        assert f'"{code}"' in source
