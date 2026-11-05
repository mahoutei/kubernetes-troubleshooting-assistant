from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import dev


@pytest.mark.parametrize("project", ["rag", "finetune", "all"])
def test_selection_and_project_working_directories(tmp_path, project):
    steps = dev.build_steps("check", project, tmp_path)
    selected = set(dev.PROJECTS) if project == "all" else {project}
    assert {s.label.split(":")[0] for s in steps} == {"tooling", *selected}
    for step in steps:
        owner = step.label.split(":")[0]
        assert step.cwd == (tmp_path if owner == "tooling" else tmp_path / dev.PROJECTS[owner])
    assert any(s.label == "rag: gold questions" for s in steps) == ("rag" in selected)


@pytest.mark.parametrize("action", ["test", "lint"])
def test_single_action_does_not_run_other_actions(action):
    steps = dev.build_steps(action, "all")
    assert len(steps) == 3
    expected = "tests" if action == "test" else "lint"
    assert all(s.label.endswith(expected) for s in steps)


def test_subprocess_uses_active_interpreter_arguments_and_explicit_cwd(monkeypatch, tmp_path):
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(dev.subprocess, "run", run)
    steps = dev.build_steps("check", "rag", tmp_path)
    assert dev.run_steps(steps) == 0
    assert len(calls) == len(steps)
    for (command, kwargs), step in zip(calls, steps, strict=True):
        assert isinstance(command, list)
        assert command[0] == dev.sys.executable
        assert command == list(step.command)
        assert kwargs == {"cwd": step.cwd, "check": False}


def test_failed_steps_do_not_skip_independent_checks(monkeypatch, capsys):
    codes = iter([1, 0, 2, 0])
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=next(codes))

    monkeypatch.setattr(dev.subprocess, "run", run)
    steps = [dev.Step(f"step {i}", Path.cwd(), ("python",)) for i in range(4)]
    assert dev.run_steps(steps) == 1
    assert len(calls) == 4
    assert "2 passed; 2 failed" in capsys.readouterr().out


def test_launch_failure_is_reported_and_later_steps_run(monkeypatch, capsys):
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        if len(calls) == 1:
            raise FileNotFoundError("missing tool")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(dev.subprocess, "run", run)
    steps = [dev.Step("missing", Path.cwd(), ("bad",)), dev.Step("later", Path.cwd(), ("ok",))]
    assert dev.run_steps(steps) == 1
    assert len(calls) == 2
    assert "missing tool" in capsys.readouterr().err


@pytest.mark.parametrize("outcome", [KeyboardInterrupt(), -2, 130])
def test_interruption_stops_subsequent_checks(monkeypatch, outcome):
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        if isinstance(outcome, BaseException):
            raise outcome
        return SimpleNamespace(returncode=outcome)

    monkeypatch.setattr(dev.subprocess, "run", run)
    steps = [dev.Step("first", Path.cwd(), ("python",)), dev.Step("later", Path.cwd(), ("python",))]
    assert dev.run_steps(steps) == 130
    assert len(calls) == 1


def test_cli_defaults_to_checking_both_projects(monkeypatch):
    captured = []
    monkeypatch.setattr(dev, "run_steps", lambda steps: captured.extend(steps) or 0)
    assert dev.main(["check"]) == 0
    assert any(s.label == "rag: tests" for s in captured)
    assert any(s.label == "finetune: tests" for s in captured)


def test_cli_rejects_unknown_project():
    with pytest.raises(SystemExit) as exc:
        dev.main(["check", "--project", "unknown"])
    assert exc.value.code == 2
