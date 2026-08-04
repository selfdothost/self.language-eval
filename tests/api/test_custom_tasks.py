"""self.language-eval#2: custom task registration.

A custom task is a YAML config naming an HF dataset — the same act as adding a
Dataset, with no code introduced. These tests pin the boundaries that keep it
that way, and the two properties that decide whether the feature actually works:

* `!function` is refused, because `language_eval/utils.py::import_function`
  resolves it to a .py next to the config and calls `spec.loader.exec_module`,
* `dataset_kwargs.trust_remote_code` is refused, because it executes the
  dataset's own loading script (only stripped for datasets>=4.0.0, and the pin
  is >=2.16.0),
* the cache is invalidated on write, or an added task stays invisible until the
  pod restarts,
* `--include_path` reaches the job subprocess, or the task is listable but not
  runnable — which looks like success and isn't.
"""

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

_API_DIR = str(Path(__file__).resolve().parents[2] / "api")
if _API_DIR not in sys.path:
    sys.path.insert(0, _API_DIR)

from tests.api.conftest import TEST_SERVICE_AUTH_SECRET, mint_test_ticket  # noqa: E402

VALID_YAML = "task: my_custom_task\ndataset_path: acme/things\noutput_type: multiple_choice\n"


@pytest.fixture
def client(temp_workspace):
    """TestClient with the service-auth secret configured.

    `auth.py` reads SERVICE_AUTH_SECRET into a module constant at import time,
    so setting the env var here would be too late — the module attribute has to
    be patched directly. Same approach as tests/api/test_auth.py.
    """
    import auth as auth_module
    import main as main_module

    original_secret = auth_module.SERVICE_AUTH_SECRET
    auth_module.SERVICE_AUTH_SECRET = TEST_SERVICE_AUTH_SECRET

    yield TestClient(main_module.app)

    auth_module.SERVICE_AUTH_SECRET = original_secret


def _hdr(scope="tasks:read tasks:write"):
    return {"X-Selfai-Ticket": mint_test_ticket(scope=scope)}


def _post(client, name, yaml_config, scope="tasks:read tasks:write"):
    return client.post(
        "/api/tasks",
        json={"name": name, "yaml_config": yaml_config},
        headers=_hdr(scope),
    )


@pytest.mark.parametrize(
    "bad_yaml,reason",
    [
        ("task: my_custom_task\ndoc_to_text: !function utils.doc_to_text\n", "!function"),
        (
            "task: my_custom_task\ndataset_path: acme/things\ndataset_kwargs:\n  trust_remote_code: true\n",
            "trust_remote_code",
        ),
    ],
)
def test_code_execution_vectors_are_refused(client, bad_yaml, reason):
    """Neither may be accepted: both end in this harness executing supplied code."""
    resp = _post(client, "my_custom_task", bad_yaml)
    assert resp.status_code == 400, resp.text
    assert reason.strip("!") in resp.json()["detail"]


def test_registering_a_task_writes_it_and_invalidates_the_cache(client, temp_workspace, monkeypatch):
    import main as main_module

    # Prime the cache so we can prove it gets dropped.
    monkeypatch.setattr(main_module, "_ALL_TASKS", ["stale"])
    monkeypatch.setattr(main_module, "_discover_builtin_tasks", lambda: ["hellaswag"])

    resp = _post(client, "my_custom_task", VALID_YAML)
    assert resp.status_code == 201, resp.text
    assert resp.json()["name"] == "my_custom_task"

    written = main_module.CUSTOM_TASKS_DIR / "my_custom_task.yaml"
    assert written.is_file()
    assert written.read_text() == VALID_YAML
    # Cache dropped — otherwise the new task stays invisible until a restart.
    assert main_module._ALL_TASKS == []


def test_task_name_must_match_the_yaml(client, monkeypatch):
    import main as main_module

    monkeypatch.setattr(main_module, "_discover_builtin_tasks", lambda: [])
    resp = _post(client, "declared_one_thing", VALID_YAML)
    assert resp.status_code == 400
    assert "must match" in resp.json()["detail"]


def test_cannot_shadow_a_builtin(client, monkeypatch):
    """include_path overrides the built-in directory, so this would silently
    change what an existing benchmark means."""
    import main as main_module

    monkeypatch.setattr(main_module, "_discover_builtin_tasks", lambda: ["hellaswag"])
    resp = _post(client, "hellaswag", "task: hellaswag\ndataset_path: acme/fake\n")
    assert resp.status_code == 409
    assert "built-in" in resp.json()["detail"]


@pytest.mark.parametrize("bad_name", ["../escape", "with/slash", "UPPER", "", ".hidden"])
def test_unsafe_names_are_refused(client, monkeypatch, bad_name):
    import main as main_module

    monkeypatch.setattr(main_module, "_discover_builtin_tasks", lambda: [])
    resp = _post(client, bad_name, f"task: {bad_name}\n")
    assert resp.status_code == 400


def test_invalid_yaml_is_refused(client, monkeypatch):
    import main as main_module

    monkeypatch.setattr(main_module, "_discover_builtin_tasks", lambda: [])
    resp = _post(client, "broken", "task: broken\n  bad: [indent\n")
    assert resp.status_code == 400


def test_oversize_config_is_refused(client, monkeypatch):
    import main as main_module

    monkeypatch.setattr(main_module, "_discover_builtin_tasks", lambda: [])
    resp = _post(client, "huge", "task: huge\npadding: " + ("x" * 300_000))
    assert resp.status_code == 400
    assert "256" in resp.json()["detail"]


def test_write_requires_the_tasks_write_scope(client, monkeypatch):
    """A tasks:read ticket must not be able to register a task."""
    import main as main_module

    monkeypatch.setattr(main_module, "_discover_builtin_tasks", lambda: [])
    resp = _post(client, "my_custom_task", VALID_YAML, scope="tasks:read")
    assert resp.status_code == 403


def test_delete_removes_the_config_and_invalidates(client, monkeypatch):
    import main as main_module

    monkeypatch.setattr(main_module, "_discover_builtin_tasks", lambda: [])
    assert _post(client, "my_custom_task", VALID_YAML).status_code == 201

    monkeypatch.setattr(main_module, "_ALL_TASKS", ["stale"])
    resp = client.delete("/api/tasks/my_custom_task", headers=_hdr())
    assert resp.status_code == 200
    assert not (main_module.CUSTOM_TASKS_DIR / "my_custom_task.yaml").exists()
    assert main_module._ALL_TASKS == []


def test_delete_unknown_task_is_404(client):
    assert client.delete("/api/tasks/never_existed", headers=_hdr()).status_code == 404


def test_discovery_includes_custom_dir_only_when_present(client, monkeypatch, temp_workspace):
    """TaskManager is constructed with include_path only if the dir exists —
    passing a nonexistent path is not the same as passing none."""
    import main as main_module

    seen = {}

    class _TM:
        def __init__(self, include_path=None):
            seen["include_path"] = include_path
            self.all_tasks = ["hellaswag"]

    import types

    fake = types.ModuleType("language_eval.tasks")
    fake.TaskManager = _TM
    monkeypatch.setitem(sys.modules, "language_eval.tasks", fake)

    main_module._discover_tasks()
    assert seen["include_path"] is None

    main_module.CUSTOM_TASKS_DIR.mkdir(parents=True, exist_ok=True)
    main_module._discover_tasks()
    assert seen["include_path"] == str(main_module.CUSTOM_TASKS_DIR)
