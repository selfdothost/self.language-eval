"""
Fixtures for self.language-eval's control-plane (api/) tests.

self.ai#25: adds an isolated workspace fixture + a service-ticket minting
helper so tests/api/test_auth.py can exercise api/auth.py's require_scope()
without touching a real /workspace directory or the real
selfai-service-auth secret.

Test-only HMAC secret for the service-ticket auth layer. Never used outside
pytest — real deployments get SERVICE_AUTH_SECRET from the
selfai-service-auth ExternalSecret. Mirrors self.llamolotl's
api/tests/conftest.py (self.llamolotl#12) and self.code-eval's
tests/api/conftest.py (self.ai#25, the twin leg) — keep all three in sync.
"""

import sys
import time
from pathlib import Path

import jwt
import pytest

_API_DIR = str(Path(__file__).resolve().parents[2] / "api")
if _API_DIR not in sys.path:
    sys.path.insert(0, _API_DIR)

TEST_SERVICE_AUTH_SECRET = "pytest-only-service-auth-secret"
TEST_SERVICE_AUTH_AUDIENCE = "self.language-eval"

# Every scope this API currently gates, so a default all-access ticket can
# hit any endpoint without individual tests needing to know about scopes —
# scope enforcement itself is covered separately in test_auth.py.
ALL_SCOPES = "tasks:read jobs:read jobs:create jobs:write"


def mint_test_ticket(
    scope=ALL_SCOPES,
    audience=TEST_SERVICE_AUTH_AUDIENCE,
    secret=TEST_SERVICE_AUTH_SECRET,
    ttl_seconds=3600,
    **extra_claims,
):
    """Mint a service ticket signed with the pytest test secret. Mirrors
    self.ai's minting side (api/selfai_ui/utils/service_auth.py) closely
    enough to exercise the same validation path as production."""
    now = int(time.time())
    payload = {
        "iss": "self.ai",
        "aud": audience,
        "scope": scope,
        "iat": now,
        "exp": now + ttl_seconds,
    }
    payload.update(extra_claims)
    return jwt.encode(payload, secret, algorithm="HS256")


@pytest.fixture
def temp_workspace(tmp_path, monkeypatch):
    """Function-scoped isolated workspace so tests never touch a real
    /workspace directory. Mirrors self.curator's/self.code-eval's
    temp_workspace fixtures."""
    import main as main_module

    results_dir = tmp_path / "results"
    logs_dir = tmp_path / "logs"
    results_dir.mkdir(parents=True)
    logs_dir.mkdir(parents=True)

    monkeypatch.setattr(main_module, "RESULTS_DIR", results_dir)
    monkeypatch.setattr(main_module, "LOGS_DIR", logs_dir)
    monkeypatch.setattr(main_module, "JOBS_STATE_FILE", results_dir / ".jobs.json")
    monkeypatch.setattr(main_module, "_jobs", {})
    monkeypatch.setattr(main_module, "_processes", {})

    yield tmp_path
