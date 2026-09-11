"""Regression tests for the implicit-runtime path in init_agent (#101916).

When a direct ``AIAgent`` caller omits both ``provider`` and ``model``
(notably ``python run_agent.py``), the ``auto`` provider router resolves
the configured main runtime correctly — but the implicit-runtime branch
in ``init_agent`` historically discarded the concrete model the router
returned, then sent ``model: ""`` to an otherwise valid endpoint.

These tests pin the fixed behaviors of ``_adopt_implicit_runtime``:

1. the router-resolved model is adopted onto ``agent.model``
2. the client's effective provider is adopted onto
   ``agent.provider`` / ``agent.requested_provider``
3. routing success without a model name raises a clear RuntimeError
   instead of a silent empty-model request
"""

# Upstream 0.21.7 added tests/home_io_guard.py, which refuses any file I/O
# against the REAL hermes home. On a default install the checkout lives at
# ~/.hermes/hermes-agent, so importing agent.agent_init (whose chain runs
# hermes_bootstrap's module-level activate_dependencies) reads the payload
# manifest at ~/.hermes/manifest.json — inside the guarded home — and every
# test below fails at import. CI checkouts have no sibling manifest.json, so
# only default installs trip this. Stub the manifest lookup before the first
# agent import: payload_venv then returns None and _require_own_dependencies
# returns early for a venv interpreter (./venv/bin/python), keeping the import
# chain fully inside the checkout (which the guard already exempts).
import pm.environments as _pm_environments

_pm_environments._payload_manifest = lambda root: None

import pytest
from unittest.mock import MagicMock, patch


class _FakeAgent:
    """Minimal AIAgent stand-in exposing the fields init_agent touches."""

    def __init__(self, provider=None, model=None):
        self.provider = provider
        self.model = model
        self.requested_provider = provider


class TestAdoptImplicitRuntime:
    """_adopt_implicit_runtime adopts router-resolved identity."""

    def test_routed_model_and_provider_adopted_when_caller_omitted(self):
        from agent.agent_init import _adopt_implicit_runtime

        agent = _FakeAgent(provider=None, model=None)
        routed = MagicMock()
        routed._hermes_aux_effective_provider = "custom"

        _adopt_implicit_runtime(agent, routed, "glm-5_3")

        assert agent.model == "glm-5_3"
        assert agent.provider == "custom"
        assert agent.requested_provider == "custom"

    def test_explicit_model_not_overwritten(self):
        from agent.agent_init import _adopt_implicit_runtime

        agent = _FakeAgent(provider=None, model="explicit-model")
        routed = MagicMock()
        routed._hermes_aux_effective_provider = "custom"

        _adopt_implicit_runtime(agent, routed, "glm-5_3")

        assert agent.model == "explicit-model"

    def test_explicit_provider_not_overwritten(self):
        from agent.agent_init import _adopt_implicit_runtime

        agent = _FakeAgent(provider="openrouter", model=None)
        routed = MagicMock()
        routed._hermes_aux_effective_provider = "custom"

        _adopt_implicit_runtime(agent, routed, "glm-5_3")

        assert agent.provider == "openrouter"
        assert agent.model == "glm-5_3"

    def test_router_success_without_model_raises(self):
        from agent.agent_init import _adopt_implicit_runtime

        agent = _FakeAgent(provider=None, model=None)
        routed = MagicMock()
        routed._hermes_aux_effective_provider = "custom"

        with pytest.raises(RuntimeError, match="did not supply a model name"):
            _adopt_implicit_runtime(agent, routed, "")

    def test_missing_effective_provider_attribute_keeps_none(self):
        """Routed clients without the marker attribute are fine — provider
        stays unset and only the model is adopted."""
        from agent.agent_init import _adopt_implicit_runtime

        agent = _FakeAgent(provider=None, model=None)
        routed = MagicMock(spec=[])  # no _hermes_aux_effective_provider

        _adopt_implicit_runtime(agent, routed, "glm-5_3")

        assert agent.model == "glm-5_3"
        assert agent.provider is None
