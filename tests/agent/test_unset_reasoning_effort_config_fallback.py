"""A caller that never applied the CLI's effort resolution still honours config.yaml.

``python run_agent.py`` builds ``AIAgent`` directly, so ``agent.reasoning_effort`` /
``agent.reasoning_config`` stay unset while config.yaml names a level. The unset-effort default
(``agent.reasoning_params.unset_reasoning_default``) then answered the custom profile's
``medium`` — a level a restricted OpenAI-compatible route rejects with HTTP 400 (its vocabulary
is only low/high/xhigh/max/int), so the first request of every such session was lost. The
configured level wins instead, read through the same chokepoint every other surface uses.
"""
from unittest.mock import patch

import pytest

from agent.chat_completion_helpers import _build_api_kwargs_for_mode
from agent.transports.chat_completions import ChatCompletionsTransport
from providers import get_provider_profile

_CONFIG = "hermes_cli.config.load_config_readonly"


class _Agent:
    """The attributes ``_build_api_kwargs_for_mode`` reads before handing off to the transport builder."""

    def __init__(self, reasoning_config, *, provider="custom:relay", api_mode="chat_completions"):
        self.reasoning_config = reasoning_config
        self.provider = provider
        self.model = "moonshotai/kimi-k3"
        self.api_mode = api_mode
        self.base_url = "http://relay.example/v1"
        self.tools = []
        self.request_overrides = None
        self.service_tier = None
        self._fast_until = 0.0
        self._ephemeral_reasoning_off = False
        self._reasoning_effort_rejected = False
        self._reasoning_disable_rejected = False
        self._ollama_num_ctx: int | None = None

    def __getattr__(self, name):
        return lambda *args, **kwargs: None


def _wire_reasoning_config(agent):
    """The ``reasoning_config`` the main loop hands the transport for this agent's next request."""
    def _capture(agent, api_messages, tools_for_api, reasoning_config, request_overrides, cache_scope_id):
        return {"reasoning_config": reasoning_config}

    with patch("agent.chat_completion_helpers._build_chat_completions_kwargs", _capture):
        return _build_api_kwargs_for_mode(agent, [], [])["reasoning_config"]


def _custom_wire_field(reasoning_config):
    kwargs = ChatCompletionsTransport().build_kwargs(
        "moonshotai/kimi-k3", [{"role": "user", "content": "hi"}], tools=None,
        provider_profile=get_provider_profile("custom:relay"), reasoning_config=reasoning_config,
        base_url="http://relay.example/v1",
    )
    return kwargs.get("reasoning_effort")


def _sent_wire_field(cfg):
    """Full path: unset agent effort + this config.yaml → the effort the custom route receives."""
    with patch(_CONFIG, return_value=cfg), patch("agent.models_dev.get_model_capabilities", return_value=None):
        return _custom_wire_field(_wire_reasoning_config(_Agent(None)))


def test_configured_global_effort_beats_the_profile_default():
    # Before: the profile default answered "medium" here, which this route 400s on.
    assert _sent_wire_field({"agent": {"reasoning_effort": "xhigh"}}) == "xhigh"


def test_per_model_override_beats_the_global_effort():
    cfg = {"agent": {"reasoning_effort": "xhigh", "reasoning_overrides": {"moonshotai/kimi-k3": "low"}}}
    assert _sent_wire_field(cfg) == "low"


def test_a_configured_disable_stays_a_disable():
    assert _sent_wire_field({"agent": {"reasoning_effort": False}}) == "none"


@pytest.mark.parametrize("cfg", [{}, {"agent": {}}, {"agent": {"reasoning_effort": ""}}, {"agent": {"reasoning_effort": "hgih"}}])
def test_without_a_usable_configured_level_the_profile_default_stands(cfg):
    assert _sent_wire_field(cfg) == "medium"


def test_an_unreadable_config_falls_back_to_the_profile_default():
    with patch(_CONFIG, side_effect=RuntimeError("no home")), patch(
        "agent.models_dev.get_model_capabilities", return_value=None
    ):
        assert _custom_wire_field(_wire_reasoning_config(_Agent(None))) == "medium"


def test_an_explicit_agent_config_is_not_overridden_by_config_yaml():
    # Every surface that already applied the CLI resolution keeps sending its own value.
    with patch(_CONFIG, return_value={"agent": {"reasoning_effort": "xhigh"}}):
        sent = _wire_reasoning_config(_Agent({"enabled": True, "effort": "low"}))
    assert _custom_wire_field(sent) == "low"
