"""Provider key selection and registry routing tests."""

import pytest
from pydantic import ValidationError

from munder_difflin.config import Settings

_PROVIDER_VARIABLES = (
    "LLM_API_KEY",
    "LLM_BASE_URL",
    "LLM_MODEL",
    "LLM_PROVIDER",
    "OPENAI_API_KEY",
    "UDACITY_OPENAI_API_KEY",
)


@pytest.fixture(autouse=True)
def _clean_provider_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in _PROVIDER_VARIABLES:
        monkeypatch.delenv(name, raising=False)


def _settings(**overrides) -> Settings:
    return Settings(_env_file=None, **overrides)


def test_key_prefix_infers_the_provider() -> None:
    settings = _settings(llm_api_key="voc-abc123")

    assert settings.llm_provider == "vocareum"
    assert settings.llm_base_url == "https://openai.vocareum.com/v1"


def test_unrecognized_keys_fall_back_to_the_registry_default() -> None:
    settings = _settings(llm_api_key="sk-abc123")

    assert settings.llm_provider == "openai"
    assert settings.llm_base_url == "https://api.openai.com/v1"


def test_named_provider_overrides_key_inference() -> None:
    settings = _settings(llm_api_key="sk-abc123", llm_provider="Vocareum")

    assert settings.llm_provider == "vocareum"
    assert settings.llm_base_url == "https://openai.vocareum.com/v1"


def test_unknown_provider_name_is_rejected() -> None:
    with pytest.raises(ValidationError, match="known providers"):
        _settings(llm_api_key="sk-abc123", llm_provider="nonsense")


def test_explicit_base_url_wins_and_labels_the_provider_custom() -> None:
    settings = _settings(llm_api_key="voc-abc123", llm_base_url="http://localhost:8080/v1")

    assert settings.llm_provider == "custom"
    assert settings.llm_base_url == "http://localhost:8080/v1"


def test_explicit_base_url_with_named_provider_keeps_both() -> None:
    settings = _settings(
        llm_api_key="sk-abc123",
        llm_provider="vocareum",
        llm_base_url="http://localhost:8080/v1",
    )

    assert settings.llm_provider == "vocareum"
    assert settings.llm_base_url == "http://localhost:8080/v1"


def test_key_selection_precedence() -> None:
    both = _settings(udacity_openai_api_key="voc-udacity", openai_api_key="sk-openai")
    explicit = _settings(
        llm_api_key="sk-explicit",
        udacity_openai_api_key="voc-udacity",
        openai_api_key="sk-openai",
    )

    assert both.llm_api_key == "voc-udacity"
    assert explicit.llm_api_key == "sk-explicit"


def test_live_model_disabled_without_any_key() -> None:
    assert _settings().live_model_enabled is False
