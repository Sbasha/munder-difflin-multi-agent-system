"""Runtime configuration with provider-registry model routing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


@dataclass(frozen=True)
class ProviderProfile:
    """One OpenAI-compatible endpoint the system knows how to reach."""

    name: str
    base_url: str
    key_prefixes: tuple[str, ...] = ()


PROVIDER_REGISTRY: dict[str, ProviderProfile] = {
    profile.name: profile
    for profile in (
        ProviderProfile("openai", "https://api.openai.com/v1"),
        ProviderProfile("vocareum", "https://openai.vocareum.com/v1", key_prefixes=("voc-",)),
    )
}
DEFAULT_PROVIDER_NAME = "openai"


class Settings(BaseSettings):
    """Validated application settings loaded from environment variables.

    Model routing resolves in one pass: an explicit ``LLM_BASE_URL`` wins,
    then ``LLM_PROVIDER`` names a registry entry, then the API key's prefix
    infers one, and the registry default is the last resort. New providers
    are registry entries, not code branches; after validation,
    ``llm_provider`` and ``llm_base_url`` always hold the resolved values.
    """

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    llm_api_key: str | None = None
    llm_provider: str | None = None
    llm_base_url: str | None = None
    llm_model: str = "gpt-5-mini"
    llm_request_limit: int = Field(default=30, ge=1)
    udacity_openai_api_key: str | None = None
    openai_api_key: str | None = None
    database_url: str = "sqlite:///munder_difflin.db"
    data_dir: Path = Field(default_factory=Path.cwd)
    markup_rate: float = Field(default=1.30, ge=1.0)
    cash_reserve: float = Field(default=5_000.0, ge=0)
    restock_target_multiplier: int = Field(default=3, ge=1)

    @model_validator(mode="after")
    def resolve_provider(self) -> Settings:
        """Select the API key, then resolve the provider and endpoint."""

        if not self.llm_api_key:
            self.llm_api_key = self.udacity_openai_api_key or self.openai_api_key
        if self.llm_base_url and not self.llm_provider:
            self.llm_provider = "custom"
            return self
        profile = self._named_or_inferred_profile()
        self.llm_provider = profile.name
        if not self.llm_base_url:
            self.llm_base_url = profile.base_url
        return self

    def _named_or_inferred_profile(self) -> ProviderProfile:
        if self.llm_provider:
            name = self.llm_provider.lower()
            if name not in PROVIDER_REGISTRY:
                known = ", ".join(sorted(PROVIDER_REGISTRY))
                raise ValueError(
                    f"Unknown LLM_PROVIDER '{self.llm_provider}'; known providers: {known}"
                )
            return PROVIDER_REGISTRY[name]
        if self.llm_api_key:
            for profile in PROVIDER_REGISTRY.values():
                if any(self.llm_api_key.startswith(prefix) for prefix in profile.key_prefixes):
                    return profile
        return PROVIDER_REGISTRY[DEFAULT_PROVIDER_NAME]

    @property
    def live_model_enabled(self) -> bool:
        """Return whether enough configuration exists for live LLM calls."""

        return bool(self.llm_api_key)
