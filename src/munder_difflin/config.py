"""Runtime configuration with Vocareum-compatible provider defaults."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Validated application settings loaded from environment variables."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    llm_api_key: str | None = None
    llm_base_url: str | None = None
    llm_model: str = "gpt-5-mini"
    udacity_openai_api_key: str | None = None
    openai_api_key: str | None = None
    database_url: str = "sqlite:///munder_difflin.db"
    data_dir: Path = Field(default_factory=Path.cwd)
    markup_rate: float = Field(default=1.30, ge=1.0)
    cash_reserve: float = Field(default=5_000.0, ge=0)
    restock_target_multiplier: int = Field(default=3, ge=1)

    @model_validator(mode="after")
    def apply_provider_defaults(self) -> Settings:
        """Select a key from the accepted variables and route Vocareum keys."""

        if not self.llm_api_key:
            self.llm_api_key = self.udacity_openai_api_key or self.openai_api_key
        if not self.llm_base_url:
            if self.llm_api_key and self.llm_api_key.startswith("voc-"):
                self.llm_base_url = "https://openai.vocareum.com/v1"
            else:
                self.llm_base_url = "https://api.openai.com/v1"
        return self

    @property
    def live_model_enabled(self) -> bool:
        """Return whether enough configuration exists for live LLM calls."""

        return bool(self.llm_api_key)
