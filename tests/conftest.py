"""Shared fixtures: isolated databases and scripted offline models."""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pydantic_ai.models
import pytest

from munder_difflin.config import Settings

ScriptedModelFactory = Callable[..., "pydantic_ai.models.function.FunctionModel"]


@pytest.fixture(autouse=True)
def _block_model_requests(monkeypatch: pytest.MonkeyPatch) -> None:
    """Hard-block accidental network calls to real model providers."""

    monkeypatch.setattr(pydantic_ai.models, "ALLOW_MODEL_REQUESTS", False)


@pytest.fixture
def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.fixture
def settings(tmp_path: Path, project_root: Path) -> Settings:
    return Settings(
        llm_api_key="test-key",
        database_url=f"sqlite:///{(tmp_path / 'test.db').as_posix()}",
        data_dir=project_root / "data",
        cash_reserve=5_000,
        markup_rate=1.3,
    )


@pytest.fixture
def scripted_model() -> ScriptedModelFactory:
    """Build a FunctionModel that replays a fixed tool-call script per run.

    The script position is derived from the number of tool returns in the
    current run's message history, so the same model can serve several
    sequential agent runs.
    """

    from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart
    from pydantic_ai.models.function import AgentInfo, FunctionModel

    def factory(
        script: list[tuple[str, dict[str, Any]]],
        final_text: str = "Done.",
    ) -> FunctionModel:
        def run(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            del info
            completed_calls = sum(
                1
                for message in messages
                for part in getattr(message, "parts", [])
                if getattr(part, "part_kind", "") == "tool-return"
            )
            if completed_calls < len(script):
                tool_name, args = script[completed_calls]
                return ModelResponse(parts=[ToolCallPart(tool_name=tool_name, args=args)])
            return ModelResponse(parts=[TextPart(final_text)])

        return FunctionModel(run)

    return factory
