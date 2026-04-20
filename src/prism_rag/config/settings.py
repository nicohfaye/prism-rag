from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)

Provider = Literal["openai", "ollama"]


class EmbeddingConfig(BaseModel):
    provider: Provider = "openai"
    model: str = "text-embedding-3-small"
    api_key: str | None = None
    # Ollama host; when None, the ollama client uses http://localhost:11434.
    base_url: str | None = None


class GenerationConfig(BaseModel):
    provider: Provider = "openai"
    model: str = "gpt-4o-mini"
    api_key: str | None = None
    # Same as EmbeddingConfig.base_url — lets embeddings and generation target
    # different Ollama hosts if needed (e.g., GPU box vs local laptop).
    base_url: str | None = None


class MilvusConfig(BaseModel):
    uri: str = "http://localhost:19530"
    default_collection: str = "prism_default"


class ChunkingConfig(BaseModel):
    chunk_size: int = 512
    chunk_overlap: int = 52


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="PRISM_",
        env_nested_delimiter="__",
        extra="ignore",
    )

    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    milvus: MilvusConfig = Field(default_factory=MilvusConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)


class _YamlFileSource(PydanticBaseSettingsSource):
    """Loads settings from a YAML file; injected at lowest priority."""

    def __init__(self, settings_cls: type[BaseSettings], path: Path) -> None:
        super().__init__(settings_cls)
        self._data: dict[str, Any] = {}
        if path.exists():
            with path.open() as f:
                self._data = yaml.safe_load(f) or {}

    def get_field_value(self, field: Any, field_name: str) -> tuple[Any, str, bool]:
        value = self._data.get(field_name)
        return value, field_name, value is not None

    def __call__(self) -> dict[str, Any]:
        return self._data


def load_settings(profile: str = "openai", configs_dir: Path | str = "configs") -> Settings:
    """Load settings with precedence: init > env > dotenv > YAML profile > secrets."""
    yaml_path = Path(configs_dir) / f"{profile}.yaml"

    class _ProfileSettings(Settings):
        @classmethod
        def settings_customise_sources(
            cls,
            settings_cls: type[BaseSettings],
            init_settings: PydanticBaseSettingsSource,
            env_settings: PydanticBaseSettingsSource,
            dotenv_settings: PydanticBaseSettingsSource,
            file_secret_settings: PydanticBaseSettingsSource,
        ) -> tuple[PydanticBaseSettingsSource, ...]:
            return (
                init_settings,
                env_settings,
                dotenv_settings,
                _YamlFileSource(settings_cls, yaml_path),
                file_secret_settings,
            )

    return _ProfileSettings()
