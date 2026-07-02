import os
from dataclasses import dataclass

from dotenv import load_dotenv


load_dotenv()


class MissingEnvironmentError(ValueError):
    """Raised when a required environment variable is missing."""


def _parse_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _require_env(var_names: list[str]) -> dict[str, str]:
    missing = [name for name in var_names if not os.getenv(name)]
    if missing:
        raise MissingEnvironmentError(
            "Missing required environment variables: " + ", ".join(missing)
        )
    return {name: os.getenv(name, "") for name in var_names}


@dataclass(frozen=True)
class LLMSettings:
    provider: str
    model_id: str
    aws_region: str | None = None
    lm_studio_endpoint: str | None = None
    lm_studio_api_key: str | None = None


@dataclass(frozen=True)
class DeviceSettings:
    device_type: str
    host: str
    port: int
    username: str
    password: str


@dataclass(frozen=True)
class BatfishSettings:
    host: str


@dataclass(frozen=True)
class MemorySettings:
    path: str
    auto_refresh: bool
    enable_topology: bool


def get_llm_settings() -> LLMSettings:
    provider = os.getenv("LLM_PROVIDER", "bedrock").strip().lower()

    if provider == "bedrock":
        values = _require_env([
            "LLM_MODEL_ID",
            "AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY",
            "AWS_DEFAULT_REGION",
        ])
        return LLMSettings(
            provider="bedrock",
            model_id=values["LLM_MODEL_ID"],
            aws_region=values["AWS_DEFAULT_REGION"],
        )

    if provider == "lmstudio":
        values = _require_env([
            "LLM_MODEL_ID",
            "LM_STUDIO_ENDPOINT",
        ])
        return LLMSettings(
            provider="lmstudio",
            model_id=values["LLM_MODEL_ID"],
            lm_studio_endpoint=values["LM_STUDIO_ENDPOINT"].rstrip("/"),
            lm_studio_api_key=os.getenv("LM_STUDIO_API_KEY", "lm-studio"),
        )

    raise MissingEnvironmentError(
        f"Unsupported LLM_PROVIDER '{provider}'. Supported values: bedrock, lmstudio"
    )


def get_device_settings() -> DeviceSettings:
    values = _require_env([
        "DEVICE_HOST",
        "DEVICE_PORT",
        "DEVICE_USERNAME",
        "DEVICE_PASSWORD",
    ])
    return DeviceSettings(
        device_type=os.getenv("DEVICE_TYPE", "cisco_ios"),
        host=values["DEVICE_HOST"],
        port=int(values["DEVICE_PORT"]),
        username=values["DEVICE_USERNAME"],
        password=values["DEVICE_PASSWORD"],
    )


def get_batfish_settings() -> BatfishSettings:
    return BatfishSettings(host=os.getenv("BATFISH_HOST", "127.0.0.1"))


def get_memory_settings() -> MemorySettings:
    return MemorySettings(
        path=os.getenv("NETWORK_MEMORY_PATH", "memory/network_memory.json"),
        auto_refresh=_parse_bool(os.getenv("NETWORK_MEMORY_AUTO_REFRESH"), default=True),
        enable_topology=_parse_bool(os.getenv("NETWORK_MEMORY_ENABLE_TOPOLOGY"), default=True),
    )
