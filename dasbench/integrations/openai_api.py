from __future__ import annotations

import os
from dataclasses import asdict, dataclass

from dotenv import load_dotenv
from openai import OpenAI

DEFAULT_MODEL = "gpt-5.2"
DEFAULT_REASONING_EFFORT = "xhigh"


@dataclass(frozen=True)
class OpenAIAPIConfig:
    api_key: str
    model: str = DEFAULT_MODEL
    reasoning_effort: str = DEFAULT_REASONING_EFFORT
    base_url: str | None = None
    organization: str | None = None
    project: str | None = None

    def public_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload.pop("api_key", None)
        return payload


def load_openai_dotenv() -> bool:
    load_dotenv()
    return True


def load_openai_api_config(*, required: bool = True) -> OpenAIAPIConfig | None:
    load_openai_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL", DEFAULT_MODEL)
    reasoning_effort = os.getenv("OPENAI_REASONING_EFFORT", DEFAULT_REASONING_EFFORT)
    base_url = os.getenv("OPENAI_BASE_URL")
    organization = os.getenv("OPENAI_ORG_ID") or os.getenv("OPENAI_ORGANIZATION")
    project = os.getenv("OPENAI_PROJECT_ID")
    if not api_key:
        if not required:
            return None
        raise RuntimeError(
            "Missing OpenAI API configuration. Set `OPENAI_API_KEY` in `.env` "
            "or the environment. Set `OPENAI_MODEL` to override the default model."
        )
    return OpenAIAPIConfig(
        api_key=api_key,
        model=model,
        reasoning_effort=reasoning_effort,
        base_url=base_url,
        organization=organization,
        project=project,
    )


def openai_api_is_configured() -> bool:
    return load_openai_api_config(required=False) is not None


def build_openai_client(config: OpenAIAPIConfig) -> OpenAI:
    return OpenAI(
        api_key=config.api_key,
        base_url=config.base_url,
        organization=config.organization,
        project=config.project,
    )
