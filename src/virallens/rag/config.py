"""LLM settings for the RAG pipeline, read from the environment and ``.env`` at startup.

Ingestion and retrieval need only ``Paths``; these settings add the provider credentials and the
model for each role. Model choices are explained in docs/DESIGN.md.
"""

from pydantic import Field, SecretStr

from virallens.common.settings import Paths


class Settings(Paths):
    # min_length rejects `GROQ_API_KEY=` left empty from .env.example at startup.
    groq_api_key: SecretStr = Field(min_length=1)
    # Optional: without it there is no fallback provider.
    openrouter_api_key: SecretStr | None = None

    # Groq meters each model separately, so the roles are spread over three models.
    groq_classifier_model: str = "qwen/qwen3.8-27b"  # router, grader, rewriter, verifier
    groq_answer_model: str = "openai/gpt-oss-120b"  # answerer, research planner
    groq_analyst_model: str = "openai/gpt-oss-20b"  # research analysts
    # A free OpenRouter model with tool calling and strict structured outputs.
    openrouter_model: str = "nvidia/nemotron-3-super-120b-a12b:free"

    # Published free-tier limits: console.groq.com/docs/rate-limits,
    # openrouter.ai/docs/api-reference/limits.
    groq_requests_per_minute: int = 30
    openrouter_requests_per_minute: int = 20


def load_settings() -> Settings:
    """Settings from the environment and ``.env``; a missing or empty key raises here."""
    # pydantic-settings fills the required fields from the environment, which pyright cannot see.
    return Settings()  # pyright: ignore[reportCallIssue]
