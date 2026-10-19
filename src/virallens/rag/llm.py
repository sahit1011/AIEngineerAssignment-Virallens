"""Chat models for the agent roles, and the policy every Groq call runs under.

Rate limits: each model gets an ``InMemoryRateLimiter`` at its provider's requests-per-minute, so a
burst of graph steps queues instead of tripping a 429. Groq's token cap (8000/min per model), which
a request limiter cannot see, is met by spreading roles over three separately metered models and,
when it is still hit, by the policy below.

The call policy (``call_with_policy``), shared by graph nodes and deep agents:

- invalid model output (``INVALID_OUTPUT_CODES``) is retried once on Groq;
- the Groq SDK first retries 429s and 5xx itself (``GROQ_MAX_RETRIES``), so a per-minute rate limit
  is waited out on Groq;
- a rate limit, rejected key, server or connection error that outlasts those retries
  (``FALLBACK_ERRORS``), or invalid output that persists, moves the call to OpenRouter when a key is
  configured, and raises otherwise;
- anything else raises.
"""

import logging
from collections.abc import Callable
from typing import Any, cast

import groq
from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langchain_core.language_models import BaseChatModel, LanguageModelInput
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_core.runnables import Runnable
from langchain_groq import ChatGroq
from langchain_openrouter import ChatOpenRouter
from pydantic import BaseModel

from virallens.rag.config import Settings

logger = logging.getLogger(__name__)

# Groq unavailable for this call: rate limited (429), key rejected (401), server error (5xx), or
# unreachable. With an OpenRouter key these move the call there; a malformed request still raises.
FALLBACK_ERRORS: tuple[type[Exception], ...] = (
    groq.RateLimitError,
    groq.AuthenticationError,
    groq.InternalServerError,
    groq.APIConnectionError,
)
# 400s where the model's output, not the request, is at fault: an empty generation under strict
# JSON, or a call to a tool that does not exist. Repeating the request usually succeeds.
INVALID_OUTPUT_CODES = frozenset({"json_validate_failed", "tool_use_failed"})
# The Groq SDK waits the `retry-after` it reports when that is at most 60 s (a per-minute limit) and
# backs off briefly otherwise, so short limits never reach the fallback, whose free tier allows only
# 50 requests a day; a daily cap or an outage does, once these retries are spent.
GROQ_MAX_RETRIES = 2
# Reasoning tokens count against Groq's per-minute token cap, so each family runs at its lowest
# setting. Keys are model-id prefixes; a model from another family is rejected at startup.
REASONING_EFFORT = {"openai/gpt-oss-": "low", "qwen/": "none"}


class Models:
    """The chat model for each role. ``fallback`` is None when no OpenRouter key is set."""

    def __init__(self, settings: Settings):
        self.fallback = _openrouter(settings) if settings.openrouter_api_key else None
        self.classifier = _groq(settings, settings.groq_classifier_model)
        self.answerer = _groq(settings, settings.groq_answer_model)
        self.analyst = _groq(settings, settings.groq_analyst_model)
        if self.fallback is None:
            logger.warning("OPENROUTER_API_KEY not set: no fallback if Groq is unavailable")

    def structured[T: BaseModel](
        self, model: BaseChatModel, schema: type[T]
    ) -> "StructuredCall[T]":
        # Strict mode has the provider enforce the schema; the looser modes failed on gpt-oss.
        primary = model.with_structured_output(schema, method="json_schema", strict=True)
        backup = (
            None
            if self.fallback is None
            else self.fallback.with_structured_output(schema, method="json_schema", strict=True)
        )
        return StructuredCall(primary, backup)

    def agent_middleware(self) -> list[AgentMiddleware]:
        """Middleware running a deep agent's model calls under the same policy."""
        return [GroqCallPolicy(self.fallback)]


class StructuredCall[T: BaseModel]:
    """A structured-output call on Groq, run under ``call_with_policy``, returning a ``T``."""

    def __init__(
        self,
        primary: Runnable[LanguageModelInput, Any],
        fallback: Runnable[LanguageModelInput, Any] | None,
    ):
        self._primary = primary
        self._fallback = fallback

    def invoke(self, messages: LanguageModelInput) -> T:
        fallback = self._fallback
        backup = None if fallback is None else (lambda: fallback.invoke(messages))
        # with_structured_output(schema) returns `schema` instances; its annotation is untyped.
        return cast(T, call_with_policy(lambda: self._primary.invoke(messages), backup))


class GroqCallPolicy(AgentMiddleware):
    """Run each deep-agent model call under ``call_with_policy``.

    ``create_deep_agent`` binds tools to its model, which a ``RunnableWithFallbacks`` does not
    support, and langchain's ``ModelFallbackMiddleware`` falls back on every exception.
    """

    def __init__(self, fallback: BaseChatModel | None):
        super().__init__()
        self._fallback = fallback

    def wrap_model_call(
        self, request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]
    ) -> ModelResponse:
        fallback = self._fallback
        backup = None if fallback is None else (lambda: handler(request.override(model=fallback)))
        return call_with_policy(lambda: handler(request), backup)


def call_with_policy[T](primary: Callable[[], T], fallback: Callable[[], T] | None) -> T:
    try:
        return _retry_invalid_output_once(primary)
    except FALLBACK_ERRORS as error:
        return _fall_back(error, fallback)
    except groq.BadRequestError as error:
        if not is_invalid_output(error):
            raise
        return _fall_back(error, fallback)


def _fall_back[T](error: Exception, fallback: Callable[[], T] | None) -> T:
    # Called inside the handler, so a fallback failure is chained to the Groq error that caused it.
    if fallback is None:
        raise error
    logger.warning("Groq call failed (%s); calling OpenRouter", _describe(error))
    return fallback()


def _retry_invalid_output_once[T](call: Callable[[], T]) -> T:
    try:
        return call()
    except groq.BadRequestError as error:
        if not is_invalid_output(error):
            raise
        logger.warning("Groq returned invalid output (%s); retrying once", _describe(error))
        return call()


def is_invalid_output(error: groq.BadRequestError) -> bool:
    return error_code(error) in INVALID_OUTPUT_CODES


def error_code(error: groq.APIStatusError) -> str | None:
    # Groq's error body is {"error": {"message": ..., "type": ..., "code": ...}}; any other shape is
    # not a recognised invalid-output error and is left to propagate.
    body = error.body
    if isinstance(body, dict) and isinstance(body.get("error"), dict):
        return body["error"].get("code")
    return None


def _describe(error: Exception) -> str:
    if isinstance(error, groq.APIStatusError):
        return f"{type(error).__name__} {error.status_code} {error_code(error)}"
    return type(error).__name__


def _groq(settings: Settings, model: str) -> ChatGroq:
    return ChatGroq(
        model=model,
        api_key=settings.groq_api_key,
        temperature=0,
        reasoning_effort=reasoning_effort(model),
        max_retries=GROQ_MAX_RETRIES,
        rate_limiter=_per_minute(settings.groq_requests_per_minute),
    )


def _openrouter(settings: Settings) -> ChatOpenRouter:
    return ChatOpenRouter(
        model=settings.openrouter_model,
        api_key=settings.openrouter_api_key,
        temperature=0,
        rate_limiter=_per_minute(settings.openrouter_requests_per_minute),
    )


def reasoning_effort(model: str) -> str:
    for prefix, effort in REASONING_EFFORT.items():
        if model.startswith(prefix):
            return effort
    raise ValueError(
        f"no reasoning setting for model {model!r}; known families: {list(REASONING_EFFORT)}"
    )


def _per_minute(requests: int) -> InMemoryRateLimiter:
    return InMemoryRateLimiter(requests_per_second=requests / 60)
