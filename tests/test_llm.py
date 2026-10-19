from dataclasses import dataclass, replace

import groq
import httpx
import pytest
from langchain_openrouter import ChatOpenRouter
from pydantic import BaseModel

from virallens.rag.config import Settings
from virallens.rag.llm import (
    GROQ_MAX_RETRIES,
    GroqCallPolicy,
    Models,
    StructuredCall,
    call_with_policy,
    reasoning_effort,
)


class _Verdict(BaseModel):
    supported: bool


_REQUEST = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")


def _status(cls: type[groq.APIStatusError], status: int, code: str | None = None):
    # Groq's real error body shape: {"error": {"message": ..., "type": ..., "code": ...}}.
    body = {"error": {"message": "scripted", "code": code}} if code else None
    return cls("scripted", response=httpx.Response(status, request=_REQUEST), body=body)


INVALID_JSON = _status(groq.BadRequestError, 400, "json_validate_failed")
PHANTOM_TOOL = _status(groq.BadRequestError, 400, "tool_use_failed")
UNAVAILABLE = [
    _status(groq.RateLimitError, 429),
    _status(groq.AuthenticationError, 401),
    _status(groq.InternalServerError, 503),
    groq.APIConnectionError(request=_REQUEST),
]


class _Scripted:
    """A call that raises the scripted errors in turn, then answers."""

    def __init__(self, *errors: Exception, answer: str = "groq"):
        self._errors = list(errors)
        self._answer = answer
        self.calls = 0

    def __call__(self):
        self.calls += 1
        if self._errors:
            raise self._errors.pop(0)
        return self._answer


@pytest.mark.parametrize("error", [INVALID_JSON, PHANTOM_TOOL])
def test_invalid_output_is_retried_once_on_groq(error):
    primary, fallback = _Scripted(error), _Scripted(answer="openrouter")
    assert call_with_policy(primary, fallback) == "groq"
    assert (primary.calls, fallback.calls) == (2, 0)


def test_persistent_invalid_output_moves_to_the_fallback():
    primary, fallback = _Scripted(INVALID_JSON, PHANTOM_TOOL), _Scripted(answer="openrouter")
    assert call_with_policy(primary, fallback) == "openrouter"
    assert primary.calls == 2


def test_persistent_invalid_output_raises_without_a_fallback():
    primary = _Scripted(INVALID_JSON, INVALID_JSON)
    with pytest.raises(groq.BadRequestError):
        call_with_policy(primary, None)
    assert primary.calls == 2


def test_other_bad_requests_raise_at_once():
    primary, fallback = (
        _Scripted(_status(groq.BadRequestError, 400, "context_length_exceeded")),
        _Scripted(),
    )
    with pytest.raises(groq.BadRequestError):
        call_with_policy(primary, fallback)
    assert (primary.calls, fallback.calls) == (1, 0)


@pytest.mark.parametrize("error", UNAVAILABLE, ids=lambda e: type(e).__name__)
def test_unavailable_groq_moves_straight_to_the_fallback(error):
    primary, fallback = _Scripted(error), _Scripted(answer="openrouter")
    assert call_with_policy(primary, fallback) == "openrouter"
    assert primary.calls == 1


@pytest.mark.parametrize("error", UNAVAILABLE, ids=lambda e: type(e).__name__)
def test_unavailable_groq_raises_without_a_fallback(error):
    with pytest.raises(type(error)):
        call_with_policy(_Scripted(error), None)


class _Invocable:
    def __init__(self, call):
        self.invoke = lambda messages: call()


def test_structured_call_runs_under_the_policy():
    call = StructuredCall(_Invocable(_Scripted(UNAVAILABLE[0])), _Invocable(_Scripted(answer="or")))
    assert call.invoke([]) == "or"


@dataclass(frozen=True)
class _Request:
    """Stands in for langchain's ModelRequest: the middleware only uses ``override(model=...)``."""

    model: str

    def override(self, model):
        return replace(self, model=model)


def test_agent_middleware_sends_the_fallback_through_the_request():
    seen: list[str] = []

    def handler(request):
        seen.append(request.model)
        if request.model == "groq":
            raise UNAVAILABLE[0]
        return f"answered by {request.model}"

    response = GroqCallPolicy(fallback="openrouter").wrap_model_call(_Request("groq"), handler)
    assert response == "answered by openrouter"
    assert seen == ["groq", "openrouter"]


@pytest.fixture(autouse=True)
def _no_provider_keys_from_the_shell(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)


def _settings(**overrides) -> Settings:
    return Settings(_env_file=None, groq_api_key="gsk-test", **overrides)


def test_structured_calls_are_built_under_the_policy():
    models = Models(_settings(openrouter_api_key="sk-or-test"))
    assert isinstance(models.structured(models.classifier, _Verdict), StructuredCall)


@pytest.mark.parametrize("openrouter_key", [None, "sk-or-test"])
def test_groq_retries_short_limits_itself_with_or_without_a_fallback(openrouter_key):
    models = Models(_settings(openrouter_api_key=openrouter_key))
    roles = (models.classifier, models.answerer, models.analyst)
    assert {m.max_retries for m in roles} == {GROQ_MAX_RETRIES} == {2}
    assert [type(m) for m in models.agent_middleware()] == [GroqCallPolicy]


def test_an_openrouter_key_configures_the_fallback():
    assert Models(_settings()).fallback is None
    assert isinstance(Models(_settings(openrouter_api_key="sk-or-test")).fallback, ChatOpenRouter)


def test_missing_groq_key_fails_at_startup():
    with pytest.raises(ValueError, match="groq_api_key"):
        Settings(_env_file=None)


def test_empty_groq_key_fails_at_startup():
    with pytest.raises(ValueError, match="groq_api_key"):
        Settings(_env_file=None, groq_api_key="")


@pytest.mark.parametrize(
    ("model", "effort"),
    [("openai/gpt-oss-120b", "low"), ("openai/gpt-oss-20b", "low"), ("qwen/qwen3.8-27b", "none")],
)
def test_each_model_family_runs_at_its_lowest_reasoning_setting(model, effort):
    assert reasoning_effort(model) == effort


def test_a_model_from_an_unknown_family_is_rejected_at_startup():
    with pytest.raises(ValueError, match="no reasoning setting for model 'acme/large'"):
        Models(_settings(groq_answer_model="acme/large"))
