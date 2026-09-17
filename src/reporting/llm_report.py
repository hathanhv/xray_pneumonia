"""Optional LLM wording layer for validated, pneumonia-focused report facts."""

from __future__ import annotations

import json
import re
from typing import Any, Callable, Mapping
from urllib.request import Request, urlopen


PROMPT_VERSION = "pneumonia-report-v1"
ALLOWED_FIELDS = {
    "findings_text",
    "impression_text",
    "recommendation_text",
    "limitations_text",
}
_NUMBER_PATTERN = re.compile(r"\b\d+(?:\.\d+)?%?")


def build_llm_prompt(facts: Mapping[str, Any]) -> str:
    """Create a strict prompt whose input contains facts, not raw model output."""
    return (
        "You are a medical report wording assistant. Return JSON only.\n"
        "Use only the supplied facts. Do not add diagnoses, findings, measurements, "
        "probabilities, recommendations, or patient details. Do not treat estimated "
        "localization as a ground-truth lesion mask. Keep pneumonia as the primary "
        "assessment and mention additional findings briefly.\n"
        "Allowed JSON keys are exactly: findings_text, impression_text, "
        "recommendation_text, limitations_text. Values must be strings.\n"
        f"FACTS:\n{json.dumps(facts, ensure_ascii=False, indent=2)}"
    )


def _allowed_numbers(facts: Mapping[str, Any]) -> set[str]:
    encoded = json.dumps(facts, ensure_ascii=False)
    allowed = set(_NUMBER_PATTERN.findall(encoded))
    for token in tuple(allowed):
        if token.endswith("%"):
            continue
        try:
            value = float(token)
        except ValueError:
            continue
        if 0 <= value <= 1:
            allowed.add(f"{value:.0%}")
    return allowed


def _validate_text(text: str, facts: Mapping[str, Any]) -> None:
    if not isinstance(text, str):
        raise ValueError("LLM report fields must be strings")
    unknown_numbers = set(_NUMBER_PATTERN.findall(text)) - _allowed_numbers(facts)
    if unknown_numbers:
        raise ValueError(
            f"LLM output contains unsupported numeric facts: {sorted(unknown_numbers)}"
        )


def validate_llm_draft(draft: Mapping[str, Any], facts: Mapping[str, Any]) -> dict[str, str]:
    """Reject unstructured or fact-changing LLM output."""
    if set(draft) != ALLOWED_FIELDS:
        raise ValueError(
            f"LLM output keys must be exactly {sorted(ALLOWED_FIELDS)}; got {sorted(draft)}"
        )
    normalized = {key: draft[key] for key in ALLOWED_FIELDS}
    for value in normalized.values():
        _validate_text(value, facts)
    return normalized


def parse_llm_response(response: str, facts: Mapping[str, Any]) -> dict[str, str]:
    """Parse a JSON-only LLM response and validate its report contract."""
    try:
        payload = json.loads(response)
    except json.JSONDecodeError as error:
        raise ValueError("LLM response is not valid JSON") from error
    if not isinstance(payload, Mapping):
        raise ValueError("LLM response must be a JSON object")
    return validate_llm_draft(payload, facts)


class OpenAICompatibleReportClient:
    """Small client for local or hosted OpenAI-compatible chat endpoints."""

    def __init__(
        self,
        endpoint: str,
        api_key: str | None = None,
        model: str = "local-report-model",
        timeout: float = 60.0,
    ):
        self.endpoint = endpoint
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    def __call__(self, prompt: str) -> str:
        body = json.dumps(
            {
                "model": self.model,
                "temperature": 0,
                "messages": [
                    {"role": "system", "content": "Return JSON only."},
                    {"role": "user", "content": prompt},
                ],
                "response_format": {"type": "json_object"},
            }
        ).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = Request(self.endpoint, data=body, headers=headers, method="POST")
        with urlopen(request, timeout=self.timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
        try:
            return str(payload["choices"][0]["message"]["content"])
        except (KeyError, IndexError, TypeError) as error:
            raise ValueError("LLM response is missing choices[0].message.content") from error


def generate_llm_draft(
    facts: Mapping[str, Any],
    client: Callable[[str], str],
    fallback: Mapping[str, str],
) -> tuple[dict[str, str], str]:
    """Generate validated wording, falling back when the optional LLM fails."""
    try:
        response = client(build_llm_prompt(facts))
        return parse_llm_response(response, facts), "llm"
    except (OSError, TimeoutError, ValueError, TypeError, KeyError, IndexError):
        return dict(fallback), "template-fallback"
