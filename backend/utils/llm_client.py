"""
Optional LLM client for dataset and dashboard summaries.

The project stays fully functional without these settings. When an API key and
endpoint are present in .env, this module adds concise AI-generated summaries
to the upload analysis and dashboard payloads.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Iterable, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from dotenv import load_dotenv


load_dotenv()

DEFAULT_GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_OPENAI_URL = "https://api.openai.com/v1/chat/completions"


def _first_env(*names: str) -> str:
    for name in names:
        value = os.getenv(name, "")
        if value and value.strip():
            return value.strip()
    return ""


def _get_api_url() -> str:
    url = _first_env("LLM_API_URL")
    if url:
        return url

    if _first_env("GROQ_API_KEY"):
        return DEFAULT_GROQ_URL

    if _first_env("OPENAI_API_KEY"):
        return DEFAULT_OPENAI_URL

    return ""


def _get_api_key(api_url: str) -> str:
    explicit_key = _first_env("LLM_API_KEY")
    if explicit_key:
        return explicit_key

    lowered_url = (api_url or "").lower()
    if "groq" in lowered_url:
        return _first_env("GROQ_API_KEY", "OPENAI_API_KEY")
    if "openai" in lowered_url:
        return _first_env("OPENAI_API_KEY", "GROQ_API_KEY")

    return _first_env("GROQ_API_KEY", "OPENAI_API_KEY")


def _get_model_name() -> str:
    return _first_env("LLM_MODEL")


def is_llm_enabled() -> bool:
    api_url = _get_api_url()
    return bool(api_url and _get_api_key(api_url) and _get_model_name())


def _compact_text(value: Any, limit: int = 160) -> str:
    text = str(value).strip()
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)] + "..."


def _compact_rows(rows: Iterable[Dict[str, Any]], row_limit: int = 3, column_limit: int = 8) -> list[Dict[str, Any]]:
    compacted: list[Dict[str, Any]] = []
    for row in list(rows)[:row_limit]:
        compact_row: Dict[str, Any] = {}
        for idx, (key, value) in enumerate(row.items()):
            if idx >= column_limit:
                break
            if isinstance(value, str):
                compact_row[key] = _compact_text(value)
            else:
                compact_row[key] = value
        compacted.append(compact_row)
    return compacted


def _call_llm(system_prompt: str, user_prompt: str, max_tokens: int = 220) -> Optional[str]:
    api_url = _get_api_url()
    api_key = _get_api_key(api_url)
    model_name = _get_model_name()

    if not api_url or not api_key or not model_name:
        return None

    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.2,
        "max_tokens": max_tokens,
    }

    request = Request(
        api_url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        method="POST",
    )

    try:
        with urlopen(request, timeout=30) as response:
            raw = response.read().decode("utf-8")
        body = json.loads(raw)
        choices = body.get("choices") or []
        if not choices:
            return None

        first_choice = choices[0] or {}
        message = first_choice.get("message") or {}
        content = message.get("content") or first_choice.get("text") or ""
        content = str(content).strip()
        return content or None
    except (HTTPError, URLError, TimeoutError, ValueError, KeyError, TypeError, OSError):
        return None


def generate_dataset_summary(analysis: Dict[str, Any], sample_rows: Optional[Iterable[Dict[str, Any]]] = None) -> Optional[str]:
    """Generate a short dataset summary for the upload/overview page."""
    if not is_llm_enabled():
        return None

    compact_analysis = {
        "rows": analysis.get("rows"),
        "columns": analysis.get("columns"),
        "numeric_features": analysis.get("numeric_features", []),
        "categorical_features": analysis.get("categorical_features", []),
        "target_column": analysis.get("target_column"),
        "problem_type": analysis.get("problem_type"),
        "missing_counts": analysis.get("missing_counts", {}),
        "duplicates": analysis.get("duplicates"),
    }

    prompt = (
        "Write a concise data-science summary in 3 short sentences. "
        "Mention the likely shape of the dataset, any obvious quality concern, "
        "and the most likely modeling direction. Do not use markdown or bullets."
    )
    prompt += "\n\nDataset summary JSON:\n"
    prompt += json.dumps(compact_analysis, ensure_ascii=True, default=str)
    if sample_rows:
        prompt += "\n\nSample rows:\n"
        prompt += json.dumps(_compact_rows(sample_rows), ensure_ascii=True, default=str)

    return _call_llm(
        system_prompt="You are a precise data-science assistant that writes short, practical dataset summaries.",
        user_prompt=prompt,
        max_tokens=180,
    )


def generate_dashboard_summary(
    target: str,
    results: Dict[str, Any],
    schema: Dict[str, Any],
) -> Optional[str]:
    """Generate a short dashboard summary for the interactive dashboard page."""
    if not is_llm_enabled():
        return None

    compact_results = {
        "best_model": results.get("best_model"),
        "metrics": results.get("metrics", {}),
        "comparison": (results.get("comparison", []) or [])[:5],
        "task_type": results.get("preparation", {}).get("task_type"),
    }
    compact_schema = {
        "target": target,
        "task_type": schema.get("task_type"),
        "kpis": schema.get("kpis", [])[:4],
    }

    prompt = (
        "Write a 2-sentence dashboard summary for a data-science product. "
        "Summarize the best model, the main metric outcome, and one practical next step. "
        "Do not use markdown or bullets."
        f"\n\nDashboard schema:\n{json.dumps(compact_schema, ensure_ascii=True, default=str)}"
        f"\n\nTraining results:\n{json.dumps(compact_results, ensure_ascii=True, default=str)}"
    )

    return _call_llm(
        system_prompt="You are a concise analytics assistant that writes short dashboard summaries.",
        user_prompt=prompt,
        max_tokens=140,
    )