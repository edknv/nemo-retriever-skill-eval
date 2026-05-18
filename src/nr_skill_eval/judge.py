# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LLM-as-judge scoring (litellm-backed, standalone).

A strong LLM scores generated answers on a 1–5 scale against a ground-truth
reference. Importing this module does not import ``litellm``; only constructing
``LLMJudge`` does.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Optional

_JUDGE_SYSTEM_PROMPT = """\
You are an expert evaluator for factual question answering.

You will receive a QUESTION, a REFERENCE answer, and a CANDIDATE answer.

Step 1 -- Identify required facts:
  Break the REFERENCE into its key terms: specific numbers, names, dates,
  percentages, units, or short phrases that constitute the factual core.
  Example: "16% of adults" -> required facts = ["16%", "adults"].

Step 2 -- Check each required fact in the CANDIDATE:
  - Allow numeric equivalence: "16.00%" = "16%", "1,000" = "1000".
  - Allow paraphrasing: "Peers" matches "Peers of those adults".
  - Allow additional correct detail: extra facts do NOT reduce the score.
  - Short but correct answers are fine: "Peers" is valid for "Peers".

Step 3 -- Score on a 1-5 scale based on the fraction of required facts present:
  5 - All required facts present. Answer is fully correct.
  4 - Nearly all required facts present. One minor fact may differ trivially.
  3 - Most required facts present but at least one non-trivial fact is missing
      or slightly wrong.
  2 - Some required facts present but the core answer is incomplete or has a
      significant factual error.
  1 - None or almost none of the required facts present. Includes: wrong answer,
      irrelevant response, or stating "the context does not contain this
      information" when the reference answer exists.

Respond ONLY with valid JSON:
{"score": <integer 1-5>, "reasoning": "<one sentence citing which required facts were matched or missed>"}

No text outside the JSON object."""

_JUDGE_USER_TEMPLATE = """\
Question: {query}

Reference answer: {reference}

Candidate answer: {candidate}"""

_DEFAULT_MODEL = "nvidia_nim/mistralai/mixtral-8x22b-instruct-v0.1"


@dataclass
class JudgeResult:
    """Result from a single judge evaluation.

    ``score`` is ``None`` when the judge could not produce a score (API error,
    parse failure, empty candidate). Valid scores are 1-5.
    """

    score: Optional[int] = None
    reasoning: str = ""
    error: Optional[str] = None


class LLMJudge:
    """LLM-as-judge that scores candidate answers on a 1-5 scale via ``litellm``."""

    def __init__(
        self,
        *,
        model: str = _DEFAULT_MODEL,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        num_retries: int = 3,
        timeout: float = 120.0,
        temperature: float = 0.0,
        max_tokens: int = 256,
        extra_params: Optional[dict[str, Any]] = None,
    ):
        try:
            import litellm  # noqa: F401  (presence check; lazy-imported in judge())
        except ImportError as exc:  # pragma: no cover - exercised in install variants
            raise ImportError(
                "LLMJudge requires `litellm`; install nemo-retriever-skill-eval[llm]."
            ) from exc
        self.model = model
        self._api_base = api_base
        self._api_key = api_key
        self._num_retries = num_retries
        self._timeout = timeout
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._extra_params = dict(extra_params or {})

    @classmethod
    def from_kwargs(cls, **kwargs: Any) -> "LLMJudge":
        """Flat-kwarg constructor kept for backwards compatibility."""
        return cls(**kwargs)

    def judge(self, query: str, reference: str, candidate: str) -> JudgeResult:
        """Score ``candidate`` against ``reference`` for the given ``query``."""
        if not candidate or not candidate.strip():
            return JudgeResult(score=None, reasoning="Candidate answer was empty.", error="empty_candidate")

        import litellm

        messages = [
            {"role": "system", "content": _JUDGE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": _JUDGE_USER_TEMPLATE.format(query=query, reference=reference, candidate=candidate),
            },
        ]
        try:
            resp = litellm.completion(
                model=self.model,
                api_base=self._api_base,
                api_key=self._api_key,
                messages=messages,
                temperature=self._temperature,
                max_tokens=self._max_tokens,
                num_retries=self._num_retries,
                timeout=self._timeout,
                **self._extra_params,
            )
            raw = resp.choices[0].message.content
            return _parse_judge_response(str(raw or ""))
        except Exception as exc:
            return JudgeResult(score=None, reasoning="", error=f"judge_api_error: {exc}")


def _parse_judge_response(raw: str) -> JudgeResult:
    """Parse the judge's JSON response into a ``JudgeResult``."""
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\s*```$", "", text, flags=re.MULTILINE)
    text = text.strip()

    try:
        data = json.loads(text)
        score = int(data["score"])
        if not (1 <= score <= 5):
            raise ValueError(f"score {score} out of range 1-5")
        return JudgeResult(score=score, reasoning=str(data.get("reasoning", "")))
    except (json.JSONDecodeError, KeyError, ValueError):
        pass

    score_match = re.search(r'"score"\s*:\s*([1-5])', text)
    reasoning_match = re.search(r'"reasoning"\s*:\s*"([^"]*)"', text)
    if score_match:
        score = int(score_match.group(1))
        reasoning = reasoning_match.group(1) if reasoning_match else ""
        return JudgeResult(score=score, reasoning=reasoning)

    return JudgeResult(
        score=None,
        reasoning="",
        error=f"parse_failure: could not extract score from response: {raw[:200]!r}",
    )
