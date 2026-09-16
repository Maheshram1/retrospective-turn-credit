"""Holistic objective scoring with sparse reasoning and tool attribution.

Gemini first scores the completed final answer against the authoritative rubric
verdict and recorded outcome. A separate call then maps that fixed score onto
only the assistant turns that materially caused success or failure. Omitted
turns receive zero weight.

The reward postprocessor uses an anchored same-task baseline for magnitude, then
applies both causal maps with fixed signs:

* success-credit tokens receive positive advantage;
* failure-blame tokens receive negative advantage.

Missing or empty maps never become uniform weights. Report-writing turns keep
V24's ordinary causal treatment: Gemini may credit or blame the concrete
synthesis and prediction they commit. Objective-score failures remain
judge-local and never replay the completed policy trajectory.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import statistics
import time
from dataclasses import dataclass, replace
from typing import Any, Sequence


SCHEMA_VERSION = 8
PROMPT_VERSION = 22
DEFAULT_NEGATIVE_ADVANTAGE_SCALE = 1.0
DEFAULT_OBJECTIVE_SUCCESS_FLOOR = 0.5
logger = logging.getLogger(__name__)


class TurnCreditError(RuntimeError):
    """A turn-credit response or token mapping violated an invariant."""


class ObjectiveScoreUnavailable(TurnCreditError):
    """Judge-only attempts produced no valid trajectory objective score."""


@dataclass(frozen=True)
class TurnSpan:
    """One assistant turn and its response-token interval."""

    turn_index: int
    response_start: int
    response_end: int
    assistant_content: str
    tool_calls: tuple[str, ...]
    following_context: tuple[str, ...]
    response_token_ids: tuple[int, ...] = ()

    def as_prompt_dict(self) -> dict[str, Any]:
        return {
            "turn_index": self.turn_index,
            "assistant_content": self.assistant_content,
            "tool_calls": list(self.tool_calls),
            "following_tool_outputs": list(self.following_context),
        }


@dataclass(frozen=True)
class TurnCreditConfig:
    judge_model: str = "google/gemini-3.7-flash"
    judge_backend: str = "openrouter"
    judge_samples: int = 2
    timeout_seconds: float = 120.0
    max_output_tokens: int | None = None
    retries: int = 0
    score_repair_calls: int = 3
    attribution_repair_calls: int = 3
    score_repair_backoff_seconds: float = 0.0
    max_concurrency: int = 24
    max_prompt_chars: int = 800_000

    @classmethod
    def from_environment(cls) -> "TurnCreditConfig":
        max_output_tokens = os.environ.get(
            "TRATA_TURN_CREDIT_MAX_OUTPUT_TOKENS", ""
        ).strip()
        config = cls(
            judge_model=os.environ.get(
                "TRATA_TURN_CREDIT_JUDGE_MODEL", "google/gemini-3.7-flash"
            ).strip(),
            judge_backend=os.environ.get(
                "TRATA_TURN_CREDIT_JUDGE_BACKEND", "openrouter"
            ).strip().lower(),
            judge_samples=int(os.environ.get("TRATA_TURN_CREDIT_JUDGE_SAMPLES", "2")),
            timeout_seconds=float(os.environ.get("TRATA_TURN_CREDIT_TIMEOUT_SECONDS", "120")),
            max_output_tokens=(
                int(max_output_tokens) if max_output_tokens else None
            ),
            retries=int(os.environ.get("TRATA_TURN_CREDIT_RETRIES", "0")),
            score_repair_calls=int(
                os.environ.get("TRATA_TURN_CREDIT_SCORE_REPAIR_CALLS", "3")
            ),
            attribution_repair_calls=int(
                os.environ.get("TRATA_TURN_CREDIT_ATTRIBUTION_REPAIR_CALLS", "3")
            ),
            score_repair_backoff_seconds=float(
                os.environ.get(
                    "TRATA_TURN_CREDIT_SCORE_REPAIR_BACKOFF_SECONDS", "0"
                )
            ),
            max_concurrency=int(os.environ.get("TRATA_TURN_CREDIT_MAX_CONCURRENCY", "24")),
            max_prompt_chars=int(os.environ.get("TRATA_TURN_CREDIT_MAX_PROMPT_CHARS", "800000")),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if not self.judge_model:
            raise ValueError("turn-credit judge model must be non-empty")
        if self.judge_backend != "openrouter":
            raise ValueError("turn-credit judge backend must be openrouter")
        if self.judge_samples < 1:
            raise ValueError("turn-credit judge_samples must be positive")
        if not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0:
            raise ValueError("turn-credit timeout must be positive")
        if self.max_output_tokens is not None and self.max_output_tokens < 128:
            raise ValueError("turn-credit max output tokens must be at least 128")
        if (
            self.retries < 0
            or self.score_repair_calls < 1
            or self.attribution_repair_calls < 0
        ):
            raise ValueError("turn-credit retry counts are invalid")
        if (
            not math.isfinite(self.score_repair_backoff_seconds)
            or self.score_repair_backoff_seconds < 0
        ):
            raise ValueError("turn-credit score repair backoff must be non-negative")
        if self.max_concurrency < 1:
            raise ValueError("turn-credit max concurrency must be positive")
        if self.max_prompt_chars < 1_000:
            raise ValueError("turn-credit max prompt chars must be at least 1000")


@dataclass(frozen=True)
class ObjectiveJudgment:
    objective_score: float
    synthesis_assessment: str
    prediction_assessment: str
    decisive_error: str


@dataclass(frozen=True)
class TurnAttributionJudgment:
    success_credit: tuple[float, ...]
    failure_blame: tuple[float, ...]
    tool_utility: tuple[float, ...] = ()
    success_empty_reason: str | None = None
    failure_empty_reason: str | None = None


OBJECTIVE_JUDGE_SYSTEM_PROMPT = """You grade the final answer of one completed financial research task.

Judge one holistic objective: gather and correctly use the evidence required by
the rubric, synthesize it into a rational answer to the exact prediction
question, and commit to the evidence-supported outcome the task requests.

The payload includes an authoritative citation-aware grader verdict, including
accepted and missed rubric moves and deterministic prediction correctness.
Treat those results as facts. Do not independently inflate rubric coverage
based on headings or broad discussion. Do not calculate a weighted blend or a
separate prediction bonus. Judge whether the answer succeeds as one complete
research-and-prediction product.

The authoritative normalized coverage and accepted/missed moves are hard
evidence about completion of the requested analysis. Inspect them before
scoring. A correct outcome cannot compensate for missed required moves, and
good prose cannot turn a missing move into coverage. The holistic score should
rise only when BOTH required evidence coverage and evidence-to-prediction
synthesis improve.

The main distinction you add is synthesis quality. Check whether the answer:
- answers the exact event, date horizon, and threshold;
- uses the strongest relevant evidence on both sides;
- draws the conclusion that follows from its own evidence rather than merely
  repeating management language;
- commits to a clear outcome consistent with the evidence and uncertainty;
- avoids unsupported confidence and semantic or temporal mistakes.

Use the full 0.0 to 1.0 range. A wrong binary prediction must be below 0.5. A
thoughtful near-boundary miss can approach 0.5; a confident prediction based on
a clear reasoning error should be much lower. A correct binary outcome is not
automatically strong and can also score below 0.5 when coverage, grounding, or
synthesis is poor. High scores require both strong accepted coverage and strong
prediction synthesis.

Use these qualitative anchors, not a weighted formula:
- Below 0.5: wrong prediction, or materially incomplete/unsupported analysis.
- 0.5 to 0.74: correct or plausible prediction but important required moves,
  evidence, or synthesis remain missing.
- 0.75 to 0.89: a clear majority of required moves are accepted and the
  prediction follows from the evidence, with only limited remaining defects.
- 0.90 to 1.0: near-complete accepted coverage and strong synthesis. Reserve
  1.0 for no material remaining defect.

For example, 0.50 normalized coverage means half of the required analytical
moves were not accepted. A correct prediction alone does not make that a 0.75+
answer.

If objective_score is below 1.0, decisive_error must name the largest remaining
coverage, synthesis, or prediction weakness, even when the prediction is correct.
Use "none" only for a genuinely complete 1.0 answer.

Return JSON only:
{
  "schema_version": 8,
  "objective_score": 0.0,
  "synthesis_assessment": "name the most important accepted and missed requirements, then state whether the conclusion follows",
  "prediction_assessment": "briefly assess the binary prediction and whether it follows from the evidence",
  "decisive_error": "the main evidence-to-conclusion error, or none"
}
"""


ATTRIBUTION_JUDGE_SYSTEM_PROMPT = """You assign sparse signed causal attribution for one completed research trajectory.

The objective score, authoritative rubric verdict, recorded outcome, and
objective diagnoses are fixed facts. Do not rescore the trajectory, compare it
with sibling rollouts, or predict which attribution channel training will use.

Perform two separate audits: V24 reasoning causality and tool evidence utility.
The final_answer_md is already graded through the fixed holistic objective
score. In the reasoning audit, report-writing turns retain V24's causal
treatment. Do not reward or penalize a trajectory merely because it is long or
short.

SUCCESS: Would removing or degrading this turn materially weaken targeted
evidence collection, accepted rubric coverage, verification, rational
synthesis, or an evidence-derived prediction?

FAILURE: Did this turn materially cause a missed requirement, unsupported
claim, premature stopping, misuse or omission of relevant evidence, faulty
synthesis, or a wrong event, horizon, threshold, or outcome?
Would correcting this turn materially improve the final answer? Every failure
attribution must use failure_kind="answer_error".

REASONING TURN MAP RULES:
- Always perform both audits, regardless of score or prediction correctness.
- Wrong or low-scoring trajectories may retain success credit for sound work.
- Correct or high-scoring trajectories may receive failure blame for remaining
  coverage, synthesis, or prediction defects.
- The same turn may appear in both maps only if it materially helped one part
  of the objective and harmed another.
- Do not put tool evidence utility in these two turn maps; tool utility has its
  own field below. A report-writing turn may receive reasoning credit or blame
  only for the concrete synthesis, omission, unsupported claim, or prediction
  it commits. Protocol syntax remains a separate deterministic diagnostic.
- Never blame sound evidence gathering because later synthesis failed.
- Never blame a unique, reasonable search merely because it returned no match.
  Repeated reads, citation checks, failed commands, and report corrections are
  neutral unless the selected turn itself makes or commits a concrete answer
  error. A verification that finds and fixes a real defect can receive success
  credit.
- Attribute an omission only to a turn that caused premature stopping, ignored
  available evidence, or committed the omission in the final synthesis.
- If the submitted binary outcome is wrong, the failure map must contain at
  least one genuinely causal turn. Find the earliest decisive reasoning turn
  that formed or accepted the bad inference or outcome and the later
  report-synthesis turn that committed it, when these are distinct.
- A forecast-decision turn qualifies only when its actual assistant content
  chooses, accepts, or materially justifies the outcome or direction.
  Planning to assess the forecast, gathering or citing evidence, and discussing
  uncertainty without making the decision do not qualify. Never invent causal
  content that is absent from the selected turn.
- Report writing is not itself a failure. Blame a report turn only for the
  incorrect prediction, faulty synthesis, unsupported claim, or material
  omission it commits. Do not blame a final chat message that merely confirms
  completion.
- Evidence collected but ultimately unused is neutral in the reasoning maps.
  Routine navigation, generic planning, formatting, malformed tool syntax, and
  final confirmation receive zero unless the turn itself commits a concrete
  reasoning error.
- Use sparse attribution. Never distribute credit or blame uniformly. There is
  no target number of nonzero turns.
- Values represent causal strength and must be in (0,1]. Omit zero values.
- A channel may be empty only when genuinely no model-controlled turn passes
  its causal test. Score or correctness alone is not a valid empty reason.

Rubric coverage is not synthesis. The forecast must answer the exact event,
horizon, and threshold; weigh the strongest evidence on both sides; and choose
an outcome consistent with the evidence. Use the objective diagnoses to locate
each identified weakness in the reasoning or report turn that caused it.

TOOL EVIDENCE UTILITY:
- Audit each assistant turn that has tool_calls against all evidence available
  before that turn and the following_tool_outputs it produced.
- Positive utility means the call added new task-relevant evidence, a new
  calculation, or a verification that resolved a real uncertainty.
- Negative utility means the call was demonstrably redundant with evidence
  already available, irrelevant to the task, or avoidably failed and produced
  no usable information. A repeated call can be negative even if its syntax is
  valid.
- A unique reasonable search that returns no match is neutral. Never punish
  sound exploration merely because later synthesis ignored its evidence.
- Omit neutral calls. Omit turns that write or modify answer.md.
- utility is signed in [-1,1] excluding zero. The utility applies to every
  semantic command token in that tool-call turn; fixed wrapper syntax is masked
  separately.

Return JSON only:
{
  "schema_version": 8,
  "turn_count": N,
  "success_attributions": [
    {"turn_index": 0, "success_credit": 0.8, "reason": "specific causal contribution"}
  ],
  "success_empty_reason": null,
  "failure_attributions": [
    {"turn_index": 1, "failure_blame": 0.7, "failure_kind": "answer_error", "reason": "specific causal defect in the reasoning or final answer"}
  ],
  "failure_empty_reason": null,
  "tool_call_attributions": [
    {"turn_index": 2, "utility": -0.4, "evidence_delta": "duplicates evidence already available from turn 0", "reason": "specific utility judgment"}
  ]
}

Copy turn_count exactly. Indices must be unique within each list and between 0
and turn_count-1. The same index may occur once in each list. When a list is
nonempty, its empty_reason must be null. When empty, empty_reason must provide a
specific explanation.
"""

# Compatibility for launch-time source checks; attribution is the expensive
# trajectory-aware prompt and therefore remains the primary judge prompt.
JUDGE_SYSTEM_PROMPT = ATTRIBUTION_JUDGE_SYSTEM_PROMPT


OBJECTIVE_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "schema_version": {
            "type": "integer",
            "minimum": SCHEMA_VERSION,
            "maximum": SCHEMA_VERSION,
        },
        "objective_score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "synthesis_assessment": {"type": "string"},
        "prediction_assessment": {"type": "string"},
        "decisive_error": {"type": "string"},
    },
    "required": [
        "schema_version",
        "objective_score",
        "synthesis_assessment",
        "prediction_assessment",
        "decisive_error",
    ],
    "additionalProperties": False,
}


_ATTRIBUTION_ITEM_BASE = {
    "turn_index": {"type": "integer", "minimum": 0},
    "reason": {"type": "string"},
}


ATTRIBUTION_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "schema_version": {
            "type": "integer",
            "minimum": SCHEMA_VERSION,
            "maximum": SCHEMA_VERSION,
        },
        "turn_count": {"type": "integer", "minimum": 1},
        "success_attributions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    **_ATTRIBUTION_ITEM_BASE,
                    "success_credit": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 1.0,
                        "description": "Must be greater than zero.",
                    },
                },
                "required": ["turn_index", "success_credit", "reason"],
                "additionalProperties": False,
            },
        },
        "success_empty_reason": {"type": ["string", "null"]},
        "failure_attributions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    **_ATTRIBUTION_ITEM_BASE,
                    "failure_blame": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 1.0,
                        "description": "Must be greater than zero.",
                    },
                    "failure_kind": {
                        "type": "string",
                        "enum": ["answer_error"],
                    },
                },
                "required": [
                    "turn_index",
                    "failure_blame",
                    "failure_kind",
                    "reason",
                ],
                "additionalProperties": False,
            },
        },
        "failure_empty_reason": {"type": ["string", "null"]},
        "tool_call_attributions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "turn_index": {"type": "integer", "minimum": 0},
                    "utility": {
                        "type": "number",
                        "minimum": -1.0,
                        "maximum": 1.0,
                        "description": "Must be nonzero.",
                    },
                    "evidence_delta": {"type": "string"},
                    "reason": {"type": "string"},
                },
                "required": [
                    "turn_index",
                    "utility",
                    "evidence_delta",
                    "reason",
                ],
                "additionalProperties": False,
            },
        },
    },
    "required": [
        "schema_version",
        "turn_count",
        "success_attributions",
        "success_empty_reason",
        "failure_attributions",
        "failure_empty_reason",
        "tool_call_attributions",
    ],
    "additionalProperties": False,
}


def _judge_response_schema(system_prompt: str) -> tuple[str, dict[str, Any]]:
    if system_prompt == OBJECTIVE_JUDGE_SYSTEM_PROMPT:
        return "trata_objective_judgment", OBJECTIVE_RESPONSE_SCHEMA
    if system_prompt == ATTRIBUTION_JUDGE_SYSTEM_PROMPT:
        return "trata_turn_attribution", ATTRIBUTION_RESPONSE_SCHEMA
    raise TurnCreditError("unknown judge system prompt for structured output")


_semaphore: asyncio.Semaphore | None = None
_semaphore_limit: int | None = None


def enabled() -> bool:
    return (
        os.environ.get("TRATA_TURN_CREDIT_MODE", "off").strip().lower()
        == "calibrated_failure"
    )


def _slots(limit: int) -> asyncio.Semaphore:
    global _semaphore, _semaphore_limit
    if _semaphore is None or _semaphore_limit != limit:
        _semaphore = asyncio.Semaphore(limit)
        _semaphore_limit = limit
    return _semaphore


def _strip_json_fence(content: str) -> str:
    text = content.strip()
    if text.startswith("```"):
        lines = text.splitlines()[1:]
        if lines and lines[-1].strip() == "```":
            lines.pop()
        text = "\n".join(lines).strip()
    return text


def _compact_grader_verdict(verdict: dict[str, Any]) -> dict[str, Any]:
    """Keep authoritative outcomes without forwarding verbose juror transcripts."""
    fields = (
        "normalized",
        "themes",
        "move_votes",
        "prediction",
        "hallucinations_detected",
        "unverifiable_claims",
        "grading_failed",
    )
    return {field: verdict[field] for field in fields if field in verdict}


def _parse_attribution_channel(
    payload: dict[str, Any],
    *,
    expected_turn_count: int,
    list_field: str,
    value_field: str,
) -> tuple[float, ...]:
    items = payload.get(list_field)
    if not isinstance(items, list):
        raise TurnCreditError(f"{list_field} must be a list")
    by_index: dict[int, float] = {}
    for item in items:
        if not isinstance(item, dict):
            raise TurnCreditError(f"{list_field} item is not an object")
        index = item.get("turn_index")
        if isinstance(index, bool) or not isinstance(index, int) or index in by_index:
            raise TurnCreditError(f"{list_field} indices must be unique integers")
        if not 0 <= index < expected_turn_count:
            raise TurnCreditError(f"{list_field} turn index is out of range")
        value = item.get(value_field)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TurnCreditError(
                f"{value_field}: turn {index} coefficient is not numeric"
            )
        weight = float(value)
        if not math.isfinite(weight) or not 0.0 < weight <= 1.0:
            raise TurnCreditError(
                f"{value_field}: turn {index} coefficient is not in (0,1]"
            )
        reason = item.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            raise TurnCreditError(
                f"{list_field}: turn {index} reason must be a non-empty string"
            )
        if value_field == "failure_blame":
            failure_kind = item.get("failure_kind")
            if failure_kind != "answer_error":
                raise TurnCreditError(
                    f"{list_field}: turn {index} failure_kind must be "
                    "answer_error"
                )
        by_index[index] = weight
    return tuple(by_index.get(index, 0.0) for index in range(expected_turn_count))


def _parse_tool_utility(
    payload: dict[str, Any],
    *,
    expected_turn_count: int,
    tool_turn_indices: set[int],
    answer_write_turn_indices: set[int],
) -> tuple[float, ...]:
    items = payload.get("tool_call_attributions")
    if not isinstance(items, list):
        raise TurnCreditError("tool_call_attributions must be a list")
    by_index: dict[int, float] = {}
    seen_indices: set[int] = set()
    for item in items:
        if not isinstance(item, dict):
            raise TurnCreditError("tool_call_attributions item is not an object")
        index = item.get("turn_index")
        if (
            isinstance(index, bool)
            or not isinstance(index, int)
            or index in seen_indices
        ):
            raise TurnCreditError(
                "tool_call_attributions indices must be unique integers"
            )
        if not 0 <= index < expected_turn_count or index not in tool_turn_indices:
            raise TurnCreditError(
                "tool_call_attributions index must identify a tool-call turn"
            )
        seen_indices.add(index)
        if index in answer_write_turn_indices:
            continue
        value = item.get("utility")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TurnCreditError(f"tool utility for turn {index} is not numeric")
        utility = float(value)
        if not math.isfinite(utility) or utility == 0.0 or not -1.0 <= utility <= 1.0:
            raise TurnCreditError(
                f"tool utility for turn {index} must be finite in [-1,1] excluding zero"
            )
        for field in ("evidence_delta", "reason"):
            text = item.get(field)
            if not isinstance(text, str) or not text.strip():
                raise TurnCreditError(
                    f"tool_call_attributions turn {index} {field} must be non-empty"
                )
        by_index[index] = utility
    return tuple(by_index.get(index, 0.0) for index in range(expected_turn_count))


def _parse_payload(content: str, *, response_name: str) -> dict[str, Any]:
    try:
        payload = json.loads(_strip_json_fence(content))
    except json.JSONDecodeError as exc:
        raise TurnCreditError(f"{response_name} returned invalid JSON: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != SCHEMA_VERSION:
        raise TurnCreditError(f"{response_name} has the wrong schema_version")
    return payload


def parse_objective_judgment(content: str) -> ObjectiveJudgment:
    payload = _parse_payload(content, response_name="objective judge response")
    objective_score = payload.get("objective_score")
    if isinstance(objective_score, bool) or not isinstance(objective_score, (int, float)):
        raise TurnCreditError("objective_score must be numeric")
    objective_score = float(objective_score)
    if not math.isfinite(objective_score) or not 0.0 <= objective_score <= 1.0:
        raise TurnCreditError("objective_score must be finite and between 0 and 1")
    text_fields: dict[str, str] = {}
    for field in (
        "synthesis_assessment",
        "prediction_assessment",
        "decisive_error",
    ):
        value = payload.get(field)
        if not isinstance(value, str) or not value.strip():
            raise TurnCreditError(f"{field} must be a non-empty string")
        text_fields[field] = value.strip()[:2_000]
    return ObjectiveJudgment(objective_score=objective_score, **text_fields)


def parse_turn_attribution_judgment(
    content: str,
    expected_turn_count: int,
    *,
    tool_turn_indices: Sequence[int] = (),
    answer_write_turn_indices: Sequence[int] = (),
) -> TurnAttributionJudgment:
    payload = _parse_payload(content, response_name="attribution judge response")
    if payload.get("turn_count") != expected_turn_count:
        raise TurnCreditError("attribution judge response has the wrong turn_count")
    success_credit = _parse_attribution_channel(
        payload,
        expected_turn_count=expected_turn_count,
        list_field="success_attributions",
        value_field="success_credit",
    )
    failure_blame = _parse_attribution_channel(
        payload,
        expected_turn_count=expected_turn_count,
        list_field="failure_attributions",
        value_field="failure_blame",
    )
    answer_turn_set = set(answer_write_turn_indices)
    tool_utility = _parse_tool_utility(
        payload,
        expected_turn_count=expected_turn_count,
        tool_turn_indices=set(tool_turn_indices),
        answer_write_turn_indices=answer_turn_set,
    )

    def empty_reason(field: str, weights: Sequence[float]) -> str | None:
        reason = payload.get(field)
        if any(weight > 0.0 for weight in weights):
            if reason is not None:
                raise TurnCreditError(f"{field} must be null for a non-empty map")
            return None
        if not isinstance(reason, str) or not reason.strip():
            raise TurnCreditError(
                f"{field} must explain why the attribution map is empty"
            )
        return reason.strip()[:2_000]

    return TurnAttributionJudgment(
        success_credit=success_credit,
        failure_blame=failure_blame,
        tool_utility=tool_utility,
        success_empty_reason=empty_reason(
            "success_empty_reason", success_credit
        ),
        failure_empty_reason=empty_reason(
            "failure_empty_reason", failure_blame
        ),
    )


def _objective_repair_prompt(prompt: str, error: BaseException) -> str:
    return (
        prompt
        + "\n\nOBJECTIVE SCORE REPAIR REQUIRED. The prior judge call failed: "
        + str(error)[:500]
        + ". Return JSON only with schema_version=8, one finite objective_score "
        + "in [0,1], and non-empty synthesis_assessment, "
        + "prediction_assessment, and decisive_error strings."
    )


def _attribution_repair_prompt(
    prompt: str,
    *,
    turn_count: int,
    error: BaseException,
    required_channels: Sequence[str] = (),
) -> str:
    requirement = (
        " The following channels must each contain at least one materially "
        f"causal turn: {', '.join(required_channels)}."
        if required_channels
        else ""
    )
    return (
        prompt
        + "\n\nATTRIBUTION REPAIR REQUIRED. The prior judge call failed: "
        + str(error)[:500]
        + f". Return schema_version=8, turn_count={turn_count}, sparse "
        + "success_attributions and failure_attributions lists, with unique "
        + f"indices from 0 through {turn_count - 1}, plus success_empty_reason "
        + "and failure_empty_reason fields that are null for non-empty maps, "
        + "plus a tool_call_attributions list. "
        + "specific strings for empty maps. Every attribution needs a non-empty "
        + "reason. Every failure attribution also needs failure_kind equal to "
        + "answer_error. Tool utility must be signed and evidence-relative. "
        + "Never attribute malformed protocol syntax or answer.md writing turns "
        + "in these semantic maps."
        + requirement
    )


def normalize_draw(raw: Sequence[float], token_counts: Sequence[int]) -> tuple[float, ...]:
    """Validate one bounded draw and zero coefficients on untrainable turns."""
    if len(raw) != len(token_counts):
        raise TurnCreditError("turn weights and token counts have different lengths")
    total_tokens = sum(token_counts)
    if total_tokens <= 0:
        raise TurnCreditError("trajectory has no trainable assistant tokens")
    bounded: list[float] = []
    for weight, count in zip(raw, token_counts, strict=True):
        value = float(weight)
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise TurnCreditError("turn-credit coefficient is not in [0,1]")
        bounded.append(value if count else 0.0)
    return tuple(bounded)


def build_response_token_weights(
    *,
    response_length: int,
    loss_mask: Sequence[int],
    turns: Sequence[TurnSpan],
    raw_vectors: Sequence[Sequence[float]],
    channel: str,
) -> tuple[list[float], dict[str, Any]]:
    """Average bounded sparse maps; missing maps remain all zero."""
    if response_length != len(loss_mask):
        raise TurnCreditError("loss mask length does not match response length")
    if not turns:
        raise TurnCreditError("trajectory has no assistant turns")
    if [turn.turn_index for turn in turns] != list(range(len(turns))):
        raise TurnCreditError("assistant turn indices are not contiguous")

    ownership = [-1] * response_length
    token_counts: list[int] = []
    for turn in turns:
        if not (0 <= turn.response_start <= turn.response_end <= response_length):
            raise TurnCreditError(f"turn {turn.turn_index} has an invalid token interval")
        for position in range(turn.response_start, turn.response_end):
            if ownership[position] != -1:
                raise TurnCreditError("assistant token intervals overlap")
            ownership[position] = turn.turn_index
        token_counts.append(
            sum(
                int(bool(loss_mask[position]))
                for position in range(turn.response_start, turn.response_end)
            )
        )
    for position, active in enumerate(loss_mask):
        if active and ownership[position] < 0:
            raise TurnCreditError(
                f"trainable response token {position} is not owned by an assistant turn"
            )

    normalized: list[tuple[float, ...]] = []
    rejected: list[str] = []
    for vector in raw_vectors:
        try:
            normalized.append(normalize_draw(vector, token_counts))
        except TurnCreditError as exc:
            rejected.append(str(exc))
    empty_map_fallback = not normalized
    if empty_map_fallback:
        normalized = [tuple(0.0 for _ in token_counts)]

    averaged = [
        statistics.fmean(vector[index] for vector in normalized)
        for index in range(len(turns))
    ]
    token_weights = [0.0] * response_length
    for position, turn_index in enumerate(ownership):
        if turn_index >= 0 and loss_mask[position]:
            token_weights[position] = averaged[turn_index]

    active_tokens = sum(int(bool(item)) for item in loss_mask)
    weighted_mass = sum(
        weight
        for weight, active in zip(token_weights, loss_mask, strict=True)
        if active
    )
    if active_tokens <= 0:
        raise TurnCreditError("trajectory has no trainable assistant tokens")
    active_turn_weights = [
        weight for weight, count in zip(averaged, token_counts, strict=True) if count
    ]
    return token_weights, {
        "channel": channel,
        "map_draws_supplied": len(raw_vectors),
        "map_draws_used": 0 if empty_map_fallback else len(normalized),
        "map_draws_rejected": len(rejected),
        "uniform_fallback": 0.0,
        "empty_map_fallback": float(empty_map_fallback),
        "fallback_reasons": rejected,
        "turn_count": len(turns),
        "eligible_turn_count": sum(count > 0 for count in token_counts),
        "eligible_action_tokens": active_tokens,
        "weight_min": min(active_turn_weights),
        "weight_max": max(active_turn_weights),
        "weight_std": (
            statistics.pstdev(active_turn_weights)
            if len(active_turn_weights) > 1
            else 0.0
        ),
        "token_weighted_mean": weighted_mass / active_tokens,
        "zero_map": float(weighted_mass == 0.0),
        "raw_vectors": [list(vector) for vector in raw_vectors],
        "normalized_vectors": [list(vector) for vector in normalized],
        "bounded_vectors": [list(vector) for vector in normalized],
        "applied_turn_weights": averaged,
        "eligible_tokens_per_turn": token_counts,
    }


def answer_write_turn_indices(turns: Sequence[TurnSpan]) -> tuple[int, ...]:
    """Return tool turns that create or modify answer.md."""
    indices: list[int] = []
    for turn in turns:
        if any(
            "answer.md" in call
            and any(
                marker in call
                for marker in (
                    ">",
                    "tee",
                    "apply_patch",
                    "write_text",
                    "sed -i",
                    ".write(",
                    "cp ",
                    "mv ",
                )
            )
            for call in turn.tool_calls
        ):
            indices.append(turn.turn_index)
    return tuple(indices)


def _openrouter_model_name(model: str) -> str:
    if model.startswith("openrouter/google/"):
        return model.removeprefix("openrouter/")
    if model.startswith("gemini/"):
        return "google/" + model.split("/", 1)[1]
    if model.startswith("google/") and model.count("/") == 1:
        return model
    raise TurnCreditError(f"unsupported OpenRouter Gemini model name: {model!r}")


def _build_judge_request(
    config: TurnCreditConfig, prompt: str, system_prompt: str
) -> tuple[str, dict[str, str], dict[str, Any]]:
    schema_name, response_schema = _judge_response_schema(system_prompt)
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise TurnCreditError("OpenRouter turn-credit judge requires OPENROUTER_API_KEY")
    payload: dict[str, Any] = {
        "model": _openrouter_model_name(config.judge_model),
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": schema_name,
                "strict": True,
                "schema": response_schema,
            },
        },
        "provider": {"require_parameters": True},
    }
    if config.max_output_tokens is not None:
        payload["max_tokens"] = config.max_output_tokens
    return (
        "https://openrouter.ai/api/v1/chat/completions",
        {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "X-Title": "Trata GRPO Turn Credit Judge",
        },
        payload,
    )


def _extract_judge_content(backend: str, response_text: str) -> str:
    try:
        payload = json.loads(response_text)
    except json.JSONDecodeError as exc:
        raise TurnCreditError("turn-credit judge returned invalid JSON") from exc

    error = payload.get("error")
    if isinstance(error, dict):
        raise TurnCreditError(
            "turn-credit judge response error "
            f"{error.get('code', 'unknown')}: {error.get('message', error)!s}"
        )

    try:
        if backend == "openrouter":
            content = payload["choices"][0]["message"]["content"]
            if isinstance(content, list):
                content = "".join(
                    str(part.get("text", ""))
                    for part in content
                    if isinstance(part, dict)
                )
            if not isinstance(content, str) or not content.strip():
                raise TypeError("empty OpenRouter message content")
            return content

        parts = payload["candidates"][0]["content"]["parts"]
        content = "".join(str(part.get("text", "")) for part in parts)
        if not content.strip():
            raise TypeError("empty native Gemini candidate content")
        return content
    except (KeyError, IndexError, TypeError) as exc:
        raise TurnCreditError(
            f"{backend} turn-credit response has no candidate text"
        ) from exc


async def _request_judge_content(
    config: TurnCreditConfig, prompt: str, system_prompt: str
) -> tuple[str, int]:
    import aiohttp

    last_error: BaseException | None = None
    for attempt in range(config.retries + 1):
        try:
            attempt_prompt = (
                prompt
                + "\n\nThe previous transport or JSON attempt failed: "
                + str(last_error)[:500]
                + ". Return valid JSON only."
                if attempt
                else prompt
            )
            url, headers, request_payload = _build_judge_request(
                config, attempt_prompt, system_prompt
            )
            async with _slots(config.max_concurrency):
                async with asyncio.timeout(config.timeout_seconds + 5):
                    timeout = aiohttp.ClientTimeout(total=config.timeout_seconds)
                    async with aiohttp.ClientSession(timeout=timeout) as session:
                        async with session.post(
                            url,
                            headers=headers,
                            json=request_payload,
                        ) as response:
                            response_text = await response.text()
                            if response.status != 200:
                                raise TurnCreditError(
                                    f"{config.judge_backend} turn-credit HTTP "
                                    f"{response.status}: "
                                    f"{response_text[:1000]}"
                                )
            content = _extract_judge_content(config.judge_backend, response_text)
            return content, attempt
        except Exception as exc:  # transport and score-contract failures are judge-local
            last_error = exc
            if attempt < config.retries:
                logger.warning(
                    "turn-credit judge retry %d/%d after: %s",
                    attempt + 1,
                    config.retries,
                    exc,
                )
                await asyncio.sleep(min(2**attempt, 4))
    assert last_error is not None
    if isinstance(last_error, TurnCreditError):
        raise last_error
    raise TurnCreditError(f"turn-credit judge failed: {last_error}") from last_error


async def _one_objective_draw(
    config: TurnCreditConfig,
    prompt: str,
    *,
    prediction_correct: bool | None,
) -> tuple[ObjectiveJudgment, int]:
    content, attempt = await _request_judge_content(
        config, prompt, OBJECTIVE_JUDGE_SYSTEM_PROMPT
    )
    judgment = parse_objective_judgment(content)
    if prediction_correct is False and judgment.objective_score >= 0.5:
        raise TurnCreditError(
            "objective judge scored a wrong binary prediction at or above 0.5"
        )
    return judgment, attempt


async def _one_attribution_draw(
    config: TurnCreditConfig,
    prompt: str,
    turns: Sequence[TurnSpan],
) -> tuple[TurnAttributionJudgment, int]:
    content, attempt = await _request_judge_content(
        config, prompt, ATTRIBUTION_JUDGE_SYSTEM_PROMPT
    )
    return (
        parse_turn_attribution_judgment(
            content,
            len(turns),
            tool_turn_indices=[
                turn.turn_index for turn in turns if turn.tool_calls
            ],
            answer_write_turn_indices=answer_write_turn_indices(turns),
        ),
        attempt,
    )


async def assign_turn_credit(
    *,
    instruction: str,
    grading_rubric: str,
    final_answer: str,
    objective_context: dict[str, Any],
    response_length: int,
    loss_mask: Sequence[int],
    turns: Sequence[TurnSpan],
    grader_verdict: dict[str, Any] | None = None,
    config: TurnCreditConfig | None = None,
) -> tuple[list[float], list[float], float, dict[str, Any]]:
    config = config or TurnCreditConfig.from_environment()
    config.validate()
    grader_verdict = _compact_grader_verdict(grader_verdict or {})
    prediction_verdict = grader_verdict.get("prediction")
    prediction_correct_value = (
        prediction_verdict.get("correct")
        if isinstance(prediction_verdict, dict)
        else objective_context.get("prediction_correct")
    )
    prediction_correct = None
    if isinstance(prediction_correct_value, bool):
        prediction_correct = prediction_correct_value
    elif isinstance(prediction_correct_value, (int, float)) and not isinstance(
        prediction_correct_value, bool
    ):
        if prediction_correct_value in (0, 1):
            prediction_correct = bool(prediction_correct_value)
    objective_payload = {
        "task_instruction": instruction,
        "hidden_grading_rubric": grading_rubric,
        "final_answer_md": final_answer,
        "authoritative_grader_verdict": grader_verdict,
        "objective_context": objective_context,
    }
    objective_prompt = json.dumps(
        objective_payload, ensure_ascii=False, separators=(",", ":")
    )
    if len(objective_prompt) > config.max_prompt_chars:
        raise TurnCreditError(
            "objective prompt has "
            f"{len(objective_prompt)} chars; maximum is {config.max_prompt_chars}"
        )
    started = time.monotonic()
    single_attempt_config = replace(config, retries=0)
    objective_initial = await asyncio.gather(
        *[
            _one_objective_draw(
                single_attempt_config,
                objective_prompt,
                prediction_correct=prediction_correct,
            )
            for _ in range(config.judge_samples)
        ],
        return_exceptions=True,
    )
    objective_valid = [
        row for row in objective_initial if not isinstance(row, BaseException)
    ]
    objective_failures: list[BaseException] = [
        row for row in objective_initial if isinstance(row, BaseException)
    ]
    objective_initial_valid_count = len(objective_valid)
    objective_initial_failed_count = len(objective_failures)
    score_repair_calls = 0
    repair_wait_seconds = 0.0
    while not objective_valid and score_repair_calls < config.score_repair_calls:
        score_repair_calls += 1
        delay = min(
            config.score_repair_backoff_seconds * (2 ** (score_repair_calls - 1)),
            120.0,
        )
        if delay:
            logger.warning(
                "turn-credit score repair %d/%d waiting %.1fs after: %s",
                score_repair_calls,
                config.score_repair_calls,
                delay,
                (
                    objective_failures[-1]
                    if objective_failures
                    else "missing objective score"
                ),
            )
            await asyncio.sleep(delay)
            repair_wait_seconds += delay
        repair_prompt = _objective_repair_prompt(
            objective_prompt,
            (
                objective_failures[-1]
                if objective_failures
                else TurnCreditError("missing objective score")
            ),
        )
        try:
            objective_valid.append(
                await _one_objective_draw(
                    single_attempt_config,
                    repair_prompt,
                    prediction_correct=prediction_correct,
                )
            )
        except Exception as exc:  # never regenerate the completed policy trajectory
            objective_failures.append(exc)
    if not objective_valid:
        raise ObjectiveScoreUnavailable(
            "objective judge produced no valid score after "
            f"{config.judge_samples + score_repair_calls} judge-only calls: "
            f"{objective_failures[-1]}"
        )

    objective_judgments = [row[0] for row in objective_valid]
    objective_scores = [
        judgment.objective_score for judgment in objective_judgments
    ]
    objective_score = statistics.fmean(objective_scores)
    objective_synthesis_diagnoses = [
        {
            "synthesis_assessment": judgment.synthesis_assessment,
            "prediction_assessment": judgment.prediction_assessment,
            "decisive_error": judgment.decisive_error,
        }
        for judgment in objective_judgments
    ]
    required_channels = possible_active_attribution_channels(
        prediction_correct=prediction_correct,
        objective_score=objective_score,
    )
    attribution_payload = {
        **objective_payload,
        "objective_score": objective_score,
        "objective_synthesis_diagnoses": objective_synthesis_diagnoses,
        "turn_count": len(turns),
        "agent_turns": [turn.as_prompt_dict() for turn in turns],
    }
    attribution_prompt = json.dumps(
        attribution_payload, ensure_ascii=False, separators=(",", ":")
    )
    if len(attribution_prompt) > config.max_prompt_chars:
        raise TurnCreditError(
            "attribution prompt has "
            f"{len(attribution_prompt)} chars; maximum is {config.max_prompt_chars}"
        )

    attribution_initial = await asyncio.gather(
        *[
            _one_attribution_draw(
                single_attempt_config, attribution_prompt, turns
            )
            for _ in range(config.judge_samples)
        ],
        return_exceptions=True,
    )
    attribution_valid = [
        row for row in attribution_initial if not isinstance(row, BaseException)
    ]
    attribution_failures: list[BaseException] = [
        row for row in attribution_initial if isinstance(row, BaseException)
    ]

    def tool_values(judgment: TurnAttributionJudgment) -> tuple[float, ...]:
        if not judgment.tool_utility:
            return tuple(0.0 for _ in turns)
        if len(judgment.tool_utility) != len(turns):
            raise TurnCreditError("tool utility vector has the wrong turn count")
        return judgment.tool_utility

    def row_has_mass(
        row: tuple[TurnAttributionJudgment, int], channel: str
    ) -> bool:
        judgment = row[0]
        if any(value > 0.0 for value in getattr(judgment, channel)):
            return True
        utilities = tool_values(judgment)
        if channel == "success_credit" and any(value > 0.0 for value in utilities):
            return True
        if channel == "failure_blame" and any(value < 0.0 for value in utilities):
            return True
        return False

    def rows_with_mass(channel: str) -> list[tuple[TurnAttributionJudgment, int]]:
        return [
            row
            for row in attribution_valid
            if row_has_mass(row, channel)
        ]

    usable_by_channel = {
        channel: rows_with_mass(channel)
        for channel in ("success_credit", "failure_blame")
    }

    def missing_required_channels() -> tuple[str, ...]:
        return tuple(
            channel for channel in required_channels if not usable_by_channel[channel]
        )

    attribution_repair_calls = 0
    while (
        missing_required_channels()
        and attribution_repair_calls < config.attribution_repair_calls
    ):
        attribution_repair_calls += 1
        missing = missing_required_channels()
        reason = (
            attribution_failures[-1]
            if attribution_failures
            else TurnCreditError(
                "empty possible training channel(s): " + ", ".join(missing)
            )
        )
        repair_prompt = _attribution_repair_prompt(
            attribution_prompt,
            turn_count=len(turns),
            error=reason,
            required_channels=tuple(
                "negative semantic attribution"
                if channel == "failure_blame"
                else "positive semantic attribution"
                for channel in missing
            ),
        )
        try:
            repaired = await _one_attribution_draw(
                single_attempt_config, repair_prompt, turns
            )
            attribution_valid.append(repaired)
            usable_by_channel = {
                channel: rows_with_mass(channel)
                for channel in ("success_credit", "failure_blame")
            }
        except Exception as exc:  # judge-only repair; never replay the trajectory
            attribution_failures.append(exc)

    # Average reasoning and tool utility independently.
    base_success_vectors = [
        row[0].success_credit
        for row in attribution_valid
        if any(row[0].success_credit)
    ]
    base_failure_vectors = [
        row[0].failure_blame
        for row in attribution_valid
        if any(row[0].failure_blame)
    ]
    base_success, success_metrics = build_response_token_weights(
        response_length=response_length,
        loss_mask=loss_mask,
        turns=turns,
        raw_vectors=base_success_vectors,
        channel="success_credit",
    )
    base_failure, failure_metrics = build_response_token_weights(
        response_length=response_length,
        loss_mask=loss_mask,
        turns=turns,
        raw_vectors=base_failure_vectors,
        channel="failure_blame",
    )
    report_turns = set(answer_write_turn_indices(turns))

    tool_success_vectors = [
        tuple(max(value, 0.0) for value in tool_values(row[0]))
        for row in attribution_valid
        if any(value > 0.0 for value in tool_values(row[0]))
    ]
    tool_failure_vectors = [
        tuple(max(-value, 0.0) for value in tool_values(row[0]))
        for row in attribution_valid
        if any(value < 0.0 for value in tool_values(row[0]))
    ]
    tool_success, tool_success_metrics = build_response_token_weights(
        response_length=response_length,
        loss_mask=loss_mask,
        turns=turns,
        raw_vectors=tool_success_vectors,
        channel="tool_utility_positive",
    )
    tool_failure, tool_failure_metrics = build_response_token_weights(
        response_length=response_length,
        loss_mask=loss_mask,
        turns=turns,
        raw_vectors=tool_failure_vectors,
        channel="tool_utility_negative",
    )
    for turn in turns:
        if turn.turn_index in report_turns:
            tool_success[turn.response_start:turn.response_end] = [0.0] * (
                turn.response_end - turn.response_start
            )
            tool_failure[turn.response_start:turn.response_end] = [0.0] * (
                turn.response_end - turn.response_start
            )
    success_weights = [
        max(base, tool)
        for base, tool in zip(base_success, tool_success, strict=True)
    ]
    failure_weights = [
        max(base, tool)
        for base, tool in zip(base_failure, tool_failure, strict=True)
    ]
    active_tokens = sum(int(bool(value)) for value in loss_mask)
    success_metrics = {
        **success_metrics,
        "combined_token_weighted_mean": (
            sum(success_weights) / active_tokens if active_tokens else 0.0
        ),
    }
    failure_metrics = {
        **failure_metrics,
        "combined_token_weighted_mean": (
            sum(failure_weights) / active_tokens if active_tokens else 0.0
        ),
    }
    missing_channels = missing_required_channels()
    required_channel_available = not missing_channels
    metrics: dict[str, Any] = {
        "enabled": 1.0,
        "applied": 1.0,
        "valid": 1.0,
        "judge_model": config.judge_model,
        "judge_backend": config.judge_backend,
        "judge_prompt_version": PROMPT_VERSION,
        "judge_seconds": time.monotonic() - started,
        "judge_initial_requested": config.judge_samples,
        "judge_initial_valid_scores": objective_initial_valid_count,
        "judge_initial_failed_scores": objective_initial_failed_count,
        "judge_score_repair_calls": score_repair_calls,
        "judge_score_repair_wait_seconds": repair_wait_seconds,
        "judge_score_repair_valid": float(
            score_repair_calls > 0 and bool(objective_valid)
        ),
        "objective_score": objective_score,
        "objective_score_draws": objective_scores,
        "objective_synthesis_diagnoses": objective_synthesis_diagnoses,
        "attribution_initial_requested": config.judge_samples,
        "attribution_initial_valid": len(
            [
                row
                for row in attribution_initial
                if not isinstance(row, BaseException)
            ]
        ),
        "attribution_initial_failed": len(
            [row for row in attribution_initial if isinstance(row, BaseException)]
        ),
        "attribution_repair_calls": attribution_repair_calls,
        "attribution_repair_valid": float(
            attribution_repair_calls > 0 and required_channel_available
        ),
        "attribution_required_channel": (
            required_channels[0] if len(required_channels) == 1 else "both"
        ),
        "attribution_required_channels": list(required_channels),
        "attribution_missing_required_channels": list(missing_channels),
        "attribution_channel_available": {
            channel: float(bool(rows)) for channel, rows in usable_by_channel.items()
        },
        "attribution_required_channel_available": float(
            required_channel_available
        ),
        "attribution_contract_errors": [
            str(error) for error in attribution_failures
        ],
        "attribution_success_empty_reasons": [
            row[0].success_empty_reason
            for row in attribution_valid
            if row[0].success_empty_reason
        ],
        "attribution_failure_empty_reasons": [
            row[0].failure_empty_reason
            for row in attribution_valid
            if row[0].failure_empty_reason
        ],
        "success_credit": success_metrics,
        "failure_blame": failure_metrics,
        "tool_utility": {
            "positive": tool_success_metrics,
            "negative": tool_failure_metrics,
            "positive_token_count": sum(value > 0.0 for value in tool_success),
            "negative_token_count": sum(value > 0.0 for value in tool_failure),
            "answer_write_tokens_suppressed": sum(
                int(bool(loss_mask[position]))
                for turn in turns
                if turn.turn_index in report_turns
                for position in range(turn.response_start, turn.response_end)
            ),
        },
    }
    return success_weights, failure_weights, objective_score, metrics


def _negative_advantage_scale() -> float:
    value = float(
        os.environ.get(
            "TRATA_NEGATIVE_ADVANTAGE_SCALE",
            str(DEFAULT_NEGATIVE_ADVANTAGE_SCALE),
        )
    )
    if not math.isfinite(value) or not 0.0 < value <= 1.0:
        raise TurnCreditError(
            "TRATA_NEGATIVE_ADVANTAGE_SCALE must be finite and in (0,1]"
        )
    return value


def _objective_success_floor() -> float:
    value = float(
        os.environ.get(
            "TRATA_OBJECTIVE_SUCCESS_FLOOR",
            str(DEFAULT_OBJECTIVE_SUCCESS_FLOOR),
        )
    )
    if not math.isfinite(value) or not 0.0 < value < 1.0:
        raise TurnCreditError(
            "TRATA_OBJECTIVE_SUCCESS_FLOOR must be finite and in (0,1)"
        )
    return value


def possible_active_attribution_channels(
    *, prediction_correct: bool, objective_score: float, floor: float | None = None
) -> tuple[str, ...]:
    """Return channels whose causal contract requires nonzero attribution."""
    success_floor = _objective_success_floor() if floor is None else float(floor)
    if not math.isfinite(objective_score) or not 0.0 <= objective_score <= 1.0:
        raise TurnCreditError("objective score must be finite and in [0,1]")
    if not math.isfinite(success_floor) or not 0.0 < success_floor < 1.0:
        raise TurnCreditError("objective success floor must be finite and in (0,1)")

    channels: list[str] = []
    if prediction_correct and objective_score > success_floor:
        channels.append("success_credit")
    if objective_score < 1.0:
        channels.append("failure_blame")
    if not channels:
        channels.append("success_credit")
    return tuple(channels)


def anchored_objective_group_advantages(
    samples: Sequence[Any], rewards: Sequence[float]
) -> list[float]:
    """Annotate and return true group-relative objective advantages."""
    if len(samples) < 2 or len(samples) != len(rewards):
        raise TurnCreditError("anchored objective group has an invalid size")
    values = [float(reward) for reward in rewards]
    if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in values):
        raise TurnCreditError("holistic objective rewards must be finite and in [0,1]")
    group_mean = statistics.fmean(values)
    baseline = group_mean
    correct_count = sum(
        int(bool((sample.metadata or {}).get("prediction_correct")))
        for sample in samples
    )
    contrast = (
        "mixed"
        if 0 < correct_count < len(samples)
        else "all_correct" if correct_count == len(samples) else "all_incorrect"
    )
    advantages: list[float] = []
    for sample, reward in zip(samples, values, strict=True):
        advantage = reward - baseline
        sample.metadata = {
            **(sample.metadata or {}),
            "objective_group_mean": group_mean,
            "objective_advantage_baseline": baseline,
            "objective_anchored_advantage": advantage,
            "objective_advantage_rule": "score_minus_sibling_group_mean",
            "prediction_group_correct_count": correct_count,
            "prediction_group_contrast": contrast,
            "prediction_sign_guard_applied": 0.0,
        }
        advantages.append(advantage)
    return advantages


def post_process_anchored_objective_rewards(
    args: Any, samples: Sequence[Any]
) -> tuple[list[float], list[float]]:
    """Keep same-task contrast without reinforcing a uniformly weak group."""
    group_size = int(getattr(args, "n_samples_per_prompt", 0) or 0)
    if group_size < 2:
        raise TurnCreditError("anchored objective rewards require sibling groups")
    if not samples or len(samples) % group_size:
        raise TurnCreditError(
            "anchored objective reward batch is not divisible by group size"
        )

    raw_rewards = [float(sample.get_reward_value(args)) for sample in samples]
    advantages: list[float] = []
    for start in range(0, len(samples), group_size):
        advantages.extend(
            anchored_objective_group_advantages(
                samples[start : start + group_size],
                raw_rewards[start : start + group_size],
            )
        )
    return raw_rewards, advantages


def apply_calibrated_turn_credit(args: Any, rollout_data: dict[str, Any]) -> None:
    """Apply both causal maps with group contrast controlling their magnitude."""
    del args
    import torch
    from megatron.core import mpu
    from slime.backends.megatron_utils.cp_utils import slice_log_prob_with_cp

    if not mpu.is_pipeline_last_stage():
        return
    rewards = rollout_data["rewards"]
    kl = rollout_data["kl"]
    loss_masks = rollout_data["loss_masks"]
    success_weights = rollout_data.get("turn_credit_weights")
    failure_weights = rollout_data.get("turn_failure_weights")
    terminal_failure_scales = rollout_data.get("terminal_failure_scales")
    response_lengths = rollout_data["response_lengths"]
    total_lengths = rollout_data["total_lengths"]
    if (
        success_weights is None
        or failure_weights is None
        or terminal_failure_scales is None
    ):
        raise TurnCreditError(
            "actor batch is missing success, failure, or terminal scale data"
        )
    if not (
        len(success_weights)
        == len(failure_weights)
        == len(terminal_failure_scales)
        == len(rewards)
    ):
        raise TurnCreditError("turn-credit batch length does not match rewards")

    negative_scale = _negative_advantage_scale()
    cp_size = mpu.get_context_parallel_world_size()
    advantages = []
    selected_channel_empty: list[bool] = []
    for (
        reward,
        local_kl,
        raw_mask,
        raw_success,
        raw_failure,
        raw_terminal_scale,
        response_len,
        total_len,
    ) in zip(
        rewards,
        kl,
        loss_masks,
        success_weights,
        failure_weights,
        terminal_failure_scales,
        response_lengths,
        total_lengths,
        strict=True,
    ):
        success = raw_success[:response_len].to(
            device=local_kl.device, dtype=torch.float32
        )
        failure = raw_failure[:response_len].to(
            device=local_kl.device, dtype=torch.float32
        )
        mask = raw_mask[:response_len].to(device=local_kl.device, dtype=torch.bool)
        if success.numel() != response_len or failure.numel() != response_len:
            raise TurnCreditError("actor turn-credit tensor length mismatch")
        for name, weights in (("success", success), ("failure", failure)):
            if (
                not torch.isfinite(weights).all()
                or (weights < 0).any()
                or (weights > 1).any()
            ):
                raise TurnCreditError(f"actor {name} weights contain invalid values")
            if not mask.any() and weights.abs().sum() != 0:
                raise TurnCreditError(
                    f"fully masked trajectory has nonzero {name} weights"
                )
        centered_reward = torch.as_tensor(
            reward, device=success.device, dtype=success.dtype
        )
        if centered_reward.numel() != 1 or not torch.isfinite(centered_reward).all():
            raise TurnCreditError("centered GRPO reward must be one finite scalar")
        terminal_scale = torch.as_tensor(
            raw_terminal_scale, device=success.device, dtype=success.dtype
        )
        if (
            terminal_scale.numel() != 1
            or not torch.isfinite(terminal_scale).all()
            or terminal_scale.item() < 1.0
        ):
            raise TurnCreditError(
                "terminal failure advantage scale must be one finite scalar >= 1"
            )
        if centered_reward.item() >= 0.0 and terminal_scale.item() != 1.0:
            raise TurnCreditError(
                "positive advantages cannot use terminal failure scaling"
            )
        channel_empty = bool(
            centered_reward.item() != 0.0
            and mask.any()
            and success[mask].sum().item() == 0.0
            and failure[mask].sum().item() == 0.0
        )
        selected_channel_empty.append(channel_empty)
        magnitude = centered_reward.abs()
        full = magnitude * (
            success - negative_scale * terminal_scale * failure
        )
        full = torch.where(mask, full, torch.zeros_like(full))
        advantages.append(
            slice_log_prob_with_cp(full, total_len, response_len)
            if cp_size > 1
            else full
        )
    rollout_data["advantages"] = advantages
    rollout_data["returns"] = [advantage.clone() for advantage in advantages]
    rollout_data["turn_credit_selected_channel_empty"] = selected_channel_empty
