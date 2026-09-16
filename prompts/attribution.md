You assign sparse signed causal attribution for one completed research trajectory.

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

