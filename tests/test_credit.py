import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from examples.offline_demo import example
from trata_slime_muse import turn_credit_synthesis_spans as credit

ROOT = Path(__file__).resolve().parents[1]


class CreditTests(unittest.TestCase):
    def test_signed_advantages_and_masks(self):
        contrast, _, _, _, advantages = example()
        self.assertAlmostEqual(contrast, -0.3)
        for actual, expected in zip(advantages, [0.24] * 3 + [0] + [0] * 4
                                    + [-0.18] * 3 + [0] + [-0.27] * 3 + [0]):
            self.assertAlmostEqual(actual, expected)

    def test_turn_weight_is_not_divided_by_length(self):
        turns = [credit.TurnSpan(0, 0, 1, "short", (), ()),
                 credit.TurnSpan(1, 1, 5, "long", (), ())]
        weights, _ = credit.build_response_token_weights(
            response_length=5, loss_mask=[1] * 5, turns=turns,
            raw_vectors=[[0.7, 0.7]], channel="success_credit")
        self.assertEqual(weights, [0.7] * 5)

    def test_empty_map_does_not_become_uniform(self):
        weights, stats = credit.build_response_token_weights(
            response_length=2, loss_mask=[1, 1],
            turns=[credit.TurnSpan(0, 0, 2, "neutral", (), ())],
            raw_vectors=[], channel="success_credit")
        self.assertEqual(weights, [0, 0])
        self.assertEqual(stats["uniform_fallback"], 0)

    def test_overlapping_spans_rejected(self):
        with self.assertRaisesRegex(credit.TurnCreditError, "overlap"):
            credit.build_response_token_weights(
                response_length=3, loss_mask=[1] * 3,
                turns=[credit.TurnSpan(0, 0, 2, "first", (), ()),
                       credit.TurnSpan(1, 1, 3, "second", (), ())],
                raw_vectors=[[0.5, 0.5]], channel="success_credit")

    def test_unowned_trainable_tokens_rejected(self):
        with self.assertRaisesRegex(credit.TurnCreditError, "not owned"):
            credit.build_response_token_weights(
                response_length=3, loss_mask=[1] * 3,
                turns=[credit.TurnSpan(0, 0, 2, "first", (), ())],
                raw_vectors=[[0.5]], channel="success_credit")

    def test_invalid_coefficients_rejected(self):
        for value in (-0.1, 1.1, float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaises(credit.TurnCreditError):
                credit.normalize_draw([value], [2])

    def test_sibling_group_mean(self):
        samples = [SimpleNamespace(metadata={}) for _ in range(3)]
        values = credit.anchored_objective_group_advantages(samples, [0.2, 0.6, 0.7])
        for actual, expected in zip(values, [-0.3, 0.1, 0.2]):
            self.assertAlmostEqual(actual, expected)

    def test_turn_payload_contains_tool_output(self):
        turn = credit.TurnSpan(0, 0, 2, "Inspect evidence", ("read report",), ("result",))
        self.assertEqual(turn.as_prompt_dict()["following_tool_outputs"], ["result"])

    def test_exported_prompts_match_code(self):
        for file, prompt in [("objective.md", credit.OBJECTIVE_JUDGE_SYSTEM_PROMPT),
                             ("attribution.md", credit.ATTRIBUTION_JUDGE_SYSTEM_PROMPT)]:
            self.assertEqual((ROOT / "prompts" / file).read_text(), prompt + "\n")

    def test_source_manifest(self):
        manifest = json.loads((ROOT / "SOURCE_MANIFEST.json").read_text())
        for entry in manifest["files"]:
            actual = hashlib.sha256((ROOT / entry["path"]).read_bytes()).hexdigest()
            self.assertEqual(actual, entry["sha256"])


class PayloadTests(unittest.IsolatedAsyncioTestCase):
    async def test_attribution_receives_completed_trajectory_and_final_answer(self):
        seen = []
        turns = [credit.TurnSpan(0, 0, 2, "Read useful evidence", ("read report",),
                                 ("Relevant disclosure",)),
                 credit.TurnSpan(1, 2, 4, "Commit a wrong inference", (), ())]

        async def fake_request(config, prompt, system_prompt):
            seen.append((system_prompt, json.loads(prompt)))
            if system_prompt == credit.OBJECTIVE_JUDGE_SYSTEM_PROMPT:
                return json.dumps({"schema_version": credit.SCHEMA_VERSION,
                                   "objective_score": 0.2,
                                   "synthesis_assessment": "Evidence was misinterpreted.",
                                   "prediction_assessment": "Incorrect outcome.",
                                   "decisive_error": "Final inference contradicted the evidence."}), 1
            return json.dumps({
                "schema_version": credit.SCHEMA_VERSION, "turn_count": 2,
                "success_attributions": [{"turn_index": 0, "success_credit": 0.8,
                                          "reason": "Found the relevant disclosure."}],
                "failure_attributions": [{"turn_index": 1, "failure_blame": 0.9,
                                          "failure_kind": "answer_error",
                                          "reason": "Misinterpreted the disclosure."}],
                "tool_call_attributions": [],
                "success_empty_reason": None, "failure_empty_reason": None,
            }), 1

        config = credit.TurnCreditConfig(judge_samples=1, score_repair_calls=1,
                                         attribution_repair_calls=0)
        with patch.object(credit, "_request_judge_content", side_effect=fake_request):
            success, failure, score, _ = await credit.assign_turn_credit(
                instruction="Answer the synthetic forecasting question.",
                grading_rubric="Use the relevant disclosure.",
                final_answer="Synthetic final answer.",
                objective_context={"prediction_correct": False},
                response_length=4, loss_mask=[1, 0, 1, 0], turns=turns,
                config=config)
        self.assertEqual(len(seen), 2)
        attribution = seen[1][1]
        self.assertEqual(attribution["agent_turns"], [t.as_prompt_dict() for t in turns])
        self.assertEqual(attribution["final_answer_md"], "Synthetic final answer.")
        self.assertEqual(attribution["objective_score"], 0.2)
        self.assertEqual(success, [0.8, 0, 0, 0])
        self.assertEqual(failure, [0, 0, 0.9, 0])
        self.assertEqual(score, 0.2)


if __name__ == "__main__":
    unittest.main()
