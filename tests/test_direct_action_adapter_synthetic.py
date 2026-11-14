"""Synthetic adapter contract tests: no saved demonstrations or private checkpoints.

These exercise the real converter's split/review/adapter API, not raw journal
validation, policy training, physical success, or independent review authority.
"""
import copy
import json
import unittest
from scripts import convert_terra_direct_continuous as converter


def synthetic_record(case_id="synthetic-a", decisions=32):
    return {
        "schema": converter.SCHEMA, "split": None, "identity_view": None,
        "training_eligibility": "unconfirmed",
        "lineage": {"case_id": case_id, "original_case_id": "synthetic-parent",
                    "initial_state_sha256": "a" * 64},
        "source_fingerprints": {key: "b" * 64 for key in
            ("assignment_sha256", "result_sha256", "public_trace_sha256")},
        "stages": [],
        "timeline": [{"kind": "teacher_robot_decision", "action_label": "OPEN",
            "direct_action": {"action": "OPEN", "requested_xyz_mm": None},
            "move_applied_target_deg": None, "derived_move": None,
            "public_input": {"window": [[float(row * 82 + col + i) for col in range(82)]
                for row in range(8)], "mask": [True] * 8}} for i in range(decisions)]}


def adapter(records):
    review = {"schema": converter.REVIEW_SCHEMA,
        "conclusion": "independent_review_selected_direct_action_training",
        "selected_originals": [converter.review_entry(r) for r in records]}
    # Test-only forged review: matching bytes are not proof of reviewer identity.
    raw = json.dumps(review).encode("utf-8")
    return converter.DirectActionAdapter(records,
        {r["lineage"]["case_id"]: "train" for r in records},
        review_bytes=raw, review_sha256=converter.digest(raw))


class SyntheticDirectContractTests(unittest.TestCase):
    def test_one_view_preserves_original_and_all_feature_columns(self):
        record = synthetic_record()
        before = copy.deepcopy(record)
        views = converter.split_and_identity_views([record], {"synthetic-a": "train"})
        self.assertEqual(len(views), 1)
        self.assertEqual(views[0]["identity_view"], [1, 2, 3, 4])
        self.assertEqual(views[0]["timeline"], record["timeline"])
        data = adapter([record])
        self.assertEqual((data.original_decision_count, data.identity_views_per_original, data.count), (32, 1, 32))
        batch = data.batch(list(range(32)))
        self.assertTrue(converter.torch.equal(batch["windows"], data.windows))
        self.assertEqual(batch["identity_permutations"], [[1, 2, 3, 4]] * 32)
        self.assertFalse(bool(batch["joint_regression_mask"].any()))
        self.assertIsNone(batch["joint_regression_targets"])
        self.assertEqual(record, before)
        with self.assertRaisesRegex(ValueError, "invalid batch indices"):
            data.batch([32])

    def test_variable_length_original_boundaries(self):
        data = adapter([synthetic_record(decisions=3), synthetic_record("synthetic-b", 5)])
        self.assertEqual(data.count, 8)
        batch = data.batch([0, 2, 3, 7])
        self.assertEqual(batch["case_ids"], ["synthetic-a"] * 2 + ["synthetic-b"] * 2)
        self.assertEqual(batch["distinct_parent_original_count"], 1)

    def test_parent_or_initial_state_cannot_leak_across_splits(self):
        a = synthetic_record()
        for shared in ("original_case_id", "initial_state_sha256"):
            b = synthetic_record("synthetic-b")
            different = "initial_state_sha256" if shared == "original_case_id" else "original_case_id"
            b["lineage"][different] = "synthetic-other"
            with self.subTest(shared=shared), self.assertRaisesRegex(ValueError, "cross partitions"):
                converter.split_and_identity_views([a, b], {"synthetic-a": "train", "synthetic-b": "holdout"})

    def test_review_digest_rejects_unreviewed_mutation(self):
        record = synthetic_record()
        entry = converter.review_entry(record)
        record["timeline"][0]["public_input"]["window"][0][0] += 1
        raw = json.dumps({"schema": converter.REVIEW_SCHEMA,
            "conclusion": "independent_review_selected_direct_action_training",
            "selected_originals": [entry]}).encode("utf-8")
        with self.assertRaisesRegex(ValueError, "source fingerprint differs"):
            converter.DirectActionAdapter([record], {"synthetic-a": "train"},
                review_bytes=raw, review_sha256=converter.digest(raw))


if __name__ == "__main__":
    unittest.main()
