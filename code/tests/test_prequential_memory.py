from __future__ import annotations

import hashlib
import json
import unittest

import numpy as np

from sigla_exp.prequential_birch import OnlineBirchMemory
from sigla_exp.prequential_memory import MemoryConfig, OnlinePrototypeMemory
from sota_compare.run_longtail_prequential import (
    composite_order,
    fast_features,
    stable_hash,
)
from sota_compare.run_representation_gate_followup import (
    feature_vector,
    fit_feature_context,
    observed_component_key,
)
import sigla_exp.longtail_bench as LT


def state_digest(memory) -> str:
    return hashlib.sha256(json.dumps(memory.state(), sort_keys=True).encode()).hexdigest()


class PrototypeMemoryTest(unittest.TestCase):
    def test_first_query_is_not_reuse_and_future_hit_is_reuse(self):
        memory = OnlinePrototypeMemory(
            MemoryConfig(name="flat", hierarchical=False, radius=1.0)
        )
        vector = np.asarray([0.0, 0.0])
        first = memory.process(vector, ("a",), "type_a", True, step=0)
        second = memory.process(vector, ("a",), "type_a", True, step=1)

        self.assertTrue(first.queried)
        self.assertFalse(first.autonomous_reuse)
        self.assertEqual(first.action, "query_create")
        self.assertFalse(second.queried)
        self.assertTrue(second.autonomous_reuse)
        self.assertEqual(second.pred_label, "type_a")

    def test_autonomous_branch_does_not_read_oracle_label(self):
        memory = OnlinePrototypeMemory(
            MemoryConfig(name="flat", hierarchical=False, radius=1.0)
        )
        vector = np.asarray([0.0, 0.0])
        memory.process(vector, ("a",), "type_a", True, step=0)
        decision = memory.process(vector, ("a",), "shuffled_test_label", True, step=1)

        self.assertFalse(decision.queried)
        self.assertEqual(decision.pred_label, "type_a")

    def test_guard_requires_two_verified_queries(self):
        memory = OnlinePrototypeMemory(
            MemoryConfig(
                name="guard",
                hierarchical=True,
                radius=1.0,
                confirm_k=2,
                reuse_margin=0.8,
            )
        )
        vector = np.asarray([0.0, 0.0])
        first = memory.process(vector, ("family",), "type_a", True, step=0)
        second = memory.process(vector, ("family",), "type_a", True, step=1)
        locked = memory.predict_locked(vector, ("family",))

        self.assertEqual(first.action, "query_create_tentative")
        self.assertTrue(second.queried)
        self.assertEqual(second.action, "query_confirm")
        self.assertEqual(memory.committed_count, 1)
        self.assertEqual(locked.pred_label, "type_a")

    def test_merge_reduces_active_not_historical_clusters(self):
        memory = OnlinePrototypeMemory(
            MemoryConfig(
                name="merge",
                hierarchical=True,
                radius=1.0,
                merge_radius=1.2,
            )
        )
        memory.process(np.asarray([0.0]), ("family",), "type_a", True, step=0)
        memory.process(np.asarray([1.1]), ("family",), "type_a", True, step=1)

        self.assertEqual(memory.active_count, 1)
        self.assertEqual(memory.historical_clusters, 2)
        self.assertEqual(len(memory.merge_events), 1)
        self.assertEqual(memory.merge_precision, 1.0)

    def test_safe_merge_blocks_queried_label_conflict(self):
        memory = OnlinePrototypeMemory(
            MemoryConfig(
                name="safe_merge",
                hierarchical=True,
                radius=1.0,
                merge_radius=1.2,
                block_label_conflict=True,
            )
        )
        memory.process(np.asarray([0.0]), ("family",), "type_a", True, step=0)
        memory.process(np.asarray([1.1]), ("family",), "type_b", True, step=1)
        self.assertEqual(memory.active_count, 2)
        self.assertEqual(len(memory.merge_events), 0)

    def test_hierarchical_global_fallback_recovers_noisy_key(self):
        memory = OnlinePrototypeMemory(
            MemoryConfig(
                name="fallback",
                hierarchical=True,
                radius=1.0,
                fallback_global=True,
            )
        )
        vector = np.asarray([0.0, 0.0])
        memory.process(vector, ("key_a",), "type_a", True, step=0)
        locked = memory.predict_locked(vector, ("noisy_key_b",))
        self.assertEqual(locked.pred_label, "type_a")
        self.assertTrue(locked.autonomous_reuse)

    def test_pending_cluster_does_not_shadow_committed_reuse(self):
        memory = OnlinePrototypeMemory(
            MemoryConfig(
                name="guard_fallback",
                hierarchical=True,
                radius=1.0,
                confirm_k=2,
                reuse_margin=0.85,
                fallback_global=True,
            )
        )
        memory.process(np.asarray([0.0]), ("key_a",), "type_a", True, step=0)
        memory.process(np.asarray([0.0]), ("key_a",), "type_a", True, step=1)
        pending = memory.process(np.asarray([0.9]), ("key_b",), "type_b", True, step=2)
        locked = memory.predict_locked(np.asarray([0.8]), ("key_b",))

        self.assertEqual(pending.action, "query_create_tentative")
        self.assertEqual(locked.pred_label, "type_a")
        self.assertTrue(locked.autonomous_reuse)

    def test_locked_prediction_does_not_mutate_full_state(self):
        memory = OnlinePrototypeMemory(
            MemoryConfig(name="flat", hierarchical=False, radius=1.0)
        )
        vector = np.asarray([0.0, 0.0])
        memory.process(vector, ("a",), "type_a", True, step=0)
        before = state_digest(memory)
        memory.predict_locked(vector, ("a",))
        after = state_digest(memory)
        self.assertEqual(before, after)


class BirchMemoryTest(unittest.TestCase):
    def test_birch_query_then_locked_reuse(self):
        memory = OnlineBirchMemory(threshold=0.5)
        vector = np.asarray([0.0, 0.0])
        first = memory.process(vector, "type_a", True, step=0)
        before = state_digest(memory)
        locked = memory.predict_locked(vector)
        after = state_digest(memory)

        self.assertTrue(first.queried)
        self.assertEqual(locked.pred_label, "type_a")
        self.assertEqual(before, after)

    def test_birch_never_queries_and_discards_existing_cluster_label(self):
        memory = OnlineBirchMemory(threshold=1.0)
        for step in range(10):
            memory.process(np.asarray([0.0]), "type_a", True, step=step)
        decision = memory.process(np.asarray([1.5]), "type_b", True, step=10)
        locked = memory.predict_locked(np.asarray([1.5]))

        self.assertFalse(decision.queried)
        self.assertEqual(decision.pred_label, "type_a")
        self.assertEqual(locked.pred_label, "type_a")


class DataProtocolTest(unittest.TestCase):
    def test_composite_order_is_complete_deterministic_and_pair_balanced(self):
        catalog = LT.generate_taxonomy(216)
        composites = [spec for spec in catalog if len(spec["components"]) == 2]
        first = composite_order(0, composites)
        second = composite_order(0, composites)

        self.assertEqual(len(first), 180)
        self.assertEqual([spec["name"] for spec in first], [spec["name"] for spec in second])
        self.assertEqual(len({spec["name"] for spec in first}), 180)
        first_fifteen = {tuple(sorted(spec["components"])) for spec in first[:15]}
        self.assertEqual(len(first_fifteen), 15)

    def test_fast_features_match_reference(self):
        rng = np.random.default_rng(7)
        mu, sd = LT.normal_stats(np.random.default_rng(8), n=12)
        for spec in [None, LT.generate_taxonomy(6)[0], LT.generate_taxonomy(216)[40]]:
            window = LT.make_window(spec, rng)
            expected = LT.features(window, mu, sd)
            actual = fast_features(window, mu, sd)
            np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)

    def test_stable_hash_is_order_independent_for_dicts(self):
        self.assertEqual(stable_hash({"a": 1, "b": 2}), stable_hash({"b": 2, "a": 1}))


class RepresentationFollowupTest(unittest.TestCase):
    def test_observable72_feature_is_finite_and_expected_dimension(self):
        rng = np.random.default_rng(930)
        context = fit_feature_context([LT.make_window(None, rng) for _ in range(8)])
        spec = LT.generate_taxonomy(216)[40]
        vector = feature_vector("observable72", LT.make_window(spec, rng), context)

        self.assertEqual(vector.shape, (72,))
        self.assertTrue(np.all(np.isfinite(vector)))

    def test_observed_component_key_uses_window_evidence(self):
        rng = np.random.default_rng(931)
        context = fit_feature_context([LT.make_window(None, rng) for _ in range(8)])
        spec = next(
            item
            for item in LT.generate_taxonomy(216)
            if tuple(item["components"]) == ("spike", "level_shift")
            and item["severity"] == "strong"
        )
        key = observed_component_key(LT.make_window(spec, rng), context, threshold=1.0)

        self.assertIsInstance(key, tuple)
        self.assertTrue(key)


if __name__ == "__main__":
    unittest.main()
