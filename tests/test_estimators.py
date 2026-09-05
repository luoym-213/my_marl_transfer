import os
import tempfile
import unittest

import numpy as np
import torch

from estimators.base import StaleEstimator
from estimators.commdrop import (
    CommDropEstimator,
    CommDropModel,
    FourierTimeEncoding,
)
from estimators.kalman import ConstantVelocityKalmanEstimator


def normalization():
    return {
        "state": {
            "mean": np.zeros(4, dtype=np.float32),
            "std": np.ones(4, dtype=np.float32),
        },
        "goal": {
            "mean": np.zeros(2, dtype=np.float32),
            "std": np.ones(2, dtype=np.float32),
        },
        "task_progress": {
            "mean": np.zeros(1, dtype=np.float32),
            "std": np.ones(1, dtype=np.float32),
        },
    }


class KalmanEstimatorTest(unittest.TestCase):
    def make_estimator(self):
        return ConstantVelocityKalmanEstimator(
            2,
            process_covariance=np.eye(4) * 1e-12,
            observation_covariance=np.eye(2) * 1e-8,
            initial_covariance=np.diag([1e-8, 1e-8, 1e-8, 1e-8]),
        )

    def test_constant_velocity_exact_extrapolation_without_packets(self):
        initial = np.array([
            [0.0, 0.0, 1.0, -0.5],
            [0.2, -0.1, -0.25, 0.75],
        ], dtype=np.float32)
        estimator = self.make_estimator()
        estimator.reset(initial)
        result = estimator.predict(10, initial)
        np.testing.assert_allclose(
            result.mean[0, 1], [-0.05, 0.65, -0.25, 0.75], atol=1e-6
        )

    def test_delayed_packet_updates_at_source_time(self):
        initial = np.zeros((2, 4), dtype=np.float32)
        estimator = ConstantVelocityKalmanEstimator(
            2, np.eye(4) * 1e-6, np.eye(2) * 1e-10,
            initial_covariance=np.eye(4),
        )
        estimator.reset(initial)
        estimator.ingest_deliveries([{
            "receiver": 0, "sender": 1,
            "source_step": 2, "arrival_step": 5,
            "motion_state": np.array([0.2, 0.0, 99.0, 99.0]),
        }])
        result = estimator.predict(5, initial)
        self.assertEqual(result.source_steps[0, 1], 2)
        self.assertAlmostEqual(result.aoi[0, 1], 0.3, places=6)
        self.assertGreater(result.mean[0, 1, 0], 0.2)
        self.assertLess(abs(result.mean[0, 1, 1]), 1e-6)

    def test_query_is_non_persistent_and_covariance_is_positive(self):
        estimator = self.make_estimator()
        estimator.reset(np.zeros((2, 4), dtype=np.float32))
        means_before = estimator._means.copy()
        covariance_before = estimator._covariances.copy()
        result = estimator.predict(20, np.zeros((2, 4), dtype=np.float32))
        np.testing.assert_array_equal(estimator._means, means_before)
        np.testing.assert_array_equal(estimator._covariances, covariance_before)
        for covariance in result.covariance.reshape(-1, 4, 4):
            self.assertGreater(np.linalg.eigvalsh(covariance).min(), 0.0)

    def test_out_of_order_packet_is_discarded(self):
        estimator = self.make_estimator()
        estimator.reset(np.zeros((2, 4), dtype=np.float32))
        common = {"receiver": 0, "sender": 1, "arrival_step": 7}
        newest = dict(common, source_step=5, motion_state=np.ones(4))
        old = dict(common, source_step=3, motion_state=np.full(4, 9.0))
        estimator.ingest_deliveries([newest, old])
        self.assertEqual(estimator._source_steps[0, 1], 5)


class CommDropTest(unittest.TestCase):
    def make_model(self):
        torch.manual_seed(4)
        return CommDropModel(normalization(), {
            "hidden_dim": 16, "task_dim": 8,
        })

    def test_fourier_shape_and_zero_value(self):
        encoding = FourierTimeEncoding(16)
        value = encoding(torch.zeros(3))
        self.assertAlmostEqual(float(encoding.omega[0]), 0.1, places=6)
        self.assertAlmostEqual(float(encoding.omega[-1]), 5.0, places=6)
        self.assertEqual(tuple(value.shape), (3, 16))
        phase_value = encoding(torch.tensor([0.1]))
        self.assertAlmostEqual(float(phase_value[0, 0]), float(torch.sin(torch.tensor(1.0))), places=6)
        torch.testing.assert_close(value[:, :8], torch.zeros(3, 8))
        torch.testing.assert_close(value[:, 8:], torch.ones(3, 8))

    def test_shapes_no_self_loops_freshness_and_positive_variance(self):
        model = self.make_model()
        hidden = torch.zeros(3, 3, 16)
        states = torch.zeros(3, 3, 4)
        options = torch.zeros(3, 3, dtype=torch.long)
        goals = torch.zeros(3, 3, 2)
        progress = torch.zeros(3, 3)
        active = torch.ones(3, 3)
        source_steps = torch.tensor([
            [0.0, 1.0, 8.0],
            [0.0, 1.0, 8.0],
            [0.0, 1.0, 8.0],
        ])
        prediction = model.predict_from_memory(
            hidden, torch.zeros(3, 3, 16), options, goals, progress, active,
            source_steps, torch.ones(3, 3, dtype=torch.bool), 8,
            torch.zeros(3, 4),
        )
        self.assertEqual(tuple(prediction["mean"].shape), (3, 3, 4))
        self.assertEqual(tuple(prediction["variance"].shape), (3, 3, 4))
        self.assertTrue(bool((prediction["variance"] > 0).all()))
        diagonal = torch.diagonal(prediction["edge_gates"], dim1=1, dim2=2)
        torch.testing.assert_close(diagonal, torch.zeros_like(diagonal))
        # For ego 0, source 1 has AoI 0.7s and is excluded. Ego 1 itself
        # remains an allowed source regardless of packet AoI.
        self.assertFalse(bool(prediction["edge_mask"][0, 1, 2]))
        self.assertTrue(bool(prediction["edge_mask"][1, 1, 2]))
        torch.testing.assert_close(
            prediction["normalization_denominator"],
            prediction["incoming_weight"].clamp(min=1.0),
        )
        self.assertEqual(float(prediction["aoi_residual_gate"][0, 2]), 0.0)
        self.assertGreater(float(prediction["aoi_residual_gate"][0, 1]), 0.0)

    def test_runtime_query_does_not_write_history(self):
        estimator = CommDropEstimator(3, self.make_model())
        states = np.zeros((3, 4), dtype=np.float32)
        estimator.reset(states)
        before = estimator.memory_snapshot()
        result = estimator.predict(4, states)
        after = estimator.memory_snapshot()
        self.assertEqual(result.mean.shape, (3, 3, 4))
        for key in before:
            torch.testing.assert_close(before[key], after[key])

    def test_checkpoint_round_trip(self):
        model = self.make_model()
        checkpoint = {
            "kind": "commdrop",
            "num_agents": 3,
            "normalization": normalization(),
            "network_config": model.config,
            "model_state": model.state_dict(),
        }
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "model.pt")
            torch.save(checkpoint, path)
            restored = CommDropEstimator.from_checkpoint(path)
            states = np.zeros((3, 4), dtype=np.float32)
            restored.reset(states)
            self.assertEqual(restored.predict(1, states).mean.shape, (3, 3, 4))


if __name__ == "__main__":
    unittest.main()
