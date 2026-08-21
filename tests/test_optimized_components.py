import unittest
from types import SimpleNamespace

import numpy as np
import torch

from utils.build_dataset_within_cluster import DM_Dataset_within_Cluster
from utils.robust_proj_utils import project_priority_gradients
from utils.training_utils import cumulative_fulfill_hinge


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([1.0, 1.0]))


class OptimizedComponentsTest(unittest.TestCase):
    def test_priority_projection_is_orthogonal(self):
        model = TinyModel()
        high = model.weight[0]
        low = model.weight[0] + model.weight[1]
        final, _, raw = project_priority_gradients(model, [high, low])
        low_component = final - raw[0]
        self.assertLess(abs(torch.dot(raw[0], low_component).item()), 1e-6)
        self.assertTrue(torch.isfinite(final).all())

    def test_causal_uncertainty_does_not_use_current_actual(self):
        dataset = object.__new__(DM_Dataset_within_Cluster)
        dataset.props = SimpleNamespace(uncertainty_scale=1.0, uncertainty_ema=0.5)
        for priority in (1, 2, 3):
            setattr(dataset, f"list_tms{priority}", [np.array([[4.0]], dtype=np.float32), np.array([[9.0]], dtype=np.float32)])
            setattr(dataset, f"list_tms{priority}_pred", [np.array([[2.0]], dtype=np.float32), np.array([[3.0]], dtype=np.float32)])
        dataset._apply_causal_uncertainty_margin()
        self.assertEqual(dataset.list_tms1_pred[0].item(), 2.0)
        self.assertEqual(dataset.list_tms1_pred[1].item(), 4.0)

    def test_fulfill_hinge_penalizes_slo_violation(self):
        admitted = torch.tensor([[8.0, 0.0]], requires_grad=True)
        demand = torch.tensor([[10.0, 0.0]])
        loss = cumulative_fulfill_hinge([admitted], [demand], 0.9, 1.0)
        self.assertAlmostEqual(loss.item(), 0.1, places=6)
        loss.backward()
        self.assertLess(admitted.grad[0, 0].item(), 0.0)


if __name__ == "__main__":
    unittest.main()
