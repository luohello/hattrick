import unittest
from types import SimpleNamespace

import numpy as np
import torch

from utils.build_dataset_within_cluster import DM_Dataset_within_Cluster
from utils.robust_proj_utils import project_priority_gradients
from utils.training_utils import cumulative_fulfill_hinge
from frameworks.hattrick_system import Hattrick, _compact_directional_edge_features


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([1.0, 1.0]))


class OptimizedComponentsTest(unittest.TestCase):
    def test_compact_directional_features_preserve_endpoints(self):
        source = torch.tensor([[[1.0, 2.0], [3.0, 5.0]]])
        destination = torch.tensor([[[4.0, 8.0], [2.0, 1.0]]])
        endpoints = torch.stack((source, destination), dim=2)
        capacities = torch.tensor([[0.0, 2.0]])

        compact = _compact_directional_edge_features(endpoints, capacities)
        endpoint_sum, endpoint_difference = compact[..., :2], compact[..., 2:4]
        recovered_source = (endpoint_sum + endpoint_difference) / 2
        recovered_destination = (endpoint_sum - endpoint_difference) / 2

        torch.testing.assert_close(recovered_source, source)
        torch.testing.assert_close(recovered_destination, destination)
        torch.testing.assert_close(compact[..., -1], torch.tensor([[0.0, 2.0]]))

        zero_capacity = _compact_directional_edge_features(
            endpoints,
            torch.zeros_like(capacities),
        )
        self.assertTrue(torch.isfinite(zero_capacity).all())

    def test_compact_directional_edge_projection(self):
        props = SimpleNamespace(
            num_gnn_layers=3,
            num_transformer_layers=1,
            dropout=0.0,
            num_mlp1_hidden_layers=0,
            num_mlp2_hidden_layers=0,
            device=torch.device("cpu"),
            topo="geant",
            directional_edge_encoding=1,
            num_heads=0,
            violation=0,
        )
        model = Hattrick(props)
        self.assertEqual(model.gnn.output_dim, 11)
        self.assertEqual(model.gnn.edge_output_dim, 23)
        self.assertEqual(model.edge_projection.in_features, 23)
        self.assertEqual(model.edge_projection.out_features, 24)
        self.assertEqual(model.input_dim, 24)
        self.assertEqual(model.num_heads, 6)

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
