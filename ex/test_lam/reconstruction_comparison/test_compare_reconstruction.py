from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch

from compare_reconstruction import (
    _completed_indices,
    _feature_metrics,
    _merge_feature_pair,
    _merged_split_sizes,
    _reshape_pair_pixel_values,
    _select_indices,
)


class MeanMerger(torch.nn.Module):
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return features.reshape(-1, 4, features.shape[-1]).mean(dim=1)


class CompareReconstructionTests(unittest.TestCase):
    def test_reshape_pair_pixel_values(self) -> None:
        pixels = torch.arange(2 * 16 * 3).reshape(32, 3)
        grid = torch.tensor([[1, 4, 4], [1, 4, 4]])
        pair = _reshape_pair_pixel_values(pixels, grid)
        self.assertEqual(tuple(pair.shape), (1, 2, 16, 3))
        self.assertTrue(torch.equal(pair.reshape(32, 3), pixels))

    def test_reshape_rejects_different_grids(self) -> None:
        with self.assertRaisesRegex(ValueError, "different grids"):
            _reshape_pair_pixel_values(
                torch.zeros(32, 3),
                torch.tensor([[1, 4, 4], [1, 2, 8]]),
            )

    def test_merge_and_split_pair(self) -> None:
        pair = torch.arange(2 * 16 * 2, dtype=torch.float32).reshape(2, 16, 2)
        grid = torch.tensor([[1, 4, 4], [1, 4, 4]])
        merged = _merge_feature_pair(MeanMerger(), pair, grid, spatial_merge_size=2)
        self.assertEqual([tuple(item.shape) for item in merged], [(4, 2), (4, 2)])
        self.assertEqual(_merged_split_sizes(grid, 2), [4, 4])

    def test_feature_metrics_identity(self) -> None:
        tensor = torch.randn(8, 4)
        metrics = _feature_metrics(tensor, tensor)
        self.assertEqual(metrics["mse"], 0.0)
        self.assertAlmostEqual(metrics["cosine_similarity"], 1.0, places=6)

    def test_index_selection_and_resume_file(self) -> None:
        self.assertEqual(list(_select_indices(total=10, start=3, limit=2)), [3, 4])
        self.assertEqual(list(_select_indices(total=10, start=8, limit=-1)), [8, 9])
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "results.jsonl"
            output.write_text(json.dumps({"index": 3}) + "\n", encoding="utf-8")
            self.assertEqual(_completed_indices(output), {3})


if __name__ == "__main__":
    unittest.main()
