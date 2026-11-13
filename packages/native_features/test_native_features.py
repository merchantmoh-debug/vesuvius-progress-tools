import unittest
import numpy as np
import torch
from phase_artifacts import periodicity, constant_token_control, sample_block_means
from point_queries import query_points


class NativeFeatureTests(unittest.TestCase):
    def test_periodic_control_distinguishes_texture_from_constant_output(self):
        # Synthetic controlled decoder, not evidence for performance on a scroll.
        class Decoder(torch.nn.Module):
            def forward(self, features):
                template = torch.arange(8 ** 3, dtype=features.dtype).reshape(8, 8, 8) / 512
                return template.repeat(2, 3, 4)[None, None] + features.mean()
        decoder = Decoder().eval()
        result = constant_token_control(decoder, torch.randn(1, 4, 2, 3, 4))
        self.assertEqual(result['constant_token_repeat_max_error'], 0)
        self.assertGreater(result['phase_standard_deviation'], .2)
        flat = periodicity(np.full((16, 24, 32), 2.0))
        self.assertEqual(flat['phase_standard_deviation'], 0)
        noisy = np.tile(np.arange(8 ** 3).reshape(8, 8, 8), (2, 3, 4))
        noisy[10, 10, 10] += 1
        self.assertEqual(periodicity(noisy)['constant_token_repeat_max_error'], 1)

    def test_physical_block_centres_and_common_support(self):
        z, y, x = torch.meshgrid(*(torch.arange(n, dtype=torch.float64) for n in (16, 24, 32)), indexing='ij')
        logits = (z + 2 * y + 3 * x)[None, None] / 100
        points = torch.tensor([[3.5, 3.5, 3.5], [8.2, 12.8, 20.1], [11.5, 19.5, 27.5], [0, 1, 2]], dtype=torch.float64)
        result = sample_block_means(logits, points)
        self.assertEqual(result['common'].tolist(), [True, True, True, False])
        expected = ((points[:, 0] + 2 * points[:, 1] + 3 * points[:, 2]) / 100).sigmoid()
        torch.testing.assert_close(result['unpooled'], expected, rtol=0, atol=1e-15)
        torch.testing.assert_close(result['pooled'][:3], expected[:3], rtol=0, atol=1e-15)
        with self.assertRaises(ValueError): sample_block_means(logits, [[16, 0, 0]])

    def test_point_query_partition_preserves_order_and_tail(self):
        class Head(torch.nn.Module):
            def forward_points(self, tokens, skips, points, shape):
                return {'ink': torch.cat([points, points[..., :1] + tokens.sum()], -1)}
        head = Head().eval()
        points = np.arange(45, dtype=np.float32).reshape(15, 3) / 10
        tokens = torch.tensor([1.0, 2.0])
        whole = query_points(head, tokens, None, points, (8, 8, 8), chunk_size=100)
        split = query_points(head, tokens, None, points, (8, 8, 8), chunk_size=4)
        np.testing.assert_array_equal(split, whole)
        with self.assertRaises(ValueError): query_points(head, tokens, None, [[9, 0, 0]], (8, 8, 8))


if __name__ == '__main__': unittest.main()
