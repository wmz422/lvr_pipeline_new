import unittest
from transformers.trainer_pt_utils import DistributedLengthGroupedSampler
from tuning.student import StressSampler


class ProfilingSamplerTests(unittest.TestCase):
    def test_stress_prefix_preserves_real_sampler_order_and_epoch(self):
        dataset = list(range(512))
        for batch in (2, 4, 8, 16, 32):
            for rank in range(4):
                original = DistributedLengthGroupedSampler(batch_size=batch * 4,
                    num_replicas=4, rank=rank, seed=42, lengths=[i % 41 + 1 for i in dataset])
                wrapped = StressSampler(original, dataset, [3, 19], 32)
                for epoch in (0, 1):
                    wrapped.set_epoch(epoch)
                    expected, actual = list(original), list(wrapped)
                    self.assertEqual(actual[:32], [3] * 32)
                    self.assertEqual(actual[32:64], [19] * 32)
                    self.assertEqual(actual[64:], expected[64:])
                    self.assertEqual(len(actual), len(wrapped))
                    self.assertEqual(wrapped.epoch, original.epoch)


if __name__ == '__main__':
    unittest.main()
