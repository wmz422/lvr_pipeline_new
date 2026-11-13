"""Exercise four-stage Flow, export and greedy inference with real tiny Qwen2.5-VL."""
import torch
from transformers import Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration

import test_qwen3_training as qwen3_tests


class Qwen25FourStageTrainingTests(qwen3_tests.Qwen3TrainingTests):
    backbone_class = Qwen2_5_VLForConditionalGeneration

    @staticmethod
    def model_config():
        return Qwen2_5_VLConfig(
            text_config=dict(vocab_size=64, hidden_size=128, intermediate_size=256, num_hidden_layers=2,
                             num_attention_heads=2, num_key_value_heads=2,
                             rope_parameters={'rope_type': 'default', 'mrope_section': [8, 12, 12]}),
            vision_config=dict(depth=2, hidden_size=64, intermediate_size=128, num_heads=4,
                               out_hidden_size=128, patch_size=14, temporal_patch_size=2,
                               spatial_merge_size=2, window_size=56, fullatt_block_indexes=[1]),
            image_token_id=1, video_token_id=5, vision_start_token_id=2, vision_end_token_id=3)

    @staticmethod
    def batch(num_latents=8):
        batch = qwen3_tests.Qwen3TrainingTests.batch(num_latents)
        for branch in ('encoder', 'qa'):
            batch[branch]['pixel_values'] = torch.randn(8, 1176)
        return batch
