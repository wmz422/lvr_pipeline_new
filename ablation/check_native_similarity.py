#!/usr/bin/env python3
"""CPU checks on tiny random official LVR/Monet teacher models, never benchmark scores."""
import argparse
import importlib
from pathlib import Path
import sys


def tiny_config():
    from transformers import Qwen2_5_VLConfig
    return Qwen2_5_VLConfig(vocab_size=64, hidden_size=128, intermediate_size=256,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
        rope_scaling={"type": "mrope", "mrope_section": [8, 12, 12]},
        vision_config=dict(depth=2, hidden_size=64, intermediate_size=128, num_heads=4,
                           out_hidden_size=128, patch_size=14, temporal_patch_size=2,
                           spatial_merge_size=2, window_size=56, fullatt_block_indexes=[1]),
        image_token_id=1, video_token_id=5, vision_start_token_id=2, vision_end_token_id=3)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--method", choices=["lvr", "monet-teacher"], required=True)
    args = parser.parse_args()
    import torch
    torch.set_num_threads(2)
    torch.manual_seed(42)
    config = tiny_config()
    config._attn_implementation = "sdpa"
    if args.method == "lvr":
        sys.path.insert(0, str(args.sources / "lvr"))
        from src.model.qwen_lvr_model import QwenWithLVR
        from src.train.monkey_patch_forward_lvr import replace_qwen2_5_with_mixed_modality_forward_lvr
        from transformers import LogitsProcessor, LogitsProcessorList
        config.lvr_head = False
        config.latent_end_token = False
        config.loss_lvr_fct = "mse"
        config.lvr_start_id, config.lvr_id, config.lvr_end_id = 40, 41, 42
        replace_qwen2_5_with_mixed_modality_forward_lvr(inference_mode=True, lvr_head=False)
        model = QwenWithLVR(config).eval()
        model.generation_config.pad_token_id = 0
        model.generation_config.eos_token_id = 63
        vectors = []
        consumed = []

        def observe(module, inputs, kwargs):
            switch = kwargs.get("lvr_mode_switch")
            vector = kwargs.get("last_position_hidden_state")
            if switch is not None and bool(switch.any()) and vector is not None:
                vectors.append(vector[switch][0].detach().clone())

        def language_input(module, inputs, kwargs):
            if vectors:
                consumed.append(kwargs["inputs_embeds"][0, -1].detach().clone())

        class ForceStart(LogitsProcessor):
            def __call__(self, input_ids, scores):
                scores.fill_(-float("inf"))
                scores[:, 40 if input_ids.shape[-1] == 4 else 10] = 0
                return scores

        hook = model.register_forward_pre_hook(observe, with_kwargs=True)
        language_hook = model.model.register_forward_pre_hook(language_input, with_kwargs=True)
        with torch.inference_mode():
            output = model.generate(input_ids=torch.tensor([[7, 8, 9, 10]]),
                attention_mask=torch.ones(1, 4, dtype=torch.long), max_new_tokens=12,
                decoding_strategy="steps", lvr_steps=[8], do_sample=False,
                logits_processor=LogitsProcessorList([ForceStart()]))
        hook.remove()
        language_hook.remove()
        assert len(vectors) == 8, len(vectors)
        for observed, actual in zip(vectors, consumed[:8]):
            torch.testing.assert_close(observed, actual)
        with torch.inference_mode():
            features = model.model.get_image_features(torch.randn(4, 1176), torch.tensor([[1, 2, 2]]))
        assert torch.cat(features).shape == (1, 128)
        print("PASS LVR: 8 native consumed vectors match the actual recurrent inputs; native visual-target extraction")
    else:
        sys.path.insert(0, str(args.sources / "monet"))
        importlib.import_module("monet_qwen_model.apply_qwen2_5_monet")
        from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
        config.latent_token_id, config.latent_start_id, config.latent_end_id = 40, 41, 42
        config.answer_start_pattern = [9, 10]
        model = Qwen2_5_VLForConditionalGeneration(config).eval()
        ids = torch.tensor([[2, 1, 3, 9, 10, 41] + [40] * 8 + [42, 12]])
        positions = (ids[0] == 40).nonzero().flatten().tolist()
        with torch.inference_mode():
            output = model(input_ids=ids, attention_mask=torch.ones_like(ids),
                           pixel_values=torch.randn(4, 1176), image_grid_thw=torch.tensor([[1, 2, 2]]),
                           latent_mode=True, alignment_poss=[positions], loss_type=[],
                           output_latent_embeds=True, return_dict=True)
        assert output.latent_embeds[0].shape == (8, 128)
        assert torch.isfinite(output.latent_embeds[0]).all()
        assert (output.latent_embeds[0].norm(dim=-1) > 0).all()
        # Real benchmarks can have several helpers. Selecting the first eight must
        # preserve the first block when later image/latent blocks are still executed.
        multi_ids = torch.tensor([[2, 1, 3, 9, 10, 2, 1, 3, 41] + [40] * 8
                                 + [42, 2, 1, 3, 41] + [40] * 8 + [42, 12]])
        multi_positions = (multi_ids[0] == 40).nonzero().flatten().tolist()
        pixels = torch.randn(12, 1176)
        targets = []
        with torch.inference_mode():
            for selected in (multi_positions, multi_positions[:8]):
                out = model(input_ids=multi_ids, attention_mask=torch.ones_like(multi_ids),
                    pixel_values=pixels, image_grid_thw=torch.tensor([[1, 2, 2]] * 3),
                    latent_mode=True, alignment_poss=[selected], loss_type=[],
                    output_latent_embeds=True, return_dict=True)
                targets.append(out.latent_embeds[0])
        assert targets[0].shape == (16, 128) and targets[1].shape == (8, 128)
        torch.testing.assert_close(targets[0][:8], targets[1])
        print("PASS Monet teacher: 8 finite targets; multi-helper first-block extraction equals the full 16-target native forward")


if __name__ == "__main__":
    main()
