"""Native latent extractors. Imports are local to keep incompatible runtimes separate."""
from __future__ import annotations

import hashlib
import importlib
import json
import os
from pathlib import Path
import sys

from common import PROJECT, model_item, safe_path

MONET_SYSTEM = ("You are a helpful multimodal assistant. You are required to answer the question "
                "based on the image provided. Put your final answer in \\boxed{}.")
DIRECT_FOOTER = "Return only the final answer inside <answer>...</answer>, without an explanation."


def messages(row, suite, *, teacher=False, monet=False):
    question = row["question"].rstrip()
    if question.endswith(DIRECT_FOOTER):
        question = question[:-len(DIRECT_FOOTER)].rstrip()
    if monet:
        # Remove only the shared answer-format footer; retain task rules and choices.
        for footer in ("Return only the final answer inside \\boxed{}.",
                       "Return only the option letter inside \\boxed{}.",
                       "Answer with only the option letter.",
                       "Answer with the option's letter from the given choices directly."):
            if question.endswith(footer):
                question = question[:-len(footer)].rstrip()
    content = [{"type": "image", "image": str(safe_path(suite, image["path"]))} for image in row["images"]]
    content.append({"type": "text", "text": question})
    result = []
    if monet:
        result.append({"role": "system", "content": MONET_SYSTEM})
    result.append({"role": "user", "content": content})
    if teacher:
        result.append({"role": "assistant", "content": [
            {"type": "image", "image": str(safe_path(suite, im["path"])), "max_pixels": 262144}
            for im in row["helper_images"]]})
    return result


def processor_for(model_path, native_path):
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    native = AutoProcessor.from_pretrained(native_path, local_files_only=True)
    processor.image_processor = native.image_processor
    # Added latent tokens/chat template must come from the released checkpoint.
    for key in ("min_pixels", "max_pixels"):
        processor.tokenizer.init_kwargs.pop(key, None)
    return processor


def processor_inputs(processor, chat, device, *, text=None):
    from qwen_vl_utils import process_vision_info
    if text is None:
        text = processor.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    images, _ = process_vision_info(chat)
    batch = processor(text=[text], images=images, return_tensors="pt", padding=True)
    return batch.to(device), text


def context_check(model, batch, extra=0):
    config = getattr(model.config, "text_config", model.config)
    if batch["input_ids"].shape[-1] + extra > config.max_position_embeddings:
        raise ValueError("Context overflow; inputs will not be truncated")


class Ours:
    def __init__(self, assets, suite, protocol, device, kind, runtime):
        import torch
        from transformers import AutoProcessor
        sys.path.insert(0, str(PROJECT / "src"))
        from lvr.data.crop_qa import CropQACollator
        from lvr.models.latent_prediction import LatentPrediction
        self.torch, self.suite, self.device, self.kind = torch, suite, device, kind
        base = assets / "models/qwen3-base"
        self.model = LatentPrediction.from_checkpoint(assets / "models/ours/joint/last_full.ckpt", device,
            source_checkpoint=assets / "models/ours/teacher/student_source.ckpt", base_model_path=base)
        network = self.model.network
        if (network.qa.config.model_type, network.hparams.num_latents, network.hparams.latent_dim,
                network.hparams.gaussian, network.hparams.encoder_includes_original) != ("qwen3_vl", 8, 512, False, True):
            raise ValueError("Expected the deterministic Qwen3-VL-4B 8x512 interface")
        processor = AutoProcessor.from_pretrained(base, local_files_only=True)
        processor.tokenizer.padding_side = "right"
        kwargs = dict(processor=processor, image_root=suite, num_latents=8,
            original_pixels=None, crop_pixels=protocol["helper_pixels"],
            max_length=protocol["inference"]["max_input_tokens"], system_prompt=protocol["inference"]["system_prompt"],
            include_answers=False, include_descriptions=False, include_reconstruction=False)
        if kind == "teacher":
            kwargs.update(include_encoder=True, encoder_uses_question=network.hparams.encoder_uses_question,
                          encoder_includes_original=True, encoder_type=network.hparams.encoder_type)
        else:
            kwargs.update(include_encoder=False)
        self.collator = CropQACollator(**kwargs)
        self.info = {"definition": "frozen_teacher_mean" if kind == "teacher" else "student_prediction_before_projector",
                     "position_type": "learned_latent_slot", "checkpoint_step": self.model.checkpoint_step,
                     "prediction_loops": self.model.prediction_loop_count(),
                     "teacher_uses_question": bool(network.hparams.encoder_uses_question),
                     "image_processor": processor.image_processor.to_dict()}

    def extract(self, row):
        torch = self.torch
        teacher = self.kind == "teacher"
        batch = self.collator([model_item(row, self.suite, teacher=teacher)])["encoder" if teacher else "qa"]
        batch = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()}
        context_check(self.model.network.encoder if teacher else self.model.network.qa, batch)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            value = self.model.network.encode(batch)[0] if teacher else self.model.predict_latents(batch)
        return value[0], {"native_positions": value.shape[1], "input_tokens": batch["input_ids"].shape[-1]}


class LVR:
    def __init__(self, assets, suite, protocol, device, kind, runtime):
        import torch
        from transformers import AutoConfig
        sys.path.insert(0, str(assets / "sources/lvr"))
        from src.train.monkey_patch_forward_lvr import replace_qwen2_5_with_mixed_modality_forward_lvr
        from src.model.qwen_lvr_model import QwenWithLVR
        path = assets / "models/lvr"
        config = AutoConfig.from_pretrained(path, local_files_only=True)
        replace_qwen2_5_with_mixed_modality_forward_lvr(inference_mode=True, lvr_head=config.lvr_head)
        self.model = QwenWithLVR.from_pretrained(path, torch_dtype=torch.bfloat16,
            attn_implementation="sdpa", local_files_only=True).to(device).eval().requires_grad_(False)
        self.processor = processor_for(path, assets / "models/qwen25-processor")
        self.torch, self.suite, self.device, self.kind = torch, suite, device, kind
        self.info = {"definition": "projected_helper_visual_targets" if kind == "teacher" else "consumed_native_recurrent_latent",
                     "position_type": "ordered_visual_patch" if kind == "teacher" else "autoregressive_latent_step",
                     "decoding_strategy": "steps", "lvr_steps": [8], "max_new_tokens": 512,
                     "attention": "sdpa", "image_processor": self.processor.image_processor.to_dict()}

    def extract(self, row):
        torch = self.torch
        if self.kind == "teacher":
            from PIL import Image
            images = []
            for item in row["helper_images"]:
                with Image.open(safe_path(self.suite, item["path"])) as image:
                    images.append(image.convert("RGB"))
            # The official training target is the vision merger output, not an LLM hidden state.
            batch = self.processor.image_processor(images=images, max_pixels=262144, return_tensors="pt").to(self.device)
            with torch.inference_mode():
                values = self.model.model.get_image_features(batch["pixel_values"].to(self.model.dtype), batch["image_grid_thw"])
                values = torch.cat(values, dim=0) if isinstance(values, (list, tuple)) else values
            return values[:8], {"native_positions": len(values), "image_grid_thw": batch["image_grid_thw"].tolist()}
        chat = messages(row, self.suite)
        batch, prompt = processor_inputs(self.processor, chat, self.device)
        context_check(self.model, batch, 512)
        vectors = []
        native_count = 0

        def observe(module, args, kwargs):
            nonlocal native_count
            switch = kwargs.get("lvr_mode_switch")
            vector = kwargs.get("last_position_hidden_state")
            if switch is not None and bool(switch.any()) and vector is not None:
                native_count += 1
                if len(vectors) < 8:
                    vectors.append(vector[switch][0].detach().float().cpu().clone())

        handle = self.model.register_forward_pre_hook(observe, with_kwargs=True)
        try:
            with torch.inference_mode():
                output = self.model.generate(**batch, do_sample=False, max_new_tokens=512,
                                             decoding_strategy="steps", lvr_steps=[8])
        finally:
            handle.remove()
        value = torch.stack(vectors) if vectors else torch.empty((0, self.model.config.hidden_size))
        generated = output[0, batch["input_ids"].shape[-1]:]
        return value, {"native_positions": native_count, "prompt": prompt, "generated_tokens": len(generated),
                       "generation": self.processor.tokenizer.decode(generated, skip_special_tokens=False)}


def teacher_text(processor, chat):
    """Official precompute layout: auxiliary image, then eight continuous latent pads."""
    text = processor.apply_chat_template(chat, tokenize=False, add_generation_prompt=False)
    marker = "<|im_start|>assistant\n"
    if text.count(marker) != 1:
        raise ValueError("Unexpected Monet teacher chat template")
    prefix, assistant = text.split(marker, 1)
    image = "<|vision_start|><|image_pad|><|vision_end|>"
    block = "<abs_vis_token>" + "<abs_vis_token_pad>" * 8 + "</abs_vis_token>"
    if assistant.count(image) != len(chat[-1]["content"]):
        raise ValueError("Teacher helper layout differs from the official image-placeholder layout")
    return prefix + marker + assistant.replace(image, image + block)


class MonetTeacher:
    def __init__(self, assets, suite, protocol, device, kind, runtime):
        import torch
        sys.path.insert(0, str(assets / "sources/monet"))
        importlib.import_module("monet_qwen_model.apply_qwen2_5_monet")
        from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
        path = assets / "models/monet-teacher/stage2"
        self.processor = processor_for(path, assets / "models/qwen25-processor")
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(path, torch_dtype=torch.bfloat16,
            attn_implementation="sdpa", local_files_only=True).to(device).eval().requires_grad_(False)
        tokenizer = self.processor.tokenizer
        for name, token in (("latent_token_id", "<abs_vis_token_pad>"), ("latent_start_id", "<abs_vis_token>"),
                            ("latent_end_id", "</abs_vis_token>")):
            ids = tokenizer(token, add_special_tokens=False)["input_ids"]
            if len(ids) != 1:
                raise ValueError(f"Missing official Monet token: {token}")
            setattr(self.model.config, name, ids[0])
        self.model.config.answer_start_pattern = tokenizer("<|im_start|>assistant", add_special_tokens=False)["input_ids"]
        self.torch, self.suite, self.device = torch, suite, device
        self.info = {"definition": "sft_stage2_auxiliary_image_conditioned_latent_targets",
                     "position_type": "first_helper_latent_block", "latent_size": 8,
                     "attention": "sdpa_2d_causal_as_official_precompute_forward",
                     "image_processor": self.processor.image_processor.to_dict()}

    def extract(self, row):
        torch = self.torch
        chat = messages(row, self.suite, teacher=True, monet=True)
        batch, prompt = processor_inputs(self.processor, chat, self.device, text=teacher_text(self.processor, chat))
        context_check(self.model, batch)
        positions = (batch["input_ids"][0] == self.model.config.latent_token_id).nonzero().flatten().tolist()
        if len(positions) != len(row["helper_images"]) * 8:
            raise ValueError("Teacher latent padding was lost during tokenization")
        # The native model still executes all blocks; only the first eight targets are retained.
        with torch.inference_mode():
            output = self.model(**batch, latent_mode=True, alignment_poss=[positions[:8]],
                                loss_type=[], output_latent_embeds=True, return_dict=True)
        return output.latent_embeds[0], {"native_positions": len(positions), "prompt": prompt,
                                        "input_tokens": batch["input_ids"].shape[-1]}


class MonetPredicted:
    def __init__(self, assets, suite, protocol, device, kind, runtime):
        import torch
        # similarity.py re-execs with the observed official runner on PYTHONPATH.
        patched = importlib.import_module("inference.vllm.monet_gpu_model_runner")
        if not hasattr(patched, "_ablation_capture") or sys.modules.get("vllm.v1.worker.gpu_model_runner") is not patched:
            raise RuntimeError("The observed native Monet runner is not active")
        from vllm import LLM, SamplingParams
        import inference.load_and_gen_vllm as native
        path = assets / "models/monet"
        self.processor = processor_for(path, assets / "models/qwen25-processor")
        native_pixels = json.loads((assets / "models/qwen25-processor/preprocessor_config.json").read_text())
        mm_kwargs = {k: native_pixels[k] for k in ("min_pixels", "max_pixels")}
        # Native latent block size remains 10. Observe first eight; do not shorten generation.
        self.engine = LLM(model=str(path), tokenizer=str(path), trust_remote_code=True,
            tensor_parallel_size=1, dtype="bfloat16", max_model_len=128000, max_num_seqs=1,
            gpu_memory_utilization=0.90, enforce_eager=True, enable_chunked_prefill=True,
            limit_mm_per_prompt={"image": 4}, mm_processor_kwargs=mm_kwargs, seed=0)
        self.trace_dir = Path(os.environ["MONET_ABLATION_TRACE_DIR"])
        confirmations = [json.loads(p.read_text()) for p in self.trace_dir.glob("ready_*.json")]
        if not confirmations or any(Path(v["runner"]).resolve() != Path(patched.__file__).resolve() for v in confirmations):
            raise RuntimeError("The current engine worker did not confirm the observed official Monet runner")
        self.sampling_config = dict(temperature=native.temperature, top_k=native.top_k,
            top_p=native.top_p, repetition_penalty=native.repetition_penalty, max_tokens=native.max_tokens,
            n=1, skip_special_tokens=False)
        self.SamplingParams = SamplingParams
        self.native, self.torch, self.suite = native, torch, suite
        self.info = {"definition": "consumed_native_vllm_recurrent_latent", "position_type": "autoregressive_latent_step",
                     "native_block_size": 10, "selection": "first_8_consumed_vectors_across_native_blocks",
                     "sampling": {"temperature": native.temperature, "top_k": native.top_k, "top_p": native.top_p,
                                  "repetition_penalty": native.repetition_penalty, "max_tokens": native.max_tokens,
                                  "engine_seed": 0, "sampling_seed": "int(sha256(case_id)[:8],16) % (2**31-1)"},
                     "worker_confirmations": confirmations,
                     "mm_processor_kwargs": mm_kwargs,
                     "image_processor": self.processor.image_processor.to_dict()}

    def extract(self, row):
        chat = messages(row, self.suite, monet=True)
        inputs = self.native.vllm_mllm_process_batch_from_messages([chat], self.processor)
        telemetry = self.processor(text=[inputs[0]["prompt"]], images=inputs[0]["multi_modal_data"]["image"],
                                   return_tensors="pt", padding=True)
        input_tokens = int(telemetry["input_ids"].shape[-1])
        if input_tokens + self.sampling_config["max_tokens"] > 128000:
            raise ValueError("Monet context overflow; input will not be truncated")
        case_seed = int(hashlib.sha256(row["id"].encode()).hexdigest()[:8], 16) % (2**31 - 1)
        sampling = self.SamplingParams(**self.sampling_config, seed=case_seed)
        outputs = self.native.vllm_generate(inputs, sampling, self.engine)
        if len(outputs) != 1 or len(outputs[0].outputs) != 1:
            raise ValueError("Expected one native Monet request and one continuation")
        output = outputs[0]
        if len(output.prompt_token_ids) != input_tokens:
            raise ValueError("Native processor and vLLM engine disagree on image tokenization")
        identity = str(output.request_id)
        path = self.trace_dir / (hashlib.sha256(identity.encode()).hexdigest() + ".pt")
        if path.exists():
            trace = self.torch.load(path, map_location="cpu", weights_only=True)
            if trace["request_id"] != identity:
                raise ValueError("Monet request/trace mismatch")
            value, count = trace["latent"], trace["native_positions"]
            path.unlink()
        else:
            # A native text-only completion is valid coverage information, not eight fabricated vectors.
            value, count = self.torch.empty((0, 3584)), 0
        return value, {"native_positions": count, "request_id": identity, "prompt": inputs[0]["prompt"],
                       "generation": output.outputs[0].text, "finish_reason": output.outputs[0].finish_reason,
                       "generated_tokens": len(output.outputs[0].token_ids), "sampling_seed": case_seed,
                       "input_tokens": input_tokens, "image_grid_thw": telemetry["image_grid_thw"].tolist()}


def load_extractor(method, kind, **kwargs):
    cls = {"ours": Ours, "lvr": LVR, "monet": MonetTeacher if kind == "teacher" else MonetPredicted}[method]
    return cls(kind=kind, **kwargs)
