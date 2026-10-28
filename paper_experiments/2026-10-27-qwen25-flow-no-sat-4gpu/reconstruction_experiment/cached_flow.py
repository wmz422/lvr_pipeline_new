"""The original stochastic Flow loss, consuming precomputed clean/ref latents."""
import torch
from flow_reconstruction.flow import FrozenFlowRuntime, verify_model_lock


class CachedFrozenFlowRuntime(FrozenFlowRuntime):
    def __init__(self, model_path, *, checkpointing=True, device=None, generator_batch_size=2):
        from diffusers import Flux2Transformer2DModel, FlowMatchEulerDiscreteScheduler
        verify_model_lock(model_path)
        self._device = torch.device("cpu" if device is None else device)
        self.transformer = Flux2Transformer2DModel.from_pretrained(
            model_path, subfolder="transformer", local_files_only=True, torch_dtype=torch.bfloat16,
            device_map={"": self._device}, low_cpu_mem_usage=True).requires_grad_(False).eval()
        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            model_path, subfolder="scheduler", local_files_only=True)
        if checkpointing:
            self.transformer.enable_gradient_checkpointing()
        if generator_batch_size < 1:
            raise ValueError("Invalid Flow generator batch size")
        self.checkpointing, self.generator_batch_size = checkpointing, generator_batch_size
        self.condition_dim = self.transformer.config.joint_attention_dim
        self.last_stats = {"generator_batch_sizes": [], "vae_batch_sizes": [], "vae_cache_hits": 0}

    def move(self, device):
        device = torch.device(device)
        if self._device != device:
            self.transformer.to(device)
            self._device = device

    def encode_many(self, images, keys=None):
        """Satisfy the inherited loss's encoding boundary without a VAE forward.

        The loader has checked payload hashes, finite values and image shapes.
        Keep the original loss implementation, including RNG order and reduction.
        """
        if keys is not None and len(images) != len(keys):
            raise ValueError("Latent/key count mismatch")
        channels = self.transformer.config.in_channels
        for value in images:
            if (value.ndim != 4 or value.shape[:2] != (1, channels) or min(value.shape[2:]) < 1
                    or value.dtype != torch.float32 or value.requires_grad or value.device != self._device):
                raise ValueError("Expected detached normalized FP32 cached latent on the Flow device")
        return images

    def generate(self, *args, **kwargs):
        raise NotImplementedError("Use the original RGB runtime for image decoding; this runtime trains from latents")
