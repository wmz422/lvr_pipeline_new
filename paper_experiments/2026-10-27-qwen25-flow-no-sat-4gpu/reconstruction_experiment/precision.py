"""Keep cached Flow latents FP32 through Lightning's DeepSpeed input conversion."""
from lightning.pytorch.plugins.precision import DeepSpeedPrecision
from .latent_cache import REPRESENTATION


class MatchedDeepSpeedPrecision(DeepSpeedPrecision):
    """Keep the default recipe's gradient scale when microbatch/GAS changes.

    This installed Lightning divides closure loss by GAS and DeepSpeed divides
    it again. Preserve the original default recipe (including its effective
    scale), while compensating only for tuning changes to GAS. Forward losses,
    their relative weights, clipping and optimizer settings remain unchanged.
    """
    def __init__(self, precision: str = "bf16-mixed", reference_accumulation: int = 8):
        if precision != "bf16-mixed":
            raise ValueError("This experiment retains the Plain bf16-mixed precision policy")
        if reference_accumulation < 1:
            raise ValueError("Reference accumulation must be positive")
        self.reference_accumulation = reference_accumulation
        super().__init__(precision=precision)

    def backward(self, tensor, model, optimizer, *args, **kwargs):
        trainer = model.trainer
        accumulation = trainer.accumulate_grad_batches
        engine = trainer.model
        if (not model.automatic_optimization
                or engine.gradient_accumulation_steps() != accumulation
                or getattr(engine, "scale_wrt_gas", None) is False
                or kwargs.get("scale_wrt_gas", True) is False):
            raise RuntimeError("Unexpected Lightning/DeepSpeed loss normalization policy")
        if accumulation != self.reference_accumulation:
            tensor = tensor * (accumulation / self.reference_accumulation)
        return super().backward(tensor, model, optimizer, *args, **kwargs)


class FlowDeepSpeedPrecision(MatchedDeepSpeedPrecision):
    def __init__(self, precision: str = "bf16-mixed", reference_accumulation: int = 16):
        super().__init__(precision=precision, reference_accumulation=reference_accumulation)

    def convert_input(self, data):
        if isinstance(data, dict) and "flow" in data:
            flow = data["flow"]
            if flow.get("representation") != REPRESENTATION:
                raise ValueError("Unexpected Flow representation at the precision boundary")
            converted = super().convert_input({key: value for key, value in data.items() if key != "flow"})
            converted["flow"] = flow
            return converted
        return super().convert_input(data)
