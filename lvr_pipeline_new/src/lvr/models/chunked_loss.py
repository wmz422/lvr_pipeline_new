"""A checkpoint-compatible LM head with one module call per chunked loss."""

import torch
from torch import nn
from torch.nn import functional as F

from lvr.training.diagnostics import trace_event


class ChunkedLMHead(nn.Linear):
    """Keep ZeRO-3's gather/release hooks outside the entire variable-length CE.

    Calling a sharded Linear module once per token chunk issues a different
    number of collectives on ranks with different target lengths. This module
    owns the same weight/bias as the original head, but evaluates all local
    chunks inside ONE forward. Its module-level backward hooks also bracket the
    whole loss graph. No padding, rank-wise token reduction or extra parameters
    are needed. Without targets it is an ordinary Linear (e.g. for generation).
    """

    @classmethod
    def from_linear(cls, head):
        if isinstance(head, cls):
            return head
        if type(head) is not nn.Linear:
            raise TypeError(f"Expected a native nn.Linear LM head, got {type(head).__name__}")
        # Do not call nn.Linear.__init__: the full vocabulary weight already
        # exists and can be tied to input embeddings. Reuse the Parameter objects
        # without allocating/reinitializing weights or consuming any RNG state.
        result = cls.__new__(cls)
        nn.Module.__init__(result)
        result.in_features, result.out_features = head.in_features, head.out_features
        result.register_parameter("weight", head.weight)
        result.register_parameter("bias", head.bias)
        result.train(head.training)
        return result

    def forward(self, input, targets=None):
        if targets is None:
            return F.linear(input, self.weight, self.bias)
        if input.ndim != 2 or targets.ndim != 1 or input.shape[0] != targets.numel():
            raise ValueError("Chunked CE expects [tokens, hidden] inputs and [tokens] targets")
        if not targets.numel():
            raise ValueError("The text branch contains no supervised target tokens")
        trace_event("answer_loss.chunks", target_tokens=targets.numel(),
                    chunk_size=256, chunk_count=(targets.numel() + 255) // 256,
                    head_module_calls=1)
        loss = torch.zeros((), device=input.device, dtype=torch.float32)
        for index, (states, target) in enumerate(zip(input.split(256), targets.split(256)), start=1):
            trace_event("answer_chunk.begin", chunk_index=index, target_tokens=target.numel())
            # Functional linear is deliberately inside this module's forward:
            # using self(states) or a child Linear would reintroduce ZeRO hooks.
            loss = loss + F.cross_entropy(F.linear(states, self.weight, self.bias).float(),
                                          target, reduction="sum")
            trace_event("answer_chunk.end", chunk_index=index)
        return loss / targets.numel()
