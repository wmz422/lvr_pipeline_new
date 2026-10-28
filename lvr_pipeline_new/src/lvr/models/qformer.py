"""Question-aware, fixed-query compression of variable-length Qwen visual features."""

import math

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from transformers import Blip2QFormerConfig, Blip2QFormerModel
from transformers.masking_utils import create_bidirectional_mask


def sinusoidal_positions(positions, width):
    frequencies = torch.exp(torch.arange(0, width, 2, device=positions.device).float()
                            * (-math.log(10000.0) / width))
    angles = positions.float().unsqueeze(-1) * frequencies
    return torch.stack((angles.sin(), angles.cos()), dim=-1).flatten(-2)


class QueryFormerEncoder(nn.Module):
    """Only queries cross-attend to vision; queries and question tokens self-attend.

    The caller supplies frozen vision features and detached Qwen word embeddings.
    This module has no second vision tower, vocabulary table, or language head.
    """

    def __init__(self, input_dim, hidden_size=768, num_layers=6, num_heads=12,
                 cross_attention_frequency=2, uses_question=True, use_geometry=False,
                 gradient_checkpointing=True):
        super().__init__()
        if hidden_size < 2 or hidden_size % 2 or num_heads < 1 or hidden_size % num_heads:
            raise ValueError("Q-Former width must be even and divisible by its positive head count")
        if num_layers < 1 or cross_attention_frequency < 1:
            raise ValueError("Q-Former layer count and cross-attention frequency must be positive")
        self.vision_projection = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, hidden_size))
        self.text_projection = (nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, hidden_size))
                                if uses_question else None)
        # bbox(4), global patch center(2), original/crop flag(1), source/crop IDs(16 each).
        self.geometry_projection = nn.Linear(39, hidden_size) if use_geometry else None
        config = Blip2QFormerConfig(
            hidden_size=hidden_size, intermediate_size=4 * hidden_size,
            num_hidden_layers=num_layers, num_attention_heads=num_heads,
            encoder_hidden_size=hidden_size, cross_attention_frequency=cross_attention_frequency,
            use_qformer_text_input=uses_question, hidden_dropout_prob=0.0, attention_probs_dropout_prob=0.0,
        )
        config._attn_implementation = "sdpa"
        self.transformer = Blip2QFormerModel(config)
        if uses_question:
            # Final text FFN outputs are not read by any subsequent query layer.
            # Freeze these unused weights so DDP(find_unused_parameters=False) remains valid.
            final = self.transformer.encoder.layer[-1]
            final.intermediate.requires_grad_(False)
            final.output.requires_grad_(False)
        if gradient_checkpointing:
            self.transformer.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False})

    def _visual_memory(self, features, batch, merge_size):
        counts = batch["image_counts"].tolist()
        if min(counts) < 1 or sum(counts) != len(features):
            raise ValueError("Visual image counts do not match the batch")
        if len(batch["image_metadata"]) != len(features) or len(batch["image_grid_thw"]) != len(features):
            raise ValueError("Every visual feature sequence requires a grid and source metadata")
        projected = []
        for feature, grid, metadata in zip(features, batch["image_grid_thw"], batch["image_metadata"]):
            time, height, width = grid.tolist()
            if time != 1 or height % merge_size or width % merge_size:
                raise ValueError("Q-Former crops require still images with complete spatial merge groups")
            height, width = height // merge_size, width // merge_size
            if feature.shape[0] != height * width:
                raise ValueError("Visual feature length does not match its spatial grid")
            value = self.vision_projection(feature.float())
            if self.geometry_projection is not None:
                yy, xx = torch.meshgrid(
                    (torch.arange(height, device=value.device).float() + 0.5) / height,
                    (torch.arange(width, device=value.device).float() + 0.5) / width, indexing="ij")
                box = metadata[:4].float()
                centers = torch.stack((box[0] + xx.flatten() * (box[2] - box[0]),
                                       box[1] + yy.flatten() * (box[3] - box[1])), dim=-1)
                ids = sinusoidal_positions(metadata[4:6], 16).flatten()
                geometry = torch.cat((box.expand(len(value), -1), centers,
                                      metadata[6:].expand(len(value), -1), ids.expand(len(value), -1)), dim=-1)
                value = value + self.geometry_projection(geometry)
            projected.append(value)
        sequences, offset = [], 0
        for count in counts:
            sequences.append(torch.cat(projected[offset:offset + count]))
            offset += count
        memory = pad_sequence(sequences, batch_first=True)
        lengths = torch.tensor([len(x) for x in sequences], device=memory.device)
        mask = torch.arange(memory.shape[1], device=memory.device)[None, :] < lengths[:, None]
        return memory, mask

    def forward(self, features, batch, queries, merge_size, question_embeddings=None):
        memory, memory_mask = self._visual_memory(features, batch, merge_size)
        count = len(queries)
        inputs = queries.unsqueeze(0).expand(len(memory), -1, -1)
        mask = torch.ones(inputs.shape[:2], dtype=torch.bool, device=inputs.device)
        if self.text_projection is not None:
            if question_embeddings is None:
                raise ValueError("Question-conditioned Q-Former requires question embeddings")
            text = self.text_projection(question_embeddings)
            positions = torch.arange(text.shape[1], device=text.device)
            text = text + sinusoidal_positions(positions, text.shape[-1]).to(text.dtype)
            inputs = torch.cat((inputs, text), dim=1)
            mask = torch.cat((mask, batch["question_attention_mask"].bool()), dim=1)
        inputs = self.transformer.layernorm(inputs.float())
        # transformers 5.15 Blip2QFormerModel.forward builds the cross mask with
        # query+text length. Only query positions cross-attend, so use K here.
        attention_mask = create_bidirectional_mask(
            config=self.transformer.config, inputs_embeds=inputs, attention_mask=mask)
        cross_mask = create_bidirectional_mask(
            config=self.transformer.config, inputs_embeds=inputs[:, :count],
            attention_mask=memory_mask, encoder_hidden_states=memory)
        return self.transformer.encoder(
            inputs, query_length=count, attention_mask=attention_mask,
            encoder_hidden_states=memory, encoder_attention_mask=cross_mask,
        ).last_hidden_state[:, :count]
