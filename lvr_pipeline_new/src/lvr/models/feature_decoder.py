"""Decode auxiliary visual features from originals and a small latent bottleneck.

Target features enter only the loss. Output queries contain a learned mask token,
2-D positions and an auxiliary-image index, never target visual content or boxes.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence


def _positions(values, width):
    frequencies = torch.exp(torch.arange(0, width, 2, device=values.device).float()
                            * (-math.log(10000.0) / width))
    angles = values.float().unsqueeze(-1) * frequencies
    return torch.stack((angles.sin(), angles.cos()), dim=-1).flatten(-2)


def _grid_positions(grid, width, merge_size, device):
    time, height, columns = map(int, grid)
    if time != 1 or min(height, columns) < 1 or height % merge_size or columns % merge_size:
        raise ValueError("Reconstruction requires still-image grids with complete spatial merge groups")
    yy, xx = torch.meshgrid(torch.arange(height // merge_size, device=device),
                            torch.arange(columns // merge_size, device=device), indexing="ij")
    return torch.cat((_positions(yy.flatten(), width // 2),
                      _positions(xx.flatten(), width // 2)), dim=-1)


class AuxiliaryFeatureDecoder(nn.Module):
    def __init__(self, feature_dim, latent_dim, hidden_size=512, num_layers=2, num_heads=8):
        super().__init__()
        if (hidden_size < 4 or hidden_size % 4 or num_heads < 1 or hidden_size % num_heads
                or num_layers < 1 or min(feature_dim, latent_dim) < 1):
            raise ValueError("Decoder dimensions must be positive; width must be divisible by 4 and head count")
        self.hidden_size = hidden_size
        self.feature_dim = feature_dim
        self.original_projection = nn.Linear(feature_dim, hidden_size)
        self.latent_projection = nn.Linear(latent_dim, hidden_size)
        self.mask_token = nn.Parameter(torch.empty(1, hidden_size))#可学习的auxiliary_image_encoding，empty表示先不进行初始化
        self.memory_types = nn.Parameter(torch.empty(2, hidden_size))#区分原图和latent
        nn.init.normal_(self.mask_token, std=0.02)
        nn.init.normal_(self.memory_types, std=0.02)
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(hidden_size, num_heads, dim_feedforward=4 * hidden_size,
                                       dropout=0.0, activation="gelu", batch_first=True, norm_first=True),
            num_layers, norm=nn.LayerNorm(hidden_size),
        )
        self.output_projection = nn.Linear(hidden_size, feature_dim)

    def _project_originals(self, features, dtype):
        # ZeRO-3 gathers parameters at module boundaries. Every rank must call
        # this Linear once, even when its batch contains a different image count.
        lengths = [len(feature) for feature in features]
        packed = torch.cat([feature.detach().to(dtype) for feature in features], dim=0)  # 编码图片feature的编码器不参与更新
        return self.original_projection(packed).split(lengths, dim=0)

    def forward(self, latents, original_features, original_grids, original_counts,
                target_grids, target_sample_ids, target_image_ids, merge_size):
        """Return padded [auxiliary images, patches, channels] predictions and valid mask.

        Each auxiliary image independently attends to its sample's original images
        and all latent slots. Variable resolutions are padded only for attention.
        """
        if latents.ndim != 3 or latents.shape[1] < 1 or merge_size < 1:
            raise ValueError("Reconstruction requires nonempty [batch, slots, channels] latents and positive merge size")
        counts = original_counts.tolist()
        samples, image_ids = target_sample_ids.tolist(), target_image_ids.tolist()
        grids, source_grids = target_grids.tolist(), original_grids.tolist()
        if (len(counts) != len(latents) or any(n < 1 for n in counts)
                or sum(counts) != len(original_features) or len(source_grids) != len(original_features)):
            raise ValueError("Original image counts/grids do not match reconstruction batch")
        if (not samples or len(samples) != len(grids) or len(samples) != len(image_ids)
                or set(samples) != set(range(len(latents))) or min(image_ids) < 0):
            raise ValueError("Each sample requires at least one auxiliary target with a valid image index")
        device, dtype = latents.device, self.mask_token.dtype
        latent_memory = self.latent_projection(latents.to(dtype))
        latent_memory = latent_memory + self.memory_types[1].to(latent_memory.dtype)
        if any(feature.ndim != 2 or feature.shape[1] != self.feature_dim for feature in original_features):
            raise ValueError("Original features must have shape [patches, feature_dim]")
        original_memory = self._project_originals(original_features, dtype)
        memories, offset = [], 0
        for sample_index, count in enumerate(counts):
            source = []
            for image_index in range(count):
                feature = original_features[offset]
                position = _grid_positions(source_grids[offset], self.hidden_size, merge_size, device)
                if feature.shape != (len(position), self.feature_dim):
                    raise ValueError("Original feature shape does not match its spatial grid")
                value = original_memory[offset]
                position = position + _positions(torch.tensor(image_index, device=device), self.hidden_size)
                source.append(value + position.to(value.dtype) + self.memory_types[0].to(value.dtype))
                offset += 1
            memories.append(torch.cat((*source, latent_memory[sample_index]), dim=0))
        memory = pad_sequence([memories[sample] for sample in samples], batch_first=True)
        lengths = torch.tensor([len(memories[sample]) for sample in samples], device=device)
        memory_padding = torch.arange(memory.shape[1], device=device)[None, :] >= lengths[:, None]
        queries = []
        for grid, image_index in zip(grids, image_ids):
            position = _grid_positions(grid, self.hidden_size, merge_size, device)
            position = position + _positions(torch.tensor(image_index, device=device), self.hidden_size)
            queries.append((self.mask_token + position.to(dtype)).to(memory.dtype))
        query = pad_sequence(queries, batch_first=True)
        lengths = torch.tensor([len(value) for value in queries], device=device)
        valid = torch.arange(query.shape[1], device=device)[None, :] < lengths[:, None]
        hidden = self.decoder(query, memory, tgt_key_padding_mask=~valid,#交叉注意力
                              memory_key_padding_mask=memory_padding)
        return self.output_projection(hidden), valid


def feature_reconstruction_loss(prediction, targets, valid, sample_ids, batch_size, *, normalize=False):
    """Average over channels/patches, then images per sample, then samples.

    Float32 loss and detached targets work with either autocast or BF16 weights.
    Optional non-affine LayerNorm removes feature scale; raw MSE is always logged.
    """
    if prediction.ndim != 3 or valid.shape != prediction.shape[:2] or valid.dtype != torch.bool:
        raise ValueError("Prediction and valid-patch mask shapes must agree")
    if len(targets) != len(prediction) or sample_ids.shape != (len(targets),):
        raise ValueError("Every predicted auxiliary image needs a target and sample ID")
    if (batch_size < 1 or sample_ids.dtype != torch.long or sample_ids.numel() == 0
            or sample_ids.min() < 0 or sample_ids.max() >= batch_size):
        raise ValueError("Invalid reconstruction sample IDs")
    for target, mask in zip(targets, valid):
        if target.shape != (int(mask.sum()), prediction.shape[-1]):
            raise ValueError("Auxiliary feature shape does not match predicted valid patches")
    target = pad_sequence([value.detach().float() for value in targets], batch_first=True)
    prediction = prediction.float()
    raw_error = (prediction - target).square().mean(-1)
    if normalize:
        prediction = F.layer_norm(prediction, (prediction.shape[-1],))
        target = F.layer_norm(target, (target.shape[-1],))
    error = (prediction - target).square().mean(-1) if normalize else raw_error

    def reduce(values):
        per_image = values.masked_fill(~valid, 0).sum(-1) / valid.sum(-1).clamp_min(1)
        counts = torch.bincount(sample_ids, minlength=batch_size)
        if not counts.gt(0).all():
            raise ValueError("Every sample requires an auxiliary reconstruction target")
        per_sample = per_image.new_zeros(batch_size).index_add(0, sample_ids, per_image)
        return (per_sample / counts).mean()

    return {"reconstruction_mse": reduce(error), "reconstruction_raw_mse": reduce(raw_error)}
