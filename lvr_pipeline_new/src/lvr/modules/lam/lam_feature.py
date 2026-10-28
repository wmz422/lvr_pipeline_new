from pathlib import Path
import json

from safetensors import safe_open
from transformers import AutoConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from .blocks import patchify, unpatchify, SpatioTemporalTransformer, SpatioTransformer
from torch import Tensor
#from transformers import AutoProcessor


def _load_qwen3_vision_model(save_dir: str, load_weights: bool = True) -> Qwen3VLVisionModel:
    """Load the vision tower from either a standalone or full Qwen3-VL tree.

    The local Qwen3-VL checkpoint is a full-model sharded checkpoint, while
    recent Transformers exposes the vision tower as a separate top-level
    ``Qwen3VLVisionModel`` whose state keys do not include ``model.visual.``.
    Calling ``Qwen3VLVisionModel.from_pretrained`` directly therefore leaves
    the entire vision tower randomly initialized.  Extract only the vision
    tensors shard-by-shard so the language-model shards are never materialized.
    """
    root = Path(save_dir)
    config = AutoConfig.from_pretrained(save_dir)
    vision_config = getattr(config, "vision_config", None)
    if vision_config is None:
        return Qwen3VLVisionModel.from_pretrained(save_dir, torch_dtype=torch.float32)

    vision_model = Qwen3VLVisionModel(vision_config)
    if not load_weights:
        return vision_model
    expected = set(vision_model.state_dict())
    loaded: dict[str, torch.Tensor] = {}
    # The Qwen3-VL repository is a full-model checkpoint.  The vision tower
    # lives only in the shard(s) listed for ``model.visual.*``; opening every
    # language-model shard here makes every DDP rank scan several gigabytes
    # unnecessarily during startup.
    index_file = root / "model.safetensors.index.json"
    if index_file.exists():
        with index_file.open() as file:
            weight_map = json.load(file).get("weight_map", {})
        vision_shards = sorted(
            {
                filename
                for key, filename in weight_map.items()
                if key.startswith("model.visual.")
            }
        )
        checkpoint_files = [root / filename for filename in vision_shards]
    else:
        checkpoint_files = sorted(root.glob("*.safetensors"))
    for checkpoint_file in checkpoint_files:
        with safe_open(str(checkpoint_file), framework="pt", device="cpu") as shard:
            for key in shard.keys():
                if not key.startswith("model.visual."):
                    continue
                target_key = key.removeprefix("model.visual.")
                if target_key in expected:
                    loaded[target_key] = shard.get_tensor(key)

    if set(loaded) != expected:
        missing = sorted(expected.difference(loaded))
        raise RuntimeError(
            f"Incomplete Qwen3-VL vision checkpoint at {save_dir}; "
            f"missing {len(missing)} tensors, e.g. {missing[:3]}"
        )
    vision_model.load_state_dict(loaded, strict=True)
    vision_model.to(dtype=torch.float32)
    return vision_model


class LatentActionModel(nn.Module):
    """
    Feature-space Latent action VAE.

    当前 alignment 实验使用这个版本：先用 Qwen3 vision encoder 提取两帧 feature，
    再从 question image -> auxiliary image 的变化中得到 latent action。
    """

    def __init__(
            self,
            in_dim: int,
            model_dim: int,
            latent_dim: int,
            patch_size: int,
            enc_blocks: int,
            dec_blocks: int,
            num_heads: int,
            dropout: float = 0.0,
            feature_space: bool=False,
            image_dim: int=1024,#qwen image encoder结果维度
            num_latent:int=4, #这里是latent token数量
            save_dir: str = None,
            load_vision_weights: bool = True,
    ) -> None:
        if not save_dir:
            raise ValueError("LatentActionModel 需要 save_dir（Qwen3 vision encoder 权重目录，由 config 提供）。")
        super(LatentActionModel, self).__init__()
        self.model_dim = model_dim
        self.latent_dim = latent_dim
        self.patch_size = patch_size
        patch_token_dim = in_dim * patch_size ** 2
        self.num_latent=num_latent #属性
        self.image_dim=image_dim#相当于qwen image feature dim

        #self.processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-4B-Thinking")
        self.action_prompt = nn.Parameter(torch.empty(1, 1, self.num_latent, model_dim))
        nn.init.uniform_(self.action_prompt, a=-1, b=1)#[-1,1]内均匀分布
        self.encoder = SpatioTemporalTransformer(
            in_dim=model_dim,# 
            model_dim=model_dim,#1024
            out_dim=model_dim,#1024
            num_blocks=enc_blocks,#16
            num_heads=num_heads,#16
            dropout=dropout
        )
        self.fc = nn.Linear(model_dim, latent_dim * 2)#latent_dim=64,前32均值，后32方差
        self.encoder_proj = nn.Linear(image_dim, model_dim)#对齐qwen和model dim
        self.decoder_proj=nn.Linear(model_dim,image_dim)
        self.visual_encoder = _load_qwen3_vision_model(save_dir, load_weights=load_vision_weights)
        self.action_up = nn.Linear(latent_dim, model_dim)
        self.decoder = SpatioTransformer(
            in_dim=model_dim,
            model_dim=model_dim,
            out_dim=model_dim,
            num_blocks=dec_blocks,
            num_heads=num_heads,
            dropout=dropout
        )

        self.mu_record = None

        for param in self.visual_encoder.parameters():#显示冻结image_encoder
            param.requires_grad = False

    def encode(self, videos: Tensor) -> Dict:
        """从两帧 visual feature 中编码 latent action。"""
        # Preprocess videos
        B, T = videos.shape[:2]
        action_pad = self.action_prompt.expand(B, T, -1, -1)
        

        padded_patches = torch.cat([action_pad, videos], dim=2)#[B,2,256+num_latent,]

        # Encode
        z = self.encoder(padded_patches)  # (B, T, 1+N, E) [B,2,257,1024]
        # Get latent action for all future frames
        z = z[:,1:, :self.num_latent]  # (B, T-1, num_latent, E)取出第二帧的a

        # VAE
        z = z.reshape(B * (T - 1)*self.num_latent, self.model_dim)
        moments = self.fc(z)#[B*1,32x2]
        z_mu, z_var = torch.chunk(moments, 2, dim=1)#一分为二，前32均值，后32方差
        # Reparameterization
        if not self.training:
            z_rep = z_mu#测试阶段不用
        else:
            z_rep = z_mu + torch.randn_like(z_var) * torch.exp(0.5 * z_var)#重参数化
        z_rep = z_rep.reshape(B, T - 1, self.num_latent, self.latent_dim)#[0,1,1,32]

        if not self.training:
            if self.mu_record is None:
                self.mu_record = z_mu
            else:
                self.mu_record = torch.cat([self.mu_record, z_mu], dim=0)

        return {
            "z_rep": z_rep,
            "z_mu": z_mu,
            "z_var": z_var
        }

    def forward(self, batch: Dict) -> Dict:
        """返回 z_rep 以及用于 LAM 自身训练的 feature 重建输出。"""
        # Encode + VAE
        #H, W = batch["videos"].shape[2:4]
        features=self.get_images_features(batch)
        outputs = self.encode(self.encoder_proj(features))#feature[B,T,S,D]
        video_patches = features[:, :-1]#只取第一帧的意思
        action_patches = self.action_up(outputs["z_rep"])
        video_action_patches = torch.cat([action_patches,video_patches],dim=2)

        # Decode
        video_recon = self.decoder(video_action_patches)[:,:,self.num_latent:]

        #video_recon = F.sigmoid(video_recon)#放缩到0-1之间，符合图像渲染,这个还要不要待定
        outputs.update(#这个还要不要待定
            {
                "gt_feature0":features[:,0],#初始帧的feature
                "gt_feature":features[:,1:],
                "feature": self.decoder_proj(video_recon)
            }
        )
        return outputs
    
    def get_images_features(self, batch: Tensor)->Tensor:
        """调用冻结的 Qwen3 vision encoder，把两帧图像转成 visual feature token。"""
        pixel_values = batch["pixel_values"]
        if pixel_values.ndim == 4:
            B, T, S, D = pixel_values.shape
        elif pixel_values.ndim == 3:
            # Backward compatibility with collators that flattened the image
            # dimension but retained the batch dimension.
            B, flat_S, D = pixel_values.shape
            T = batch["image_grid_thw"].reshape(B, -1, 3).shape[1]
            if flat_S % T:
                raise ValueError(
                    f"pixel_values token count {flat_S} is not divisible by {T} images"
                )
            S = flat_S // T
            pixel_values = pixel_values.reshape(B, T, S, D)
        else:
            raise ValueError(
                "Expected pixel_values with shape [B,T,S,D] or [B,T*S,D], "
                f"got {tuple(pixel_values.shape)}"
            )
        pixel_values = pixel_values.reshape(-1, D)
        grid_thw = batch["image_grid_thw"].reshape(-1, 3)
        self.visual_encoder.eval()
        
        with torch.no_grad():
            image_outs = self.visual_encoder(
                hidden_states=pixel_values, # 确保精度对齐(fp16/bf16)
                grid_thw=grid_thw
            )
        images_embeddings=image_outs['last_hidden_state'].reshape(B,T,-1,self.image_dim)#torch.Size([B, 2, S, 1024])
        return images_embeddings#[10, 2, 256, 1024])
