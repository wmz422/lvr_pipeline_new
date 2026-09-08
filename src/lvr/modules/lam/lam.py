from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from .blocks import patchify, unpatchify, SpatioTemporalTransformer, SpatioTransformer
from torch import Tensor



class LatentActionModel(nn.Module):
    """
    Pixel-space Latent action VAE.

    这是原始图像 patch 版本，当前 alignment 默认使用 lam_feature.py 中的
    feature-space 版本；保留这个文件方便和原 LAM 工程对照。
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
            num_latent:int=4, #这里是latent token数量
    ) -> None:
        super(LatentActionModel, self).__init__()
        self.model_dim = model_dim
        self.latent_dim = latent_dim
        self.patch_size = patch_size
        patch_token_dim = in_dim * patch_size ** 2
        self.num_latent=num_latent #属性

        self.action_prompt = nn.Parameter(torch.empty(1, 1, self.num_latent, patch_token_dim))
        nn.init.uniform_(self.action_prompt, a=-1, b=1)
        self.encoder = SpatioTemporalTransformer(
            in_dim=patch_token_dim,# 3*256
            model_dim=model_dim,#1024
            out_dim=model_dim,#1024
            num_blocks=enc_blocks,#16
            num_heads=num_heads,#16
            dropout=dropout
        )
        self.fc = nn.Linear(model_dim, latent_dim * 2)#latent_dim=64,前32均值，后32方差
        self.patch_up = nn.Linear(patch_token_dim, model_dim)
        self.action_up = nn.Linear(latent_dim, model_dim)
        self.decoder = SpatioTransformer(
            in_dim=model_dim,
            model_dim=model_dim,
            out_dim=patch_token_dim,
            num_blocks=dec_blocks,
            num_heads=num_heads,
            dropout=dropout
        )

        self.mu_record = None

    def encode(self, videos: Tensor) -> Dict:
        """把两帧图像 patch 编码成 latent action。"""
        # Preprocess videos
        B, T = videos.shape[:2]
        patches = patchify(videos, self.patch_size)#[B,2,256,256x3]
        action_pad = self.action_prompt.expand(B, T, -1, -1)
        

        padded_patches = torch.cat([action_pad, patches], dim=2)#[B,2,256+num_latent,256x3]

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
            "patches": patches,
            "z_rep": z_rep,
            "z_mu": z_mu,
            "z_var": z_var
        }

    def forward(self, batch: Dict) -> Dict:
        """原始 LAM 训练 forward：编码 latent，并重建后一帧图像。"""
        # Encode + VAE
        H, W = batch["videos"].shape[2:4]
        outputs = self.encode(batch["videos"])
        video_patches = self.patch_up(outputs["patches"][:, :-1])#只取第一帧的意思
        action_patches = self.action_up(outputs["z_rep"])
        video_action_patches = torch.cat([action_patches,video_patches],dim=2)

        del outputs["patches"]#删掉outputs中的patches关键字

        # Decode
        video_recon = self.decoder(video_action_patches)[:,:,self.num_latent:]
        video_recon = F.sigmoid(video_recon)#放缩到0-1之间，符合图像渲染
        outputs.update(
            {
                "recon": unpatchify(video_recon, self.patch_size, H, W)
            }
        )
        return outputs
