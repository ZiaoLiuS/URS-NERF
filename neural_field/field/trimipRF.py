import math
from typing import Callable

import gin
import torch
from torch import Tensor, nn
import tinycudann as tcnn

from neural_field.encoding.tri_mip import TriMipEncoding
from neural_field.nn_utils.activations import trunc_exp
import torch.nn.functional as F


@gin.configurable()
class TriMipRF(nn.Module):
    def __init__(
            self,
            beta_min: float = 0.01,
            n_levels: int = 10,
            plane_size: int = 512,
            feature_dim: int = 32,
            geo_feat_dim: int = 15,
            net_depth_base: int = 3,
            net_depth_color: int = 5,
            net_width: int = 64,
            density_activation: Callable = lambda x: trunc_exp(x - 1),
    ) -> None:
        super().__init__()
        self.plane_size = plane_size
        self.log2_plane_size = math.log2(plane_size)
        self.geo_feat_dim = geo_feat_dim
        self.density_activation = density_activation
        self.n_levels = n_levels
        self.beta_min = beta_min
        self.feature_dim = feature_dim
        self.net_depth_base = net_depth_base
        self.net_depth_color = net_depth_color
        self.net_width = net_width

        self.encoding = TriMipEncoding(n_levels, plane_size, feature_dim)
        self.direction_encoding = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                "otype": "SphericalHarmonics",
                "degree": 4,
            },
        )
        self.mlp_base = tcnn.Network(
            n_input_dims=self.encoding.dim_out,
            n_output_dims=geo_feat_dim + 1,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": net_width,
                "n_hidden_layers": net_depth_base,
            },
        )
        self.mlp_head = tcnn.Network(
            n_input_dims=self.direction_encoding.n_output_dims + geo_feat_dim,
            n_output_dims=3,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "Sigmoid",
                "n_neurons": net_width,
                "n_hidden_layers": net_depth_color,
            },
        )

        # self.mlp_head2 = tcnn.Network(
        #     n_input_dims=self.direction_encoding.n_output_dims + geo_feat_dim,
        #     n_output_dims=1,
        #     network_config={
        #         "otype": "FullyFusedMLP",
        #         "activation": "ReLU",
        #         "output_activation": "Sigmoid",
        #         "n_neurons": net_width,
        #         "n_hidden_layers": net_depth_color,
        #     },
        # )

    def query_density(
            self, x: Tensor, level_vol: Tensor, step, return_feat: bool = False
    ):
        level = (
            level_vol if level_vol is None else level_vol + self.log2_plane_size
        )
        selector = ((x > 0.0) & (x < 1.0)).all(dim=-1)

        enc = self.encoding(
            x.view(-1, 3),
            level=level.view(-1, 1),
            step=step)

        x = (
            self.mlp_base(enc)
            .view(list(x.shape[:-1]) + [1 + self.geo_feat_dim])
            .to(x)
        )
        density_before_activation, base_mlp_out = torch.split(
            x, [1, self.geo_feat_dim], dim=-1
        )
        density = (
                self.density_activation(density_before_activation)
                * selector[..., None]
        )
        return {
            "density": density,
            "feature": base_mlp_out if return_feat else None,
        }

    def query_rgb(self, dir, embedding):
        # dir in [-1,1]
        dir = (dir + 1.0) / 2.0  # SH encoding must be in the range [0, 1]
        d = self.direction_encoding(dir.view(-1, dir.shape[-1]))
        h = torch.cat([d, embedding.view(-1, self.geo_feat_dim)], dim=-1)
        rgb_tensor = (
            self.mlp_head(h)
            .view(list(embedding.shape[:-1]) + [3])
            .to(embedding)
        )
        # s = rgb_tensor[:, -1].unsqueeze(1)  # 提取张量的最后一列作为 s，并将其形状调整为 [1328854, 1]
        # rgb = rgb_tensor[:, :3]  # 提取张量的前三列作为 RGB 值
        # ref_rgb = rgb_tensor[:, 4:7]  # 提取张量的接下来三列作为参考 RGB 值
        #
        # # 计算最终的 RGB 值
        # rgb_final = rgb + s * ref_rgb

        return {"rgb": rgb_tensor}

    # def query_uncertainty(self, dir, embedding):
    #     # dir in [-1,1]
    #     dir = (dir + 1.0) / 2.0  # SH encoding must be in the range [0, 1]
    #     d = self.direction_encoding(dir.view(-1, dir.shape[-1]))
    #     h = torch.cat([d, embedding.view(-1, self.geo_feat_dim)], dim=-1)
    #     uncertainry = (
    #         self.mlp_head2(h)
    #         .view(list(embedding.shape[:-1]) + [1])
    #         .to(embedding)
    #     )
    #
    #     uncertainry = F.softplus(uncertainry) + self.beta_min
    #     return {"uncertainty": uncertainry}

    def init_parameters(self):
        self.encoding.init_parameters()
        self.direction_encoding = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                "otype": "SphericalHarmonics",
                "degree": 4,
            },
        )
        self.mlp_base = tcnn.Network(
            n_input_dims=self.encoding.dim_out,
            n_output_dims=self.geo_feat_dim + 1,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": self.net_width,
                "n_hidden_layers": self.net_depth_base,
            },
        )
        self.mlp_head = tcnn.Network(
            n_input_dims=self.direction_encoding.n_output_dims + self.geo_feat_dim,
            n_output_dims=3,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "Sigmoid",
                "n_neurons": self.net_width,
                "n_hidden_layers": self.net_depth_color,
            },
        )

        self.mlp_head2 = tcnn.Network(
            n_input_dims=self.direction_encoding.n_output_dims + self.geo_feat_dim,
            n_output_dims=1,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "Sigmoid",
                "n_neurons": self.net_width,
                "n_hidden_layers": self.net_depth_color,
            },
        )
