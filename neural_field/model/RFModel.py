import math
from typing import Union, List, Dict

import gin
import torch
from torch import nn
import torch.nn.functional as F
from torchmetrics.functional import peak_signal_noise_ratio
from external.pohsun_ssim import pytorch_ssim
from utils.ray import RayBundle
from utils.render_buffer import RenderBuffer
import dataset.utils.poses as camera
import lpips
# @gin.configurable()
class RFModel(nn.Module):
    def __init__(
        self,
        aabb: Union[torch.Tensor, List[float]],
        samples_per_ray: int = 2048,
    ) -> None:
        super().__init__()
        if not isinstance(aabb, torch.Tensor):
            aabb = torch.tensor(aabb, dtype=torch.float32)
        self.register_buffer("aabb", aabb)
        self.samples_per_ray = samples_per_ray
        self.render_step_size = (
            (self.aabb[3:] - self.aabb[:3]).max()
            * math.sqrt(3)
            / samples_per_ray
        ).item()
        aabb_min, aabb_max = torch.split(self.aabb, 3, dim=-1)
        self.aabb_size = aabb_max - aabb_min
        assert (
            self.aabb_size[0] == self.aabb_size[1] == self.aabb_size[2]
        ), "Current implementation only supports cube aabb"
        self.field = None
        self.ray_sampler = None
        self.lpips_loss = lpips.LPIPS(net="alex").cuda()

    def contraction(self, x):
        aabb_min, aabb_max = self.aabb[:3].unsqueeze(0), self.aabb[
            3:
        ].unsqueeze(0)
        x = (x - aabb_min) / (aabb_max - aabb_min)
        return x

    def before_iter(self, step):
        pass

    def after_iter(self, step):
        pass

    def forward(
        self,
        rays: RayBundle,
        background_color=None,
    ):
        raise NotImplementedError

    @gin.configurable()
    def get_optimizer(
        self, lr=1e-3, weight_decay=1e-5, feature_lr_scale=10.0, **kwargs
    ):
        raise NotImplementedError

    @gin.configurable()
    def compute_loss(
        self,
        rays: RayBundle,
        rb: RenderBuffer,
        target: RenderBuffer,
        # Configurable
        metric='smooth_l1',
        **kwargs
    ) -> Dict:
        if 'smooth_l1' == metric:
            loss_fn = F.smooth_l1_loss
        elif 'mse' == metric:
            loss_fn = F.mse_loss
        elif 'mae' == metric:
            loss_fn = F.l1_loss
        else:
            raise NotImplementedError

        alive_ray_mask = (rb.alpha.squeeze(-1) > 0).detach()

        loss_render = loss_fn(
            rb.rgb[alive_ray_mask], target.rgb[alive_ray_mask], reduction='none'
        )

        loss_render = (
            loss_render * target.loss_multi[alive_ray_mask]
        ).sum() / target.loss_multi[alive_ray_mask].sum()


        loss = loss_render

        return {'loss': loss}



    def get_vel_prior_loss(self):
        pose, _ = self.get_all_training_poses()
        pose0 = pose[:-1]
        pose1 = pose[1:]
        t_c0_c1 = camera.pose.compose_pair(camera.pose.invert(pose0), pose1)

        # sign_tensor1 = torch.sign(camera.lie.SE3_to_se3(t_c0_c1))
        # sign_tensor2 = torch.sign(self.se3_vel_refine.weight.data[:-1])
        # delta_t = 481
        # shutter_t = 480
        # 对应位置相乘得到结果，1表示正负相同，-1表示正负不同



        # result = sign_tensor1 * sign_tensor2
        # torch.set_printoptions(sci_mode=False)
        # for i in range(len(t_c0_c1)):
        #     print("gt: ", camera.lie.SE3_to_se3(t_c0_c1)[i])
        #     print("est: ", self.se3_vel_refine.weight.data[:-1][i])


        # 计算余弦相似度
        cosine_similarity = F.cosine_similarity(camera.lie.SE3_to_se3(t_c0_c1), self.se3_vel_refine.weight.data[:-1], dim=-1)


        # 使用余弦相似度计算角度损失
        angle_loss = 1.0 - cosine_similarity  # 1.0减去余弦相似度表示角度差异

        # print(result)
        # print(torch.isclose(self.se3_vel_refine.weight.data[:-1],camera.lie.SE3_to_se3(t_c0_c1),1))
        # return torch.mean(angle_loss)
        return {"loss": torch.mean(angle_loss)}

    def compose_joint_loss(self, render_loss, velocity_loss, step, coefficient=1e-5):
        # The jointly training loss is composed by the convex_combination:
        #   L = a * L1 + (1-a) * L2
        # alpha = math.pow(2.0, -coefficient * step)
        # loss = alpha * sfm_loss + (1 - alpha) * render_loss

        loss = render_loss
        return loss


    @gin.configurable()
    def compute_metrics_eval_img(
        self,
        rays: RayBundle,
        rb: RenderBuffer,
        target: RenderBuffer,
        # Configurable
        **kwargs
    ) -> Dict:
        # ray info
        alive_ray_mask = (rb.alpha.squeeze(-1) > 0).detach()
        rendering_samples_actual = rb.num_samples[0].item()
        ray_info = {
            'num_alive_ray': alive_ray_mask.long().sum().item(),
            'rendering_samples_actual': rendering_samples_actual,
            'num_rays': len(target),
        }

        rgb_eval_map = rb.rgb.unsqueeze(0).permute(0, 3, 1, 2)
        rgb_gt_map = target.rgb.unsqueeze(0).permute(0, 3, 1, 2)

        # quality
        quality = {'PSNR': peak_signal_noise_ratio(rb.rgb, target.rgb).item(),
                    'SSIM': pytorch_ssim.ssim(rgb_eval_map, rgb_gt_map).item(),
                   'LPIPS':self.lpips_loss(rgb_eval_map*2-1, rgb_gt_map*2-1).item()
                   }
        print(quality)
        return {**ray_info, **quality}

    @gin.configurable()
    def compute_metrics(
        self,
        rays: RayBundle,
        rb: RenderBuffer,
        target: RenderBuffer,
        # Configurable
        **kwargs
    ) -> Dict:
        # ray info
        alive_ray_mask = (rb.alpha.squeeze(-1) > 0).detach()
        rendering_samples_actual = rb.num_samples[0].item()
        ray_info = {
            'num_alive_ray': alive_ray_mask.long().sum().item(),
            'rendering_samples_actual': rendering_samples_actual,
            'num_rays': len(target),
        }
        # quality
        quality = {'PSNR': peak_signal_noise_ratio(rb.rgb, target.rgb).item()}
        return {**ray_info, **quality}

    @torch.no_grad()
    def pre_align_train_gt_cameras(self, block):
        NotImplementedError

    def clear_network_parameters(self):
        NotImplementedError
