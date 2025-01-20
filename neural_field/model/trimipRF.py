from typing import Callable

import gin
import torch
import nerfacc
from nerfacc import render_weight_from_density, accumulate_along_rays
from neural_field.model.RFModel import RFModel
from utils.ray import RayBundle
from utils.render_buffer import RenderBuffer
from neural_field.field.trimipRF import TriMipRF
import dataset.utils.poses as camera
from easydict import EasyDict as edict
import utils.Spline as Spline
from typing import Union, List, Dict
from dataset.utils import poses
import numpy as np


@gin.configurable()
class TriMipRFModel(RFModel):
    def __init__(
            self,
            aabb: Union[torch.Tensor, List[float]],
            train_frame_poses: dict,
            train_frame_times: dict,
            eval_frame_poses: dict,
            eval_frame_times: dict,
            read_out_time: float,
            period: float,
            samples_per_ray: int = 1024,
            occ_grid_resolution: int = 128,
            image_H: int = 480,
            pose_optimization: bool = False,
            spline_interpolation: bool = False,
            rolling_velocity: bool = False,
            beta_min: float = 0.01,
            poses_disturb: bool = True

    ) -> None:
        super().__init__(aabb=aabb, samples_per_ray=samples_per_ray)
        self.field = TriMipRF(beta_min)
        self.ray_sampler = nerfacc.OccupancyGrid(
            roi_aabb=self.aabb, resolution=occ_grid_resolution
        )

        self.feature_vol_radii = self.aabb_size[0] / 2.0
        self.register_buffer(
            "occ_level_vol",
            torch.log2(
                self.aabb_size[0]
                / occ_grid_resolution
                / 2.0
                / self.feature_vol_radii
            ),
        )

        # 创建要优化的pose变量
        self.pose_optimization = pose_optimization
        self.spline_interpolation = spline_interpolation
        self.poses_disturb = poses_disturb
        self.linear = True
        self.spline = False
        self.train_gt_times = torch.tensor(train_frame_times[0]).cuda(non_blocking=True)
        self.train_gt_poses = torch.tensor(train_frame_poses[0]).cuda(non_blocking=True)
        self.eval_gt_times = torch.tensor(eval_frame_times[0]).cuda(non_blocking=True)
        self.eval_gt_poses = torch.tensor(eval_frame_poses[0]).cuda(non_blocking=True)
        self.read_out_time = read_out_time
        self.period = period
        self.rolling_velocity = rolling_velocity
        self.image_H = image_H
        self.train_num_poses = len(self.train_gt_poses)

        if self.pose_optimization:
            self.se3_refine = torch.nn.Embedding(self.train_num_poses, 6).to('cuda')
            torch.nn.init.zeros_(self.se3_refine.weight)
            self.init_poses = torch.eye(3, 4).to('cuda')
            if self.poses_disturb:
                self.init_poses = self.train_gt_poses
                so3_noise = torch.rand(self.train_num_poses, 3, device='cuda') * 0.02
                t_noise = torch.rand(self.train_num_poses, 3, device='cuda') * 0.03
                self.pose_noise = torch.cat([camera.lie.so3_to_SO3(so3_noise), t_noise[..., None]], dim=-1)
                self.init_poses = camera.pose.compose_pair(self.pose_noise, self.init_poses)

        if self.spline_interpolation:
            self.init_poses = torch.eye(3, 4).unsqueeze(0).repeat(self.train_num_poses, 1, 1).to('cuda')
            if self.poses_disturb:
                self.init_poses = self.train_gt_poses
                torch.cuda.manual_seed(42)
                so3_noise = torch.rand(self.train_num_poses, 3, device='cuda') * 0.02
                t_noise = torch.rand(self.train_num_poses, 3, device='cuda') * 0.03
                self.pose_noise = torch.cat([camera.lie.so3_to_SO3(so3_noise), t_noise[..., None]], dim=-1)
                self.init_poses = camera.pose.compose_pair(self.pose_noise, self.init_poses)

            poses_start_se3 = camera.lie.SE3_to_se3(self.init_poses).to('cuda')

            if self.linear:
                pose_params = torch.cat([poses_start_se3, poses_start_se3[-1:]], dim=0)
            if self.spline:
                pose_params = torch.cat(
                    [poses_start_se3[:1], poses_start_se3, poses_start_se3[-1:], poses_start_se3[-1:]], dim=0)
            low, high = 1e-4, 1e-3
            pose_params = pose_params + (high - low) * torch.randn_like(pose_params) + low
            self.se3_refine = torch.nn.Embedding(pose_params.shape[0], 6).to('cuda')
            self.se3_refine.weight.data = torch.nn.Parameter(pose_params)

        if self.rolling_velocity:
            self.se3_refine = torch.nn.Embedding(self.train_num_poses, 6).to('cuda')
            torch.nn.init.zeros_(self.se3_refine.weight)
            self.se3_vel_refine = torch.nn.Embedding(self.train_num_poses, 6).to('cuda')
            torch.nn.init.zeros_(self.se3_vel_refine.weight)
            self.init_poses = torch.eye(3, 4).to('cuda')

            if self.poses_disturb:
                self.init_poses = self.train_gt_poses
                torch.cuda.manual_seed(42)
                so3_noise = torch.rand(self.train_num_poses, 3, device='cuda') * 0.02
                t_noise = torch.rand(self.train_num_poses, 3, device='cuda') * 0.03
                self.pose_noise = torch.cat([camera.lie.so3_to_SO3(so3_noise), t_noise[..., None]], dim=-1)
                self.init_poses = camera.pose.compose_pair(self.pose_noise, self.init_poses)

    @torch.no_grad()
    def get_all_training_poses(self):
        if self.spline_interpolation:
            cam_idxs = torch.tensor(range(self.train_num_poses)).cuda()
            row_idxs = torch.zeros_like(cam_idxs)
            if self.linear:
                pose_compose = self.get_linear_pose(cam_idxs, row_idxs, False)
            if self.spline:
                pose_compose = self.get_spline_pose(cam_idxs, row_idxs, False)
        else:
            se3_refine = self.se3_refine.weight
            pose_refine = camera.lie.se3_to_SE3(se3_refine)
            pose_compose = camera.pose.compose_pair(pose_refine, self.init_poses)
        return pose_compose, self.train_gt_poses

    def get_all_training_velocities(self):
        if self.rolling_velocity:
            vw = camera.lie.se3_to_SE3(self.se3_vel_refine.weight / self.image_H)
        else:
            vw = camera.lie.se3_to_SE3(torch.zeros_like(self.se3_refine.weight[:, :6]))
        return vw

    @torch.no_grad()
    def prealign_cameras(self, pose, pose_GT, device):

        pose_est = pose
        pose_gt = pose_GT
        # Align X1 to X0: X1to0 = (X1-t1)/s1@R.t()*s0+t0
        # 获取当前时间
        # current_time = datetime.now()
        #
        # current_time_str = current_time.strftime("%M_%S")
        # self.convert_to_idx_xyz_qxyzw(pose, current_time_str+'_1.txt')
        # self.convert_to_idx_xyz_qxyzw(pose_gt, 'gt.txt')
        self.convert_to_idx_xyz_qxyzw(pose, '1.txt')
        self.convert_to_idx_xyz_qxyzw(pose_gt, '2.txt')
        try:
            sim3 = camera.procrustes_analysis(pose_gt[:, :3, 3], pose_est[:, :3, 3])
        except:
            print("warning: SVD did not converge...")
            sim3 = edict(t0=0, t1=0, s0=1, s1=1, R=torch.eye(3, device=device))
            # align the camera poses

        t_aligned = (pose_est[:, :3, 3] - sim3.t1) / sim3.s1 @ sim3.R.t() * sim3.s0 + sim3.t0
        R_aligned = pose_est[..., :3] @ sim3.R

        pose_aligned = pose_est
        pose_aligned[:, :3, :3] = R_aligned
        pose_aligned[:, :3, 3] = t_aligned

        # R_align = align_rotations(pose_aligned[:, :3].cpu().numpy(), pose_gt[:, :3].cpu().numpy())
        # sim3.R_align = torch.tensor(R_align.transpose()).float().cuda()

        sim3.R_align = torch.tensor(np.eye(3)).float().cuda()

        pose_aligned[:, :3, :3] = sim3.R_align @ pose_aligned[:, :3, :3]

        return pose_aligned, sim3

    @torch.no_grad()
    def evaluate_camera_alignment(self, pose_gt, pose_aligned):
        # measure errors in rotation and translation
        R_aligned, t_aligned = pose_aligned.split([3, 1], dim=-1)
        R_GT, t_GT = pose_gt.split([3, 1], dim=-1)
        R_error = poses.rotation_distance(R_aligned, R_GT)
        t_error = (t_aligned - t_GT)[..., 0].norm(dim=-1)
        error = edict(R=R_error, t=t_error)
        return error

    @torch.no_grad()
    def pre_align_train_gt_cameras(self):
        pose, pose_gt = self.get_all_training_poses()
        pose_aligned, self.sim3 = self.prealign_cameras(pose, pose_gt, 'cuda')
        return pose_gt, pose_aligned, self.sim3

    def convert_to_idx_xyz_qxyzw(self, tensor, output_file):
        n, _, _ = tensor.size()
        results = []

        for i in range(n):
            # if i < 6 or i > n - 6:
            #     continue
            rotation_matrix = tensor[i, :, :3]
            translation_vector = tensor[i, :, 3]

            # Extract quaternion from rotation matrix
            qw = torch.sqrt(1 + rotation_matrix[0, 0] + rotation_matrix[1, 1] + rotation_matrix[2, 2]) / 2
            qx = (rotation_matrix[2, 1] - rotation_matrix[1, 2]) / (4 * qw)
            qy = (rotation_matrix[0, 2] - rotation_matrix[2, 0]) / (4 * qw)
            qz = (rotation_matrix[1, 0] - rotation_matrix[0, 1]) / (4 * qw)

            idx = i + 1  # Assuming index starts from 1
            x, y, z = translation_vector.tolist()

            result = f"{idx} {x} {y} {z} {qx} {qy} {qz} {qw}"
            results.append(result)

        with open(output_file, 'w') as file:
            file.write('\n'.join(results))

    def get_spline_pose(self, idx, row_idxs, training):
        if training:
            ts = row_idxs * self.read_out_time
        else:
            ts = torch.zeros_like(row_idxs) * self.read_out_time

        pose0 = self.se3_refine.weight[idx, :6]
        pose1 = self.se3_refine.weight[idx + 1, :6]
        pose2 = self.se3_refine.weight[idx + 2, :6]
        pose3 = self.se3_refine.weight[idx + 3, :6]
        spline_poses = Spline.SplineN_cubic(pose0, pose1, pose2, pose3, ts, self.period)

        return spline_poses

    def get_linear_pose(self, idx, row_idxs, training):
        if training:
            ts = row_idxs * self.read_out_time
        else:
            ts = torch.zeros_like(row_idxs) * self.read_out_time

        se3_start = self.se3_refine.weight[:-1, :6][idx]
        se3_end = self.se3_refine.weight[1:, :6][idx]
        spline_poses = Spline.SplineN_linear(se3_start, se3_end, ts, self.period)

        return spline_poses

    def get_vel_pose(self, idx, row_idxs, training):
        if training:
            # vw = camera.lie.se3_to_SE3((torch.tensor(
            #     [-0.01845715, -0.0136289, 0.0070197, -0.00613985, 0.04883738, -0.03979552]).unsqueeze(0).expand(
            #     self.train_num_poses, -1).to('cuda')
            #                             [idx] + self.se3_vel_refine.weight[idx]) * (
            #                                        row_idxs.view(-1, 1) / (self.image_H - 1)))
            vw = camera.lie.se3_to_SE3(self.se3_vel_refine.weight[idx] * ((row_idxs.view(-1, 1)) / (self.image_H - 1)))
            # vw = camera.lie.se3_to_SE3(torch.zeros_like(self.se3_vel_refine.weight[idx])).cuda()
            # ts = row_idxs * self.read_out_time
            # se3_end = self.se3_refine.weight[idx]
            # se3_start = torch.zeros_like(se3_end)
            # vw = Spline.SplineN_linear(se3_start, se3_end, ts, self.image_H)
        else:
            # vw = camera.lie.se3_to_SE3(self.se3_vel_refine.weight[idx] * 1 * ((row_idxs.view(-1, 1)) / self.image_H))
            p0 = torch.cat([self.init_poses[1:2],self.init_poses[:-1]], dim=0)
            p1 = torch.cat([self.init_poses[1:2],self.init_poses[1:]], dim=0)
            wv_SE3 = camera.pose.compose_pair(camera.pose.invert(p1), p0)
            tensor = camera.lie.SE3_to_se3(wv_SE3[idx])
            vw = camera.lie.se3_to_SE3(tensor * 0.0 * ((row_idxs.view(-1, 1)) / self.image_H))

            # vw = camera.lie.se3_to_SE3(self.se3_refine.weight[idx] * 1 * ((row_idxs.view(-1, 1)) / self.image_H))
            # print(self.se3_vel_refine.weight)
            # vw = camera.lie.se3_to_SE3(torch.zeros_like(self.se3_vel_refine.weight[idx])).cuda()
        return vw

    def get_vel_pose_sp(self, idx, row_idxs, training):
        if training:
            ts = row_idxs * self.read_out_time
        else:
            ts = torch.zeros_like(row_idxs) * self.read_out_time
            # vw = camera.lie.se3_to_SE3((torch.tensor(
            #     [-0.01845715, -0.0136289, 0.0070197, -0.00613985, 0.04883738, -0.03979552]).unsqueeze(0).expand(
            #     self.train_num_poses, -1).to('cuda')
            #                             [idx] + self.se3_vel_refine.weight[idx]) * (
            #                                        row_idxs.view(-1, 1) / (self.image_H - 1)))
        pose_refine = camera.lie.se3_to_SE3(self.se3_refine.weight)
        pose_compose = camera.pose.compose_pair(pose_refine, self.init_poses)
        se3_start = camera.lie.SE3_to_se3(pose_compose[idx])
        se3_end = se3_start + self.se3_vel_refine.weight[idx]
        spline_poses = Spline.SplineN_linear(se3_start, se3_end, ts, self.image_H)

        return spline_poses

    def get_pose(self, idx, training):
        if training:
            pose_refine = camera.lie.se3_to_SE3(self.se3_refine.weight)
            pose_compose = camera.pose.compose_pair(pose_refine, self.init_poses)
        else:
            pose_refine = camera.lie.se3_to_SE3(self.se3_refine.weight)
            pose_compose = camera.pose.compose_pair(pose_refine, self.init_poses)
            # pose_compose = self.init_poses
            # sim3 = self.sim3
            # t_aligned = (self.eval_gt_poses[:, :3, 3] - sim3.t0) / sim3.s0 @ sim3.R * sim3.s1 + sim3.t1
            # R_aligned = self.eval_gt_poses[:, :3, :3] @ self.sim3.R.t()
            # pose_compose = torch.zeros_like(self.eval_gt_poses)
            # pose_compose[:, :3, 3] = t_aligned
            # pose_compose[:, :3, :3] = R_aligned @ sim3.R_align.t()

        return pose_compose[idx]

    def before_iter(self, step):
        # update_ray_sampler
        self.ray_sampler.every_n_step(
            step=step,
            occ_eval_fn=lambda x:
            self.field.query_density(
                x=self.contraction(x),
                level_vol=torch.empty_like(x[..., 0]).fill_(self.occ_level_vol),
                step=step
            )['density']
            * self.render_step_size,
            occ_thre=5e-3,
        )

    @staticmethod
    def compute_ball_radii(distance, radiis, cos):
        inverse_cos = 1.0 / cos
        tmp = (inverse_cos * inverse_cos - 1).sqrt() - radiis
        sample_ball_radii = distance * radiis * cos / (tmp * tmp + 1.0).sqrt()
        return sample_ball_radii

    def forward(
            self,
            rays: RayBundle,
            cam_idxs,
            row_idxs,
            step=25000,
            alpha_thre=0.0,
            ray_marching_aabb=None,
            training=True,
    ):

        if self.pose_optimization:
            self.cam_poses = self.get_pose(cam_idxs, training)
            rays.origins = self.cam_poses[:, :3, -1]
            rays.directions = (self.cam_poses[:, :3, :3] @ rays.directions[..., None]).squeeze(-1)

        if self.rolling_velocity:
            # embedding camera poses
            self.rolling_poses = self.get_vel_pose(cam_idxs, row_idxs, training)
            self.cam_poses = self.get_pose(cam_idxs, training)

            # Twc' = Twc * Tcc'
            self.rolling_cam_poses = camera.pose.compose_pair(self.cam_poses, self.rolling_poses)
            # self.rolling_cam_poses = self.cam_poses

            # self.rolling_cam_poses = self.get_vel_pose_sp(cam_idxs, row_idxs, training)

            # self.rolling_cam_poses = camera.lie.se3_to_SE3(
            #     camera.lie.SE3_to_se3(self.rolling_poses) + camera.lie.SE3_to_se3(self.cam_poses))`
            rays.origins = self.rolling_cam_poses[:, :3, -1]
            rays.directions = (self.rolling_cam_poses[:, :3, :3] @ rays.directions[..., None]).squeeze(-1)

        # Ray sampling with occupancy grid
        if self.spline_interpolation:
            if self.linear:
                self.cam_poses = self.get_linear_pose(cam_idxs, row_idxs, training)
            if self.spline:
                self.cam_poses = self.get_spline_pose(cam_idxs, row_idxs, training)

            rays.origins = self.cam_poses[:, :3, -1]
            rays.directions = (self.cam_poses[:, :3, :3] @ rays.directions[..., None]).squeeze(-1)

        with torch.no_grad():
            def sigma_fn(t_starts, t_ends, ray_indices):
                ray_indices = ray_indices.long()
                t_origins = rays.origins[ray_indices]
                t_dirs = rays.directions[ray_indices]
                radiis = rays.radiis[ray_indices]
                cos = rays.ray_cos[ray_indices]

                distance = (t_starts + t_ends) / 2.0
                positions = t_origins + t_dirs * distance

                positions = self.contraction(positions)
                sample_ball_radii = self.compute_ball_radii(
                    distance, radiis, cos
                )
                level_vol = torch.log2(
                    sample_ball_radii / self.feature_vol_radii
                )  # real level should + log2(feature_resolution)

                return self.field.query_density(positions, level_vol, step)['density']

            ray_indices, t_starts, t_ends = nerfacc.ray_marching(
                rays.origins,
                rays.directions,
                scene_aabb=self.aabb,
                grid=self.ray_sampler,
                sigma_fn=sigma_fn,
                render_step_size=self.render_step_size,
                stratified=training,
                early_stop_eps=1e-4,
            )

        # Ray rendering
        def rgb_sigma_fn(t_starts, t_ends, ray_indices):
            t_origins = rays.origins[ray_indices]
            t_dirs = rays.directions[ray_indices]
            radiis = rays.radiis[ray_indices]
            cos = rays.ray_cos[ray_indices]

            distance = (t_starts + t_ends) / 2.0
            positions = t_origins + t_dirs * distance

            positions = self.contraction(positions)
            sample_ball_radii = self.compute_ball_radii(distance, radiis, cos)

            # 从sample的角度看，如果越大，说明越远，属于低频部分，应该是层数越高
            # 从feature_vol的角度来看，如果越大，说明sample越小，属于高频部分，层数越低
            level_vol = torch.log2(
                sample_ball_radii / self.feature_vol_radii
            )  # real level should + log2(feature_resolution)
            res = self.field.query_density(
                x=positions,
                level_vol=level_vol,
                return_feat=True,
                step=step
            )
            density, feature = res['density'], res['feature']
            # rgb = self.field.query_rgb(dir=t_dirs, embedding=feature)['rgb']
            rgb = self.field.query_rgb(dir=-t_dirs, embedding=feature)['rgb']
            # uncert = self.field.query_uncertainty(dir=t_dirs, embedding=feature)['uncertainty']
            return rgb, density

        return self.rendering(
            t_starts,
            t_ends,
            ray_indices,
            rays,
            rgb_sigma_fn=rgb_sigma_fn
        )

    def rendering(
            self,
            # ray marching results
            t_starts: torch.Tensor,
            t_ends: torch.Tensor,
            ray_indices: torch.Tensor,
            rays: RayBundle,
            # radiance field
            rgb_sigma_fn: Callable = None  # rendering options
    ) -> RenderBuffer:
        n_rays = rays.origins.shape[0]
        # Query sigma/alpha and color with gradients
        rgbs, sigmas = rgb_sigma_fn(t_starts, t_ends, ray_indices.long())

        # Rendering
        weights = render_weight_from_density(
            t_starts,
            t_ends,
            sigmas,
            ray_indices=ray_indices,
            n_rays=n_rays,
        )
        sample_buffer = {
            'num_samples': torch.as_tensor(
                [len(t_starts)], dtype=torch.int32, device=rgbs.device
            ),
        }
        # Rendering: accumulate rgbs, opacities, and depths along the rays.
        colors = accumulate_along_rays(
            weights, ray_indices=ray_indices, values=rgbs, n_rays=n_rays
        )

        # uncertainties = accumulate_along_rays(
        #     weights, ray_indices=ray_indices, values=uncert, n_rays=n_rays
        # )  # + self.beta_min

        opacities = accumulate_along_rays(
            weights, values=None, ray_indices=ray_indices, n_rays=n_rays
        )
        opacities.clamp_(
            0.0, 1.0
        )  # sometimes it may slightly bigger than 1.0, which will lead abnormal behaviours

        depths = accumulate_along_rays(
            weights,
            ray_indices=ray_indices,
            values=(t_starts + t_ends) / 2.0,
            n_rays=n_rays,
        )

        depths = (
                depths * rays.ray_cos
        )  # from distance to real depth (z value in camera space)

        return RenderBuffer(
            rgb=colors,
            alpha=opacities,
            depth=depths,
            **sample_buffer,
            _static_field=set(sample_buffer),
        )

    @gin.configurable()
    def get_optimizer(
            self, lr=2e-3, weight_decay=1e-5, feature_lr_scale=10.0, **kwargs
    ):
        params_list = []
        pose_params_list = []
        params_list.append(
            dict(
                params=self.field.encoding.parameters(),
                lr=lr * feature_lr_scale,
            )
        )
        params_list.append(
            dict(params=self.field.direction_encoding.parameters(), lr=lr)
        )

        params_list.append(dict(params=self.field.mlp_base.parameters(), lr=lr))
        params_list.append(dict(params=self.field.mlp_head.parameters(), lr=lr))
        # params_list.append(dict(params=self.field.mlp_head2.parameters(), lr=lr))

        if self.rolling_velocity:
            pose_params_list.append(dict(params=self.se3_vel_refine.parameters(), lr=lr*0.1))
            pose_params_list.append(dict(params=self.se3_refine.parameters(), lr=lr))

        if self.pose_optimization:
            pose_params_list.append(dict(params=self.se3_refine.parameters(), lr=lr))

        if self.spline_interpolation:
            pose_params_list.append(dict(params=self.se3_refine.parameters(), lr=lr))

        optim = torch.optim.AdamW(
            params_list,
            weight_decay=weight_decay,
            **kwargs,
            eps=1e-15,
        )

        optim_pose = torch.optim.AdamW(
            pose_params_list,
            weight_decay=weight_decay,
            **kwargs,
            eps=1e-15,
        )

        return optim, optim_pose

    @gin.configurable()
    def get_eval_optimizer(
            self, lr=2e-3, weight_decay=1e-5, feature_lr_scale=10.0, **kwargs
    ):
        pose_params_list = []

        if self.rolling_velocity:
            # pose_params_list.append(dict(params=self.se3_vel_refine.parameters(), lr=lr))
            pose_params_list.append(dict(params=self.se3_refine.parameters(), lr=lr))

        if self.pose_optimization:
            pose_params_list.append(dict(params=self.se3_refine.parameters(), lr=lr))

        if self.spline_interpolation:
            pose_params_list.append(dict(params=self.se3_refine.parameters(), lr=lr))

        optim_pose = torch.optim.AdamW(
            pose_params_list,
            weight_decay=weight_decay,
            **kwargs,
            eps=1e-15,
        )

        return optim_pose

    def clear_network_parameters(self):
        self.field.init_parameters()
        optim, optim_pose = self.get_optimizer()
        return optim, optim_pose

    def compute_sfm_loss(self,
                         key_points) -> Dict:

        if key_points != {}:
            pose0 = self.cam_poses[key_points.cam0]
            pose1 = self.cam_poses[key_points.cam1]
            kpt0 = key_points.kpt0
            kpt1 = key_points.kpt1

        return {"loss": torch.tensor(0)}

    # @torch.no_grad()
    # def prealign_cameras(self, pose, pose_GT, device):
    #     pose_est = pose
    #     pose_gt = pose_GT
    #     # Align X1 to X0: X1to0 = (X1-t1)/s1@R.t()*s0+t0
    #     try:
    #         sim3 = camera.procrustes_analysis(pose_gt[:, :3, 3], pose_est[:, :3, 3])
    #     except:
    #         print("warning: SVD did not converge...")
    #         sim3 = edict(t0=0, t1=0, s0=1, s1=1, R=torch.eye(3, device=device))
    #         # align the camera poses
    #
    #     t_aligned = (pose_est[:, :3, 3] - sim3.t1) / sim3.s1 @ sim3.R.t() * sim3.s0 + sim3.t0
    #     R_aligned = pose_est[..., :3] @ sim3.R
    #
    #     pose_aligned = pose_est
    #     pose_aligned[:, :3, :3] = R_aligned
    #     pose_aligned[:, :3, 3] = t_aligned
    #
    #     return pose_aligned, sim3
