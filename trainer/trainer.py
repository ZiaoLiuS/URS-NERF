import time

import gin
import numpy as np
import torch
from pathlib import Path
from loguru import logger
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, List
import os, sys, time
from neural_field.model.RFModel import RFModel
from utils.writer import TensorboardWriter
from utils.colormaps import apply_depth_colormap
import dataset.utils.io as data_io
import matplotlib.pyplot as plt
from dataset.utils import utils
from dataset.utils import util_vis
import shutil
from dataset.utils import poses
from easydict import EasyDict as edict
import dataset.utils.poses as camera


@gin.configurable()
class Trainer:
    def __init__(
            self,
            model: RFModel,
            # configurable
            base_exp_dir: str = 'experiments',
            exp_name: str = 'Tri-MipRF',
            block_max_steps: int = 5000,
            log_step: int = 500,
            eval_step: int = 500,
            target_sample_batch_size: int = 65536,
            test_chunk_size: int = 8192,
            dynamic_batch_size: bool = True,
            num_rays: int = 8192,
            varied_eval_img: bool = True,
    ):
        self.model = model.cuda()
        self.max_steps = block_max_steps
        self.target_sample_batch_size = target_sample_batch_size
        # exp_dir
        self.exp_dir = Path(base_exp_dir) / exp_name
        self.log_step = log_step
        self.eval_step = eval_step
        self.test_chunk_size = test_chunk_size
        self.dynamic_batch_size = dynamic_batch_size
        self.num_rays = num_rays
        self.varied_eval_img = varied_eval_img

        self.writer = TensorboardWriter(log_dir=self.exp_dir)

        self.optimizer, self.pose_optimizer = self.model.get_optimizer()
        self.scheduler, self.pose_scheduler = self.get_scheduler()

        self.grad_scaler = torch.cuda.amp.GradScaler(2 ** 10)

        # Save configure
        conf = gin.operative_config_str()
        logger.info(conf)
        self.save_config(conf)

    def update_train_loader(self, train_loader: DataLoader):
        self.train_loader = train_loader

    def update_eval_loader(self, eval_loader: DataLoader):
        self.eval_loader = eval_loader

    def train_iter(self, step: int, data: Dict, logging=False, downLevel=0):
        tic = time.time()
        cam_rays = data['cam_rays']
        num_rays = min(self.num_rays, len(cam_rays))
        cam_rays = cam_rays[:num_rays].cuda(non_blocking=True)
        target = data['target'][:num_rays].cuda(non_blocking=True)
        cam_idxs = data['idx'][:num_rays].cuda(non_blocking=True)
        row_idxs = data['row_idx'][:num_rays].cuda(non_blocking=True)

        rb = self.model(cam_rays, cam_idxs, row_idxs, step, training=True)

        # compute loss
        loss_dict = {}

        render_loss = self.model.compute_loss(cam_rays, rb, target)
        loss_dict["render_loss"] = render_loss["loss"]

        if downLevel == 0 and self.model.rolling_velocity:
            # velocity_prior_loss = self.model.get_vel_prior_loss()
            # loss_dict["velocity_loss"] = velocity_prior_loss["loss"]
            # loss_all = self.model.compose_joint_loss(loss_dict["render_loss"], loss_dict["velocity_loss"], step)
            loss_all = render_loss["loss"]
        else:
            loss_all = render_loss["loss"]

        metrics = self.model.compute_metrics(cam_rays, rb, target)
        if 0 == metrics.get("rendering_samples_actual", -1):
            return metrics

        # update
        self.optimizer.zero_grad()
        self.pose_optimizer.zero_grad()
        self.grad_scaler.scale(loss_all).backward()
        # 非叶子节点查看是否为nan
        # self.model.cam_poses.retain_grad()
        # grad 中查找NAN的值给成0
        # self.model.se3_refine.weight.grad = torch.where(torch.isnan(self.model.se3_refine.weight.grad),
        #                                                 torch.zeros_like(self.model.se3_refine.weight.grad),
        #                                                 self.model.se3_refine.weight.grad)

        self.optimizer.step()
        self.pose_optimizer.step()

        # todo debug 对学习率等参数进行调整
        # if step < self.max_steps*3 // 4:
        #     self.pose_optimizer.step()
        #
        # with torch.no_grad():
        #     if step % (self.max_steps // 6) == 0 and step < self.max_steps *5 // 8:
        #         self.model.init_poses, _ = self.model.get_all_training_poses()
        #         torch.nn.init.normal_(self.model.se3_refine.weight, mean=0, std=0.01 * (0.75 - step / self.max_steps))
        #     if step == (self.max_steps // 2):
        #         self.model.field.encoding.init_parameters()

        # logging
        if logging:
            with torch.no_grad():
                iter_time = time.time() - tic
                remaining_time = (self.max_steps - step) * iter_time
                status = {
                    'lr': self.optimizer.param_groups[0]["lr"],
                    'step': step,
                    'iter_time': iter_time,
                    'ETA': remaining_time,
                }
                self.writer.write_scalar_dicts(
                    ['loss', 'metrics', 'status'],
                    [
                        {k: v.item() for k, v in loss_dict.items()},
                        metrics,
                        status,
                    ],
                    step,
                )
        return metrics

    def pose_optimize_iter(self, step: int, data: Dict, logging=False):
        tic = time.time()
        cam_rays = data['cam_rays']
        num_rays = min(self.num_rays, len(cam_rays))
        cam_rays = cam_rays[:num_rays].cuda(non_blocking=True)
        target = data['target'][:num_rays].cuda(non_blocking=True)
        cam_idxs = data['idx'][:num_rays].cuda(non_blocking=True)
        row_idxs = data['row_idx'][:num_rays].cuda(non_blocking=True)
        rb = self.model(cam_rays, cam_idxs, row_idxs, training=False)

        # compute loss
        loss_dict = self.model.compute_loss(cam_rays, rb, target)
        metrics = self.model.compute_metrics(cam_rays, rb, target)
        if 0 == metrics.get("rendering_samples_actual", -1):
            return metrics

        # update
        self.pose_optimizer.zero_grad()
        self.grad_scaler.scale(loss_dict['loss']).backward()
        self.pose_optimizer.step()

        if logging:
            with torch.no_grad():
                iter_time = time.time() - tic
                remaining_time = (self.max_steps - step) * iter_time
                status = {
                    'lr': self.pose_optimizer.param_groups[0]["lr"],
                    'step': step,
                    'iter_time': iter_time,
                    'ETA': remaining_time,
                }
                self.writer.write_scalar_dicts(
                    ['loss', 'metrics', 'status'],
                    [
                        {k: v.item() for k, v in loss_dict.items()},
                        metrics,
                        status,
                    ],
                    step,
                )

        return metrics

    def evaluate_test_time_photometric_optim(self, max_steps, eval_frame_poses, eval_frame_times, image_H):
        logger.info("==> evaluate test photometric optim ...")
        self.max_steps = max_steps
        # clear the pose variables
        _, _, sim3 = self.model.pre_align_train_gt_cameras()
        t_aligned = (eval_frame_poses[:, :3, 3] - sim3.t0) / sim3.s0 @ sim3.R * sim3.s1 + sim3.t1
        R_aligned = eval_frame_poses[:, :3, :3] @ sim3.R.t()
        R_aligned = R_aligned @ sim3.R_align.t()
        pose_compose = torch.zeros_like(eval_frame_poses)
        pose_compose[:, :3, 3] = t_aligned
        pose_compose[:, :3, :3] = R_aligned
        pose_compose = eval_frame_poses

        eval_num_poses = len(eval_frame_poses)
        poses = torch.eye(3, 4).repeat(eval_num_poses, 1, 1).cuda()
        poses[:eval_num_poses, :, :] = pose_compose.cuda()

        self.model.train_gt_poses = poses
        self.model.train_gt_times = torch.tensor(eval_frame_times)
        self.model.train_num_poses = len(self.model.train_gt_poses)
        self.model.image_H = image_H

        if self.model.pose_optimization:
            self.model.init_poses = poses
            self.model.se3_refine = torch.nn.Embedding(eval_num_poses, 6).to('cuda')
            torch.nn.init.zeros_(self.model.se3_refine.weight)

        if self.model.rolling_velocity:
            self.model.init_poses = poses
            self.model.se3_refine = torch.nn.Embedding(eval_num_poses, 6).to('cuda')
            torch.nn.init.zeros_(self.model.se3_refine.weight)
            self.model.se3_vel_refine = torch.nn.Embedding(eval_num_poses, 6).to('cuda')
            torch.nn.init.zeros_(self.model.se3_vel_refine.weight)

        if self.model.spline_interpolation:
            self.model.init_poses = poses
            poses_start_se3 = camera.lie.SE3_to_se3(self.model.init_poses).to('cuda')
            if self.model.linear:
                pose_params = torch.cat([poses_start_se3, poses_start_se3[:1]], dim=0)
            if self.model.spline:
                pose_params = torch.cat(
                    [poses_start_se3[:1], poses_start_se3, poses_start_se3[-1:], poses_start_se3[-1:]], dim=0)

            low, high = 1e-4, 1e-3
            pose_params = pose_params + (high - low) * torch.randn_like(pose_params) + low
            self.model.se3_refine = torch.nn.Embedding(pose_params.shape[0], 6).to('cuda')
            self.model.se3_refine.weight.data = torch.nn.Parameter(pose_params)

        iter_eval_loader = iter(self.train_loader)
        self.model.train()

        self.pose_optimizer = self.model.get_eval_optimizer()
        _, self.pose_scheduler = self.get_scheduler()

        for step in range(max_steps):
            metrics = self.pose_optimize_iter(
                step,
                data=next(iter_eval_loader),
                logging=(step % self.log_step == 0 and step > 0)
                        or (step == 100),
            )
            if 0 == metrics.get("rendering_samples_actual", -1):
                continue
            self.pose_scheduler.step()

        return []

    @torch.no_grad()
    def eval_img(self, data, compute_metrics=True, training=False):

        # 在有了camera ray 之后，恢复pose
        cam_rays = data['cam_rays'].cuda(non_blocking=True)
        target = data['target'].cuda(non_blocking=True)
        cam_idxs = data['idx'].cuda(non_blocking=True)
        row_idxs = data['row_idx'].cuda(non_blocking=True)

        final_rb = None
        flatten_rays = cam_rays.reshape(-1)
        flatten_target = target.reshape(-1)
        flatten_cam_idxs = cam_idxs.reshape(-1)
        flatten_row_idxs = row_idxs.reshape(-1)

        pose_gt, pose_aligned, sim3 = self.model.pre_align_train_gt_cameras()

        error = self.evaluate_camera_alignment(pose_gt, pose_aligned)
        t_mse = torch.mean((error.t) ** 2)  # Mean Squared Error
        t_rmse = torch.sqrt(t_mse)  # Root
        r_mse = torch.mean((error.R) ** 2)  # Mean Squared Error
        r_rmse = torch.sqrt(r_mse)  # Root

        print("--------------------------")
        print("rot:        {:8.3f}   rmse: {:10.5f}  deg".format(np.rad2deg(error.R.mean().cpu()),
                                                                 np.rad2deg(r_rmse.cpu())))
        print("trans: mean {:10.5f}, rmse: {:10.5f}  m".format(error.t.mean(), t_rmse))
        print("--------------------------")

        with open(str(self.exp_dir) + "pose_error.txt", 'a') as file:
            file.write(str(float(t_rmse)) + "\n")

        for i in range(0, len(cam_rays), self.test_chunk_size):
            rb = self.model(
                flatten_rays[i: i + self.test_chunk_size],
                flatten_cam_idxs[i: i + self.test_chunk_size],
                flatten_row_idxs[i: i + self.test_chunk_size],
                training=training)
            final_rb = rb if final_rb is None else final_rb.cat(rb)
        final_rb = final_rb.reshape(cam_rays.shape)
        metrics = None
        if compute_metrics:
            metrics = self.model.compute_metrics_eval_img(cam_rays, final_rb, target)
        return metrics, final_rb, target

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
    def eval(
            self,
            save_results: bool = False,
            rendering_channels: List[str] = ["rgb"],
    ):
        # ipdb.set_trace()
        logger.info("==> Start evaluation on testset ...")
        if save_results:
            res_dir = self.exp_dir / 'rendering'
            res_dir.mkdir(parents=True, exist_ok=True)
            results = {"names": []}
            results.update({k: [] for k in rendering_channels})

        self.model.eval()
        metrics = []
        for idx, data in enumerate(tqdm(self.eval_loader)):
            # if idx < 0 or idx > 90:
            #     continue
            metric, rb, target = self.eval_img(data, training=False)
            metrics.append(metric)
            if save_results:
                results["names"].append(data['name'])
                for channel in rendering_channels:
                    if hasattr(rb, channel):
                        values = getattr(rb, channel).cpu().numpy()
                        if 'depth' == channel:
                            values = (values * 10000.0).astype(
                                np.uint16
                            )  # scale the depth by 10k, and save it as uint16 png images
                        results[channel].append(values)
                    else:
                        raise NotImplementedError
            del rb
        if save_results:
            for idx, name in enumerate(tqdm(results['names'])):
                for channel in rendering_channels:
                    channel_path = res_dir / channel
                    data = results[channel][idx]
                    data_io.write_rendering(data, channel_path, name)

        metrics = {k: [dct[k] for dct in metrics] for k in metrics[0]}
        logger.info("==> Evaluation done")
        for k, v in metrics.items():
            metrics[k] = sum(v) / len(v)
        self.writer.write_scalar_dicts(['benchmark'], [metrics], 0)
        self.writer.tb_writer.close()

    def save_config(self, config):
        dest = self.exp_dir / 'config.gin'
        if dest.exists():
            return
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        with open(self.exp_dir / 'config.gin', 'w') as f:
            f.write(config)
        md_config_str = gin.config.markdown(config)
        self.writer.write_config(md_config_str)
        self.writer.tb_writer.flush()

    def save_ckpt(self, iterNum):
        dest = self.exp_dir / 'model.ckpt'
        logger.info('==> Saving checkpoints to ' + str(dest))
        torch.save(
            {
                "model": self.model.state_dict(),
            },
            dest,
        )
        shutil.copy("{0}/model.ckpt".format(self.exp_dir),
                    "{0}/model{1}.ckpt".format(self.exp_dir, iterNum))  # if ep is None, track it instead

    def load_ckpt(self):
        dest = self.exp_dir / 'model.ckpt'
        loaded_state = torch.load(dest, map_location="cpu")
        logger.info('==> Loading checkpoints from ' + str(dest))
        self.model.load_state_dict(loaded_state['model'])

    @gin.configurable()
    def get_mlp_scheduler(self, gamma=0.6, **kwargs):
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer,
            milestones=[
                # self.max_steps *5// 12,
                self.max_steps // 2,
                self.max_steps * 3 // 4,
                self.max_steps * 5 // 6,
                self.max_steps * 9 // 10,
            ],
            gamma=gamma,
            **kwargs,
        )

        return scheduler

    @gin.configurable()
    def get_scheduler(self, gamma=0.6, **kwargs):
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer,
            milestones=[
                # self.max_steps *5// 12,
                self.max_steps // 2,
                self.max_steps * 3 // 4,
                self.max_steps * 5 // 6,
                self.max_steps * 9 // 10,
                # self.max_steps * 11 // 12,
            ],
            gamma=gamma,
            **kwargs,
        )
        pose_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.pose_optimizer,
            milestones=[
                self.max_steps // 12,
                self.max_steps // 6,
                self.max_steps // 4,
                self.max_steps // 3,
                # self.max_steps * 3 // 4,
            ],
            gamma=gamma,
            **kwargs,
        )
        return scheduler, pose_scheduler

    @torch.no_grad()
    def generate_videos_pose(self):
        fig = plt.figure(figsize=(18, 8))
        cam_path = "{}/poses".format(self.exp_dir)
        os.makedirs(cam_path, exist_ok=True)
        ep_list = []
        for ep in range(0, self.max_steps, 1):
            if ep % self.eval_step != 0 or ep == 0:
                continue
            # load checkpoint (0 is random init)
            dest = str(self.exp_dir) + '/model' + str(ep) + '.ckpt'
            print(dest)
            loaded_state = torch.load(dest, map_location="cpu")
            self.model.load_state_dict(loaded_state['model'])
            pose_gt, pose_aligned, sim3 = self.model.pre_align_train_gt_cameras()
            pose_gt = pose_gt.detach().cpu()
            pose_aligned = pose_aligned.detach().cpu()
            util_vis.plot_save_poses(cam_depth=0.3, fig=fig, pose=pose_aligned, pose_ref=pose_gt, path=cam_path, ep=ep)

        plt.close()
        # write videos
        print("writing videos...")
        list_fname = "{}/temp.list".format(cam_path)
        with open(list_fname, "w") as file:
            for ep in ep_list: file.write("file {}.png\n".format(ep))
        cam_vid_fname = "{}/poses.mp4".format(cam_path)
        os.system(
            "ffmpeg -y -r 4 -f concat -i {0} -pix_fmt yuv420p {1} >/dev/null 2>&1".format(list_fname, cam_vid_fname))
        os.remove(list_fname)
        pass

    def load_max_step_model(self):
        dest = str(self.exp_dir) + '/model' + '.ckpt'
        print('Load step:', dest)
        loaded_state = torch.load(dest, map_location="cuda")
        self.model.load_state_dict(loaded_state['model'])

    def fit(self, downLevel):
        logger.info("==> Start training ...")

        # todo debug modifty here
        self.scheduler = self.get_mlp_scheduler()

        iter_train_loader = iter(self.train_loader)
        iter_eval_loader = iter(self.eval_loader)
        eval_0 = next(iter_eval_loader)
        self.model.train()
        for step in range(self.max_steps):
            self.model.before_iter(step)
            metrics = self.train_iter(
                step,
                data=next(iter_train_loader),
                logging=(step % self.log_step == 0 and step > 0)
                        or (step == 100),
                downLevel=downLevel
            )

            if 0 == metrics.get("rendering_samples_actual", -1):
                continue

            self.pose_scheduler.step()
            self.scheduler.step()
            if self.dynamic_batch_size:
                rendering_samples_actual = metrics.get(
                    "rendering_samples_actual",
                    self.target_sample_batch_size,
                )
                self.num_rays = (
                        self.num_rays
                        * self.target_sample_batch_size
                        // rendering_samples_actual
                        + 1
                )

            self.model.after_iter(step)
            if step > 0 and step % self.eval_step == 0:
                self.model.eval()
                metrics, final_rb, target = self.eval_img(
                    next(iter_eval_loader) if self.varied_eval_img else eval_0,
                    compute_metrics=True, training=False
                )
                self.writer.write_scalar_dicts(['eval'], [metrics], step)
                self.writer.write_image('eval/rgb', final_rb.rgb, step)
                self.writer.write_image('gt/rgb', target.rgb, step)
                self.writer.write_image(
                    'eval/depth',
                    apply_depth_colormap(final_rb.depth),
                    step,
                )
                self.writer.write_image('eval/alpha', final_rb.alpha, step)

                self.model.train()
                self.save_ckpt(step)

        logger.info('==> Training done!')
        self.generate_videos_pose()
        estimated_poses, gt_poses = self.model.get_all_training_poses()
        estimated_velocities = self.model.get_all_training_velocities()

        self.model.convert_to_idx_xyz_qxyzw(estimated_poses, str(self.exp_dir) + '/est.txt')
        self.model.convert_to_idx_xyz_qxyzw(gt_poses, str(self.exp_dir) + '/gt.txt')
        return estimated_poses, estimated_velocities

    @torch.no_grad()
    def update_estimated_pose(self, pose_prior):
        # t = pose_prior[:, :, -1]
        # # 计算欧几里得距离
        # distances = torch.norm(t[:, :, None] - t[:, None, :], dim=1)
        # # 计算平均距离
        # avg_distance = distances.mean()
        # pose_prior[:, :, -1] = t * 0.5 / avg_distance.item()
        self.model.init_poses = pose_prior

        if self.model.spline_interpolation:
            poses_start_se3 = camera.lie.SE3_to_se3(self.model.init_poses).to('cuda')
            if self.model.linear:
                pose_params = torch.cat([poses_start_se3, poses_start_se3[:1]], dim=0)
            if self.model.spline:
                pose_params = torch.cat(
                    [poses_start_se3[:1], poses_start_se3, poses_start_se3[-1:], poses_start_se3[-1:]], dim=0)
            low, high = 1e-4, 1e-3
            pose_params = pose_params + (high - low) * torch.randn_like(pose_params) + low
            self.model.se3_refine = torch.nn.Embedding(pose_params.shape[0], 6).to('cuda')
            self.model.se3_refine.weight.data = torch.nn.Parameter(pose_params)

    def update_image_H(self, image_H):
        self.model.image_H = image_H

    # def clear_train_parameters(self):
    #     self.model.get_parameter()

    # def train_block(self, block, max_step=5000):
    #     logger.info("==> Start training ...")
    #     self.block = block
    #     self.max_steps = max_step
    #     iter_train_loader = iter(self.train_loader)
    #     iter_eval_loader = iter(self.eval_loader)
    #     eval_0 = next(iter_eval_loader)
    #     self.model.train()
    #     for step in range(self.max_steps):
    #         self.model.before_iter(step)
    #         metrics = self.train_iter(
    #             step,
    #             data=next(iter_train_loader),
    #             logging=(step % self.log_step == 0 and step > 0)
    #                     or (step == 100),
    #         )
    #
    #         if 0 == metrics.get("rendering_samples_actual", -1):
    #             continue
    #
    #         self.pose_scheduler.step()
    #         self.scheduler.step()
    #         if self.dynamic_batch_size:
    #             rendering_samples_actual = metrics.get(
    #                 "rendering_samples_actual",
    #                 self.target_sample_batch_size,
    #             )
    #             self.num_rays = (
    #                     self.num_rays
    #                     * self.target_sample_batch_size
    #                     // rendering_samples_actual
    #                     + 1
    #             )
    #
    #         self.model.after_iter(step)
    #
    #         # todo debug 先去掉保存model和生成pose的代码
    #         if step > 0 and step % self.eval_step == 0:
    #             self.model.eval()
    #             metrics, final_rb, target = self.eval_img(
    #                 next(iter_eval_loader) if self.varied_eval_img else eval_0,
    #                 compute_metrics=True,
    #             )
    #             self.writer.write_scalar_dicts(['eval'], [metrics], step)
    #             self.writer.write_image('eval/rgb', final_rb.rgb, step)
    #             self.writer.write_image('gt/rgb', target.rgb, step)
    #             self.writer.write_image(
    #                 'eval/depth',
    #                 apply_depth_colormap(final_rb.depth),
    #                 step,
    #             )
    #             self.writer.write_image('eval/alpha', final_rb.alpha, step)
    #
    #             self.model.train()
    #             self.save_ckpt(step)
    #
    #     logger.info('==> Training done!')
    #     self.generate_videos_pose()

    # def clear_network_parameters(self):
    #     # todo (bx) build new netword parameters and return new optimizer
    #     self.optimizer, self.pose_optimizer = self.model.clear_network_parameters()
    #     # todo (bx) build new rate learn scheduler
    #     self.scheduler, self.pose_scheduler = self.get_scheduler()

    # @torch.no_grad()
    # def eval(
    #         self,
    #         max_step=5000,
    #         save_results: bool = False,
    #         rendering_channels: List[str] = ["rgb"],
    # ):
    #     # ipdb.set_trace()
    #     logger.info("==> Start evaluation on testset ...")
    #
    #     pose_gt, pose_aligned, sim3 = self.model.pre_align_train_gt_cameras()
    #     pose_trans_error = pose_gt[..., 3] - pose_aligned[..., 3]
    #     error = self.evaluate_camera_alignment(pose_gt, pose_aligned)
    #     print("--------------------------")
    #     print("rot:   {:8.3f} deg".format(np.rad2deg(error.R.mean().cpu())))
    #     print("trans: {:10.5f} m".format(error.t.mean()))
    #     print("--------------------------")
    #     if save_results:
    #         res_dir = self.exp_dir / 'rendering'
    #         res_dir.mkdir(parents=True, exist_ok=True)
    #         results = {"names": []}
    #         results.update({k: [] for k in rendering_channels})
    #         # dump numbers
    #         quant_fname = "{}/quant_pose.txt".format(self.exp_dir)
    #         with open(quant_fname, "w") as file:
    #             for i, (err_R, err_t) in enumerate(zip(error.R, error.t)):
    #                 file.write("{} {} {}\n".format(i, err_R.item(), err_t.item()))
    #
    #     # self.evaluate_test_time_photometric_optim(max_step)
    #
    #     self.model.eval()
    #     metrics = []
    #     for idx, data in enumerate(tqdm(self.eval_loader)):
    #         metric, rb, target = self.eval_full(data)
    #         metrics.append(metric)
    #         if save_results:
    #             results["names"].append(data['name'])
    #             for channel in rendering_channels:
    #                 if hasattr(rb, channel):
    #                     values = getattr(rb, channel).cpu().numpy()
    #                     if 'depth' == channel:
    #                         values = (values * 10000.0).astype(
    #                             np.uint16
    #                         )  # scale the depth by 10k, and save it as uint16 png images
    #                     results[channel].append(values)
    #                 else:
    #                     raise NotImplementedError
    #         del rb
    #     if save_results:
    #         for idx, name in enumerate(tqdm(results['names'])):
    #             for channel in rendering_channels:
    #                 channel_path = res_dir / channel
    #                 data = results[channel][idx]
    #                 data_io.write_rendering(data, channel_path, name)
    #
    #     metrics = {k: [dct[k] for dct in metrics] for k in metrics[0]}
    #     logger.info("==> Evaluation done")
    #     for k, v in metrics.items():
    #         metrics[k] = sum(v) / len(v)
    #     self.writer.write_scalar_dicts(['benchmark'], [metrics], 0)
    #     self.writer.tb_writer.close()
