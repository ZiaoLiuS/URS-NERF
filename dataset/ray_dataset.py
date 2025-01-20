from concurrent.futures import ThreadPoolExecutor
import multiprocessing
from itertools import combinations

import gin
import ipdb
import numpy as np
import torch
from loguru import logger
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

from tqdm import tqdm
from easydict import EasyDict as edict
import os
from utils.pose_filter import PoseFilter
from dataset.parsers import get_parser
from dataset.utils import io as data_io
from thirdparty.LightGlue.lightglue import SuperPoint, LightGlue
from thirdparty.LightGlue.lightglue.utils import load_image, rbd
from utils.ray import RayBundle
from utils.render_buffer import RenderBuffer
from utils.tensor_dataclass import TensorDataclass
import pickle


@gin.configurable()
class RayDataset(Dataset):
    def __init__(
            self,
            base_path: str,
            scene: str = 'lego',
            scene_type: str = 'nerf_synthetic_multiscale',
            split: str = 'train',
            num_rays: int = 8192,
            down_level: int = 0,
            is_eval: bool = False,
            is_rolling: bool = True,
            use_colmap_pose: bool = False,
            **kwargs
    ):
        self.base_path = base_path
        super().__init__()
        parser = get_parser(scene_type)
        data_source = parser(
            base_path=Path(base_path), scene=scene, split=split, down_level=down_level, is_eval=is_eval,
            is_rolling=is_rolling, **kwargs
        )
        self.training = split.find('train') >= 0

        self.cameras = data_source['cameras']
        self.read_out_time = data_source['read_out_time']
        self.period = data_source["period"]
        self.pose_transfer = data_source["pose_transfer"]
        self.ray_bundles = [c.build('cpu') for c in self.cameras]

        logger.info('==> Find {} cameras'.format(len(self.cameras)))

        self.poses = {
            k: torch.tensor(np.asarray(v)).float()  # Nx4x4
            for k, v in data_source["poses"].items()
        }
        self.image_file_names = data_source['frames']

        # parallel loading frames
        self.frames = {}
        self.block_frame_id = []
        for k, cam_frames in data_source['frames'].items():
            with ThreadPoolExecutor(
                    max_workers=min(multiprocessing.cpu_count(), 32)
            ) as executor:
                frames = list(
                    tqdm(
                        executor.map(
                            lambda f:
                            torch.tensor(
                                data_io.imread(f['image_filename'])
                            ),
                            cam_frames,
                        ),
                        total=len(cam_frames),
                        dynamic_ncols=True,
                    )
                )

            def down2(img):
                sh = img.shape
                return torch.mean(torch.reshape(img, [sh[0] // 2, 2, sh[1] // 2, 2, -1]), (1, 3))
                # return torch.mean(torch.reshape(img, [sh[0], 1, sh[1] // 2, 2, -1]), (1, 3))

            n_down = down_level
            down_frames = []
            for img in frames:
                for j in range(n_down):
                    img = down2(img)
                down_frames.append(img)

            # self.frames[k] = torch.stack(frames, dim=0)
            self.frames[k] = torch.stack(down_frames, dim=0)

        self.frame_number = {k: x.shape[0] for k, x in self.frames.items()}
        self.aabb = torch.tensor(np.asarray(data_source['aabb'])).float()
        self.loss_multi = {
            k: torch.tensor([x['lossmult'] for x in v])
            for k, v in data_source['frames'].items()
        }

        self.file_names = {
            k: [x['image_filename'].stem for x in v]
            for k, v in data_source['frames'].items()
        }

        self.file_times = {
            k: [x['image_times'] for x in v]
            for k, v in data_source['frames'].items()
        }

        self.num_rays = num_rays
        self.image_H, self.image_W = self.cameras[0].height, self.cameras[0].width

        self.cam_intrins = self.cameras[0].K

        print("image width, height: ", self.image_W, self.image_H)
        # try to read a data to initialize RenderBuffer subclass
        self[0]

    def __len__(self):
        if self.training:
            return 10 ** 9  # hack of streaming dataset
        else:
            return sum([x.shape[0] for k, x in self.poses.items()])

    def update_num_rays(self, num_rays):
        self.num_rays = num_rays

    def update_frames(self, block_frame_id):
        self.block_frame_id = block_frame_id

    @torch.no_grad()
    def __getitem__(self, index):

        if self.training:
            rgb, c2w, cam_rays, loss_multi = [], [], [], []
            # 获取采样的ray
            for cam_idx in range(len(self.cameras)):
                num_rays = int(
                    self.num_rays
                    * (1.0 / self.loss_multi[cam_idx][0])
                    / sum([1.0 / v[0] for _, v in self.loss_multi.items()])
                )

                # 如果没有设定block_frame_id，那么就从全部图片范围进行采样，如果设定了那么就从给定的图片范围内进行采样
                if len(self.block_frame_id) == 0:
                    idx = torch.randint(
                        0,
                        self.frames[cam_idx].shape[0],
                        size=(num_rays,),
                    )
                else:
                    pos = torch.randint(
                        0,
                        len(self.block_frame_id),
                        size=(num_rays,),
                    )

                    idx = torch.tensor(self.block_frame_id)[pos]

                sample_x = torch.randint(
                    0,
                    self.cameras[cam_idx].width,
                    size=(num_rays,),
                )  # uniform sampling
                sample_y = torch.randint(
                    0,
                    self.cameras[cam_idx].height,
                    size=(num_rays,),
                )

                # uniform sampling
                rgb.append(self.frames[cam_idx][idx, sample_y, sample_x])
                cam_rays.append(self.ray_bundles[cam_idx][sample_y, sample_x])
                loss_multi.append(self.loss_multi[cam_idx][idx, None])

            rgb = torch.cat(rgb, dim=0)
            cam_rays = RayBundle.direct_cat(cam_rays, dim=0)
            loss_multi = torch.cat(loss_multi, dim=0)

        else:
            for cam_idx, num in self.frame_number.items():
                if index < num:
                    break
                index = index - num
            num_rays = len(self.ray_bundles[cam_idx])
            idx = torch.ones(size=(num_rays,), dtype=torch.int64) * index
            sample_x, sample_y = torch.meshgrid(
                torch.arange(self.cameras[cam_idx].width),
                torch.arange(self.cameras[cam_idx].height),
                indexing="xy",
            )
            sample_x = sample_x.reshape(-1)
            sample_y = sample_y.reshape(-1)

            rgb = self.frames[cam_idx][idx, sample_y, sample_x]

            cam_rays = self.ray_bundles[cam_idx][sample_y, sample_x]
            loss_multi = self.loss_multi[cam_idx][idx, None]

        target = RenderBuffer(
            rgb=rgb[..., :3],
            alpha=torch.ones_like(rgb[..., [-1]]),
            loss_multi=loss_multi)

        if not self.training:
            cam_rays = cam_rays.reshape(
                (self.cameras[cam_idx].height, self.cameras[cam_idx].width)
            )
            target = target.reshape(
                (self.cameras[cam_idx].height, self.cameras[cam_idx].width)
            )

        outputs = {
            # 'c2w': c2w,
            'cam_rays': cam_rays,
            'target': target,
            'idx': idx,
            # 'row_idx': sample_y,
            'row_idx': sample_y,
            # 'key_point': random.sample(self.key_point_sets, 50)
        }
        if not self.training:
            outputs['name'] = self.file_names[cam_idx][index]
        return outputs

    def compute_match_points(self, downlevel):

        cache_path = os.path.join(self.base_path, f'multi_view_matches_{downlevel}.pkl')

        if os.path.exists(cache_path):
            multi_view_matches = pickle.load(open(cache_path, 'rb'))
            return multi_view_matches

        extractor = SuperPoint(max_num_keypoints=2048).eval().cuda()
        matcher = LightGlue(features='superpoint').eval().cuda()

        cam_frames = self.image_file_names[0]

        multi_view_matches = {}

        for i in tqdm(range(len(cam_frames) - 1), desc="construct scene graph"):
            per_view_matches_keys = []
            per_view_matches_values = []
            for j in range(len(cam_frames) - 1):
                if i == j: continue

                image0 = load_image(cam_frames[i]['image_filename']).cuda()
                feats0 = extractor.extract(image0)

                image1 = load_image(cam_frames[j]['image_filename']).cuda()
                feats1 = extractor.extract(image1)

                matches01 = matcher({'image0': feats0, 'image1': feats1})
                feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]
                matches = self.extract_matches(matches01['matches'], matches01["scores"])

                points0 = feats0['keypoints'][matches[..., 0]]
                points1 = feats1['keypoints'][matches[..., 1]]

                one_col = torch.ones(points0.shape[0], 1).cuda()

                points0 = torch.cat((points0, one_col), dim=1).cpu()
                points1 = torch.cat((points1, one_col), dim=1).cpu()

                diff_uv = points0 - points1
                diff_uv = torch.norm(diff_uv, dim=1)
                diff_uv = torch.sum(diff_uv) / diff_uv.shape[0]

                per_view_matches_keys.append(diff_uv.float())
                per_view_matches_values.append(edict({"idx": j, "pt_i": points0, "pt_j": points1}))

            sorted_id = sorted(range(len(per_view_matches_keys)), key=lambda k: per_view_matches_keys[k])
            sorted_per_view_mathes_values = [per_view_matches_values[i] for i in sorted_id]
            multi_view_matches[i] = sorted_per_view_mathes_values

        pickle.dump(multi_view_matches, open(cache_path, 'wb'))
        return multi_view_matches

    def extract_matches(self, matches, scores, threshold=0.99):
        selected_matches = []
        selected_scores = []

        for match, score in zip(matches, scores):
            if score >= threshold:
                selected_matches.append(match)
                selected_scores.append(score)

        return torch.stack(selected_matches)


def ray_collate(batch):
    res = {k: [] for k in batch[0].keys()}
    for data in batch:
        for k, v in data.items():
            res[k].append(v)
    for k, v in res.items():
        if isinstance(v[0], RenderBuffer) or isinstance(v[0], RayBundle):
            res[k] = TensorDataclass.direct_cat(v, dim=0)
        else:
            res[k] = torch.cat(v, dim=0)
    return res


def extract_matches(matches, scores, threshold=0.99):
    selected_matches = []
    selected_scores = []

    for match, score in zip(matches, scores):
        if score >= threshold:
            selected_matches.append(match)
            selected_scores.append(score)

    return torch.stack(selected_matches)


def generate_combinations(n):
    numbers = list(range(n))
    result = list(combinations(numbers, 2))
    return result


if __name__ == '__main__':
    training_dataset = RayDataset(
        # '/mnt/bn/wbhu-nerf/Dataset/nerf_synthetic',
        'E:/DeepLearn/L2G-NeRF-main/data/whu',
        'rsvi_t1_fast',
        # 'nerf_synthetic',
        'whu',
    )
    train_loader = iter(
        DataLoader(
            training_dataset,
            batch_size=8,
            shuffle=False,
            num_workers=0,
            collate_fn=ray_collate,
            pin_memory=True,
            worker_init_fn=None,
            pin_memory_device='cuda',
        )
    )
    for i in tqdm(range(1000)):
        data = next(train_loader)
        pass
