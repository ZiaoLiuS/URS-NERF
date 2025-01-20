import os
import argparse
from datetime import datetime
import gin
from loguru import logger
from torch.utils.data import DataLoader

from utils.common import set_random_seed
from dataset.ray_dataset import RayDataset, ray_collate
from neural_field.model import get_model
from trainer import Trainer
from utils.block_manager import blockManager
from utils.pose_filter import PoseFilter

def dataset_loader(train_dataset, test_dataset, batch_size, num_workers):
    logger.info("==> Init dataloader ...")
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        collate_fn=ray_collate,
        pin_memory=True,
        worker_init_fn=None,
        pin_memory_device='cuda',
        prefetch_factor=2,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=None,
        num_workers=1,
        shuffle=False,
        pin_memory=True,
        worker_init_fn=None,
        pin_memory_device='cuda',
    )

    return train_loader, test_loader


@gin.configurable()
def main(
        seed: int = 42,
        num_workers: int = 0,
        min_block_size: int = 5,
        train_split: str = "train",
        stages: str = "train_eval",
        batch_size: int = 16,
        model_name="Tri-MipRF",
):
    stages = list(stages.split("_"))
    set_random_seed(seed)
    downLevel = 0
    train_dataset = RayDataset(split=train_split, down_level=downLevel)
    test_dataset = RayDataset(split='test')
    logger.info("==> Init model ...")
    model = get_model(model_name=model_name)(aabb=train_dataset.aabb,
                                             train_frame_poses=train_dataset.poses,
                                             train_frame_times=train_dataset.file_times,
                                             eval_frame_poses=test_dataset.poses,
                                             eval_frame_times=test_dataset.file_times,
                                             read_out_time=train_dataset.read_out_time,
                                             image_H=train_dataset.image_H)

    # logger.info(model)
    logger.info("==> Init trainer ...downsample level {}".format(downLevel))
    trainer = Trainer(model, varied_eval_img=True)
    trainer.load_max_step_model()
    train_loader, test_loader = dataset_loader(train_dataset, test_dataset, batch_size, num_workers)
    # train 中加载train_loader和 test_loader
    trainer.update_train_loader(train_loader)
    trainer.update_eval_loader(test_loader)
    estimated_poses, estimated_velocities = trainer.fit(downLevel=downLevel)

    if "eval" in stages:
        if "train" not in stages:
            trainer.load_ckpt()
        trainer.eval(save_results=True, rendering_channels=["rgb", "depth"])



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ginc",
        action="append",
        help="gin config file",
    )
    parser.add_argument(
        "--ginb",
        action="append",
        help="gin bindings",
    )
    args = parser.parse_args()

    ginbs = []
    if args.ginb:
        ginbs.extend(args.ginb)

    gin.parse_config_files_and_bindings(args.ginc, ginbs, finalize_config=False)

    exp_name = gin.query_parameter("Trainer.exp_name")
    exp_name = (
        "%s/%s/%s"
        % (
            gin.query_parameter("RayDataset.scene_type"),
            gin.query_parameter("RayDataset.scene"),
            gin.query_parameter("main.model_name"),
            # datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        )
        if exp_name is None
        else exp_name
    )
    gin.bind_parameter("Trainer.exp_name", exp_name)
    gin.finalize()
    main()
