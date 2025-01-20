import json
import gzip
import numpy as np
import cv2
from pathlib import Path
from typing import Union, Any
import open3d as o3d

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)


def load_from_json(file_path: Path):
    assert file_path.suffix == ".json"
    with open(file_path, encoding="UTF-8") as file:
        return json.load(file)


def load_from_jgz(file_path: Path):
    assert file_path.suffix == ".jgz"
    with gzip.GzipFile(file_path, "rb") as file:
        return json.load(file)


def write_to_json(file_path: Path, content: dict):
    assert file_path.suffix == ".json"
    with open(file_path, "w", encoding="UTF-8") as file:
        json.dump(content, file)


def imread(file_path: Path, dtype: np.dtype = np.float32) -> np.ndarray:
    im = cv2.imread(str(file_path), flags=cv2.IMREAD_UNCHANGED)
    if 2 == len(im.shape):
        im = im[..., None]
    if 4 == im.shape[-1]:
        im = cv2.cvtColor(im, cv2.COLOR_BGRA2RGBA)
    elif 3 == im.shape[-1]:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    elif 1 == im.shape[-1]:
        pass
    else:
        raise NotImplementedError
    if dtype != np.uint8:
        im = im / 255.0

    # im = cv2.GaussianBlur(im, (15, 15), 5.5)
    # im = im - im1
    return im.astype(dtype)

def high_pass_channel_filter(channel, kernel_size):
    # 定义拉普拉斯滤波器
    kernel = np.ones(kernel_size, np.float32) * -1
    center_value = kernel_size[0] * kernel_size[1] - 1
    kernel[int(kernel_size[0]/2), int(kernel_size[1]/2)] = center_value
    filtered_channel = cv2.filter2D(channel, cv2.CV_64F, kernel)

    # 将结果缩放到0-255范围
    filtered_channel = cv2.normalize(filtered_channel, None, 0, 255, cv2.NORM_MINMAX)

    # 转换为8位无符号整数
    filtered_channel = np.uint8(filtered_channel)

    return filtered_channel

def imwrite(im: np.ndarray, file_path: Path) -> None:
    if not file_path.parent.exists():
        file_path.parent.mkdir(parents=True, exist_ok=True)
    if len(im.shape) == 4:
        assert im.shape[0] == 1
        im = im[0]
    assert len(im.shape) == 3
    if im.dtype == np.float32:
        im = (im.clip(0.0, 1.0) * 255).astype(np.uint8)
    if 4 == im.shape[-1]:
        im = cv2.cvtColor(im, cv2.COLOR_RGBA2BGRA)
    elif 3 == im.shape[-1]:
        im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
    elif 1 == im.shape[-1]:
        im = im[..., 0]
    else:
        raise NotImplementedError
    cv2.imwrite(str(file_path), im)


def write_rendering(data: Any, parrent_path: Path, name: str):
    if isinstance(data, np.ndarray):
        imwrite(data, parrent_path / (name + '.png'))
    elif isinstance(data, o3d.geometry.PointCloud):
        if not parrent_path.exists():
            parrent_path.mkdir(exist_ok=True, parents=True)
        o3d.io.write_point_cloud(str(parrent_path / (name + '.ply')), data)
