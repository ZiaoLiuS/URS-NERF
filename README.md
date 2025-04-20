[comment]: <> (# URS-NeRF: Unordered Rolling Shutter Bundle Adjustment for Neural Radiance Fields)

<p align="center">
  <h1 align="center">URS-NeRF: Unordered Rolling Shutter Bundle Adjustment for Neural Radiance Fields</h1>
  <p align="center">
    <a href="https://boxulibrary.github.io/"><strong>Bo Xu</strong></a>
    ·
    <a href=""><strong>Ziao Liu</strong></a>
       ·
    <a href="https://dreamguo.github.io/"><strong>Mengqi Guo</strong></a>
       ·
    <a href=""><strong>Jiancheng Li</strong></a>
    ·
    <a href="https://www.comp.nus.edu.sg/~leegh/"><strong>Gim Hee Lee</strong></a>

[comment]: <> (  <h2 align="center">PAPER</h2>)
  <h3 align="center"><a href="https://arxiv.org/abs/2403.10119">Paper</a> | <a href="https://boxulibrary.github.io/projects/URS-NeRF/">Project Page</a></h3>
  <div align="center"></div>

## **Installation**
Please install the following dependencies first

- [PyTorch (1.13.1 + cu11.6)](https://pytorch.org/get-started/locally/) 
- [tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn)
- [nvdiffrast](https://nvlabs.github.io/nvdiffrast/)

And then install the following dependencies using *pip*

```shell
pip3 install av==9.2.0 \
    beautifulsoup4==4.11.1 \
    entrypoints==0.4 \
    gdown==4.5.1  \
    gin-config==0.5.0 \
    h5py==3.7.0 \
    imageio==2.21.1 \
    imageio-ffmpeg \
    ipython==7.19.0 \
    kornia==0.6.8 \
    loguru==0.6.0 \
    lpips==0.1.4 \
    mediapy==1.1.0 \
    mmcv==1.6.2 \
    ninja==1.10.2.3 \
    numpy==1.23.3 \
    open3d==0.16.0 \
    opencv-python==4.6.0.66 \
    pandas==1.5.0 \
    Pillow==9.2.0 \
    plotly==5.7.0 \
    pycurl==7.43.0.6 \
    PyMCubes==0.1.2 \
    pyransac3d==0.6.0 \
    PyYAML==6.0 \
    rich==12.6.0 \
    scipy==1.9.2 \
    tensorboard==2.9.0 \
    torch-fidelity==0.3.0 \
    torchmetrics==0.10.0 \
    torchtyping==0.1.4 \
    tqdm==4.64.1 \
    tyro==0.3.25 \
    appdirs \
    nerfacc==0.3.5 \
    plyfile \
    scikit-image \
    trimesh \
    torch_efficient_distloss \
    umsgpack \
    pyngrok \
    cryptography==39.0.2 \
    omegaconf==2.2.3 \
    segmentation-refinement \
    xatlas \
    protobuf==3.20.0 \
	jinja2 \
    click==8.1.7 \
    tensorboardx \
    termcolor
```

## **Dataset**

### WHU-RS dataset
```
Please download from https://pan.baidu.com/s/1pzbhxaJZjexwQ86_AkrH-A?pwd=mhuj 

code: mhuj
```
## **Training and evaluation**

### Run URS-NeRF：

```shell
python main.py --ginc config_files/TriMipRF_whu_URS.gin
```

### **Run the B-Spline Interpolation Method**:

```shell
python main.py --ginc config_files/TriMipRF_whu_spline.gin
```

# Citation
If you found this code/work to be useful in your own research, please considering citing the following:

```bibtex
@inproceedings{xu2024urs,
  title={URS-NeRF: Unordered Rolling Shutter Bundle Adjustment for Neural Radiance Fields},
  author={Xu, Bo and Liu, Ziao and Guo, Mengqi and Li, Jiancheng and Lee, Gim Hee},
  booktitle={European Conference on Computer Vision},
  pages={458--475},
  year={2024},
  organization={Springer}
}    
```
