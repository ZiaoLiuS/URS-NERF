from typing import Callable

from . import nerf_synthetic, nerf_zju, nerf_carla, nerf_carla1, nerf_llff_unreal
from . import nerf_llff
from . import nerf_tum
from . import nerf_whu
from . import nerf_unreal
from . import nerf_synthetic_multiscale
from . import nerf_colmap_whu


def get_parser(parser_name: str) -> Callable:
    if 'nerf_synthetic' == parser_name:
        return nerf_synthetic.load_data
    elif 'nerf_synthetic_multiscale' == parser_name:
        return nerf_synthetic_multiscale.load_data
    elif 'nerf_llff' == parser_name:
        return nerf_llff.load_data
    elif 'tum' == parser_name:
        return nerf_tum.load_data
    elif 'whu' == parser_name:
        return nerf_whu.load_data
    elif 'zju' == parser_name:
        return nerf_zju.load_data
    elif 'carla' == parser_name:
        return nerf_carla.load_data
    elif 'carla1' == parser_name:
        return nerf_carla1.load_data
    elif 'unreal' == parser_name:
        return nerf_colmap_whu.load_data
    else:
        raise NotImplementedError
