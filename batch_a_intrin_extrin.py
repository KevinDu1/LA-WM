from mile.data.NuscenesData import FuturePredictionDataset
from nuscenes.nuscenes import NuScenes
from torch.utils.data import DataLoader
import torch
from PIL import Image
import numpy as np
from torchvision import transforms
from mile.constants import BIRDVIEW_COLOURS
from dataset_real_virtual import CarlaDataset
from config import get_cfg_djt
from mile.models.preprocess import PreProcess
from mile.bev_trans import trans_black_to_bev
from config import get_cfg_djt

def calculate_geometry(image_fov, height, width, forward, right, up, pitch, yaw, roll):
    """Intrinsics and extrinsics for a single camera.
    See https://github.com/bradyz/carla_utils_fork/blob/dynamic-scene/carla_utils/leaderboard/camera.py
    and https://github.com/bradyz/carla_utils_fork/blob/dynamic-scene/carla_utils/recording/sensors/camera.py
    """
    f = width / (2 * np.tan(image_fov * np.pi / 360.0))
    cx = width / 2
    cy = height / 2
    intrinsics = np.float32([[f, 0, cx], [0, f, cy], [0, 0, 1]])
    extrinsics = get_extrinsics(forward, right, up, pitch, yaw, roll)
    return intrinsics, extrinsics

def get_extrinsics(forward, right, up, pitch, yaw, roll):
    # After multiplying the image coordinates by in the inverse intrinsics,
    # the resulting coordinates are defined with the axes (right, down, forward)
    assert pitch == yaw == roll == 0.0

    # After multiplying by the extrinsics, we want the axis to be (forward, left, up), and centered in the
    # inertial center of the ego-vehicle.
    mat = np.float32([
        [0,  0,  1, forward],
        [-1, 0,  0, -right],
        [0,  -1, 0, up],
        [0,  0,  0, 1],
    ])

    return mat

def calculate_geometry_from_config(cfg):
    """ Intrinsics and extrinsics for a single camera.
    See https://github.com/bradyz/carla_utils_fork/blob/dynamic-scene/carla_utils/leaderboard/camera.py
    and https://github.com/bradyz/carla_utils_fork/blob/dynamic-scene/carla_utils/recording/sensors/camera.py
    """
    # Intrinsics
    fov = cfg.IMAGE.FOV
    h, w = cfg.IMAGE.SIZE

    # Extrinsics
    forward, right, up = cfg.IMAGE.CAMERA_POSITION
    pitch, yaw, roll = cfg.IMAGE.CAMERA_ROTATION

    return calculate_geometry(fov, h, w, forward, right, up, pitch, yaw, roll)
