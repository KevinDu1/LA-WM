import os
from glob import glob
from PIL import Image

import numpy as np
import pandas as pd
import scipy.ndimage
import torch
from torch.utils.data import Dataset, DataLoader
import random
import torchvision
from mile.constants import CARLA_FPS
from mile.data.dataset_utils import integer_to_binary, calculate_birdview_labels, pkl_preprocess
from mile.utils.geometry_utils import get_out_of_view_mask, calculate_geometry
from mile.data.NuscenesData import FuturePredictionDataset
from nuscenes.nuscenes import NuScenes
from config import get_cfg_djt
import torchvision.transforms.functional as tvf
from typing import List, Optional

# self.nusc = NuScenes(version='v1.0-trainval', dataroot='/media/ps/data/dujiatong/nusc', verbose=True)
# self.train_nusc = FuturePredictionDataset(nusc=self.nusc, is_train=0)
# class CombineDataModule(Dataset):
#     def __init__(self, cfg):
#         super().__init__()
#         self.cfg = cfg
#         self.batch_size = self.cfg.BATCHSIZE
#         self.sequence_length = self.cfg.RECEPTIVE_FIELD + self.cfg.FUTURE_HORIZON
#         self.nusc = NuScenes(version='v1.0-trainval', dataroot='/media/ps/data/dujiatong/nusc', verbose=True) #v1.0-trainval

#         # Will be populated with self.setup()
#         self.train_carla, self.val_carla = None, None
#         self.train_nusc, self.val_nusc = None, None

#     def setup(self, stage=None):
#         self.train_carla = CarlaDataset(self.cfg, mode='train', sequence_length=self.sequence_length)
#         # self.val_carla = CarlaDataset(self.cfg, mode='val', sequence_length=self.sequence_length)

#         self.train_nusc = FuturePredictionDataset(nusc=self.nusc, is_train=0, cfg=self.cfg)
#         # self.val_nusc = FuturePredictionDataset(nusc=self.nusc, is_train=1, cfg=self.cfg)

#         print(f'{len(self.train_carla)} data points in {self.train_carla.dataset_path}')
#         print(f'{len(self.train_nusc)} data points in {self.train_nusc.nusc.dataroot}')

#         self.train_sampler = None
#         self.val_sampler = None

#         self.train_carla.shuffle_data()
#         self.train_nusc.shuffle_data()

#     def __len__(self):
#         self.total_len = max(len(self.train_carla), len(self.train_nusc))
#         # self.total_len = len(self.train_carla)
#         return self.total_len

#     def __getitem__(self, idx):
#         #假设nuscence少
#         if idx < len(self.train_nusc):
#             return {'data_carla': self.train_carla[idx], 'data_nusc': self.train_nusc[idx]}
#         else:
#             return {'data_carla': self.train_carla[idx], 'data_nusc': self.train_nusc[idx % len(self.train_nusc)]}
#         # return self.train_carla[idx]


#     def train_dataloader(self):
#         return DataLoader(
#             self.train_dataset,
#             batch_size=self.batch_size,
#             shuffle=self.train_sampler is None,
#             num_workers=self.cfg.N_WORKERS,
#             pin_memory=True,
#             drop_last=True,
#             sampler=self.train_sampler,
#         )

#     def val_dataloader(self):
#         return DataLoader(
#             self.val_dataset,
#             batch_size=self.batch_size,
#             shuffle=False,
#             num_workers=self.cfg.N_WORKERS,
#             pin_memory=True,
#             drop_last=True,
#             sampler=self.val_sampler,
#         )


class CarlaDataset(Dataset):
    def __init__(self, sequence_length=1):
        self.cfg = get_cfg_djt()
        self.sequence_length = sequence_length

        self.dataset_path = '/media/ps/data/dujiatong/2025carla-real-0315/Town05'
        # self.dataset_path = '/media/ps/data/dujiatong/carla-nusc-0227/Town05'
        self.intrinsics, self.extrinsics = calculate_geometry_from_config(self.cfg)
        self.bev_out_of_view_mask = get_out_of_view_mask(self.cfg)

        self.img_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                            torchvision.transforms.Normalize((0.5, 0.5, 0.5),
                                                                                             (0.5, 0.5, 0.5))])

        # Iterate over all runs in the data folder

        self.data = dict()

        runs = sorted(glob(os.path.join(self.dataset_path, '*')))
        # print(runs)
        for run_path in runs:
            run = os.path.basename(run_path)
            pd_dataframe_path = os.path.join(run_path, 'pd_dataframe.pkl')

            if os.path.isfile(pd_dataframe_path):
                self.data[f'{run}'] = pd.read_pickle(pd_dataframe_path).transpose() #因为过来的数据倒过来了，加个转置
                self.data[f'{run}'] = pkl_preprocess(self.data[f'{run}'], self.cfg.FUTURE_HORIZON)

        self.data_pointers = self.get_data_pointers()

    def get_data_pointers(self):
        data_pointers = []

        n_filtered_run = 0
        for run, data_run in self.data.items():
            # Calculate normalised reward of the run
            # run_length = len(data_run['speed'])
            # cumulative_reward = data_run['reward'].sum()
            # normalised_reward = cumulative_reward / run_length
            # if normalised_reward < self.cfg.DATASET.FILTER_NORM_REWARD:
            #     n_filtered_run += 1
            #     continue

            stride = int(self.cfg.DATASET.STRIDE_SEC * CARLA_FPS)  #stride=2
            # Loop across all elements in the dataset, and make all elements in a sequence belong to the same run
            start_index = int(CARLA_FPS * self.cfg.DATASET.FILTER_BEGINNING_OF_RUN_SEC) #10
            total_length = len(data_run) - stride * self.sequence_length
            for i in range(start_index, total_length):
                frame_indices = range(i, i + stride * self.sequence_length, stride)
                data_pointers.append((run, list(frame_indices)))

        print(f'Filtered {n_filtered_run} runs in {self.dataset_path}')

        if self.cfg.EVAL.DATASET_REDUCTION:
            import random
            random.seed(0)
            final_size = int(len(data_pointers) / self.cfg.EVAL.DATASET_REDUCTION_FACTOR)
            data_pointers = random.sample(data_pointers, final_size)

        return data_pointers

    def __len__(self):
        return len(self.data_pointers)

    def __getitem__(self, i):
        batch = {}

        run_id, indices = self.data_pointers[i]
        for t in indices:
            single_element_t = self.load_single_element_time_t(run_id, t)

            for k, v in single_element_t.items():
                batch[k] = batch.get(k, []) + [v]

        for k, v in batch.items():
            batch[k] = torch.from_numpy(np.stack(v))
            # print(f'the {k} shape of {i} is {batch[k].shape}')

        return batch
    
    def shuffle_data(self):
        random.shuffle(self.data_pointers)

    def load_single_element_time_t(self, run_id, t):
        data_row = self.data[run_id].iloc[t]
        single_element_t = {}

        # Load image
        image = Image.open(
            os.path.join(self.dataset_path, run_id, data_row['image_path'])
        )
        # image = self.img_transform(image).numpy()
        single_element_t['image'] = image

        # Load route map
        # route_map = Image.open(
        #     os.path.join(self.dataset_path, run_id, data_row['routemap_path'])
        # )
        # route_map = np.asarray(route_map)[None]
        # # Make the grayscale image an RGB image
        # _, h, w = route_map.shape
        # route_map = np.broadcast_to(route_map, (3, h, w)).copy()
        # single_element_t['route_map'] = route_map

        # Load bird's-eye view segmentation label
        birdview = np.asarray(Image.open(
            os.path.join(self.dataset_path, run_id, data_row['birdview_path'])
        ))
        h, w = birdview.shape
        n_classes = data_row['n_classes']
        birdview = integer_to_binary(birdview.reshape(-1), n_classes).reshape(h, w, n_classes)
        birdview = birdview.transpose((2, 0, 1))
        single_element_t['birdview'] = birdview
        birdview_label = calculate_birdview_labels(torch.from_numpy(birdview), n_classes).numpy()
        birdview_label = birdview_label[None]
        single_element_t['birdview_label'] = birdview_label

        # TODO: get person and car instance ids with json
        instance_mask = birdview[3].astype(np.bool) | birdview[4].astype(np.bool)
        instance_label, _ = scipy.ndimage.label(instance_mask[None].astype(np.int64))
        single_element_t['instance_label'] = instance_label

        # Load action and reward
        # throttle, steering, brake = data_row['action']
        # throttle_brake = throttle if throttle > 0 else -brake

        # single_element_t['steering'] = np.array([steering], dtype=np.float32)
        # single_element_t['throttle_brake'] = np.array([throttle_brake], dtype=np.float32)
        

        # single_element_t['reward'] = np.array([data_row['reward']], dtype=np.float32).clip(-1.0, 1.0)
        # single_element_t['value_function'] = np.array([data_row['value']], dtype=np.float32)

        # Geometry
        single_element_t['intrinsics'] = self.intrinsics.copy()
        single_element_t['extrinsics'] = self.extrinsics.copy()

        # single_element_t['speed'] = data_row['speed']
        # single_element_t['command'] = data_row['command']
        # if data_row['command'] == 0 or data_row['command'] == 3 or data_row['command'] == 4:
        #     single_element_t['command'] = 3
        # elif data_row['command'] == 1 or data_row['command'] == 5:
        #     single_element_t['command'] = 1
        # else:
        #     single_element_t['command'] = 2

        # single_element_t['future_location'] = data_row['future_location']
        # single_element_t['location_actor'] = data_row['location_actor']
        single_element_t = functional_crop(single_element_t, self.cfg.IMAGE.CROP)
        single_element_t['image'] = self.img_transform(single_element_t['image']).numpy()

        return single_element_t


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

def functional_crop(batch, crop):
    left, top, right, bottom = crop
    height = bottom - top
    width = right - left
    real = True
    if 'image' in batch:
        batch['image'] = tvf.crop(batch['image'], top, left, height, width)
    if 'depth' in batch:
        batch['depth'] = tvf.crop(batch['depth'], top, left, height, width)
    if 'semseg' in batch:
        batch['semseg'] = tvf.crop(batch['semseg'], top, left, height, width)
    if real:
        final_height = height // 2
        final_width = width // 2
        batch['image'] = tvf.resize(batch['image'], (final_height,final_width))
    if 'intrinsics' in batch:
        intrinsics = batch['intrinsics'].copy()
        intrinsics[..., 0, 2] -= left
        intrinsics[..., 1, 2] -= top
        batch['intrinsics'] = intrinsics
        if real:
            intrinsics = batch['intrinsics'].copy()
            intrinsics[..., 0, 2] = (intrinsics[..., 0, 2] - left) / 2  # 更新 cx
            intrinsics[..., 1, 2] = (intrinsics[..., 1, 2] - top) / 2   # 更新 cy
            intrinsics[..., 0, 0] /= 2  # fx 缩小一半
            intrinsics[..., 1, 1] /= 2  # fy 缩小一半
            batch['intrinsics'] = intrinsics

    return batch


class RealDataset(Dataset):
    def __init__(self):
        root_dir = '/media/ps/data/dujiatong/img_real/train'
        self.image_paths = []
        self.cfg = get_cfg_djt()
        if not os.path.isdir(root_dir):
            raise ValueError(f"目录 '{root_dir}' 不存在。")

        # 遍历 root_dir 下的每个子文件夹，收集图片路径
        for subdir in sorted(os.listdir(root_dir)):
            subdir_path = os.path.join(root_dir, subdir)
            if os.path.isdir(subdir_path):
                for fname in os.listdir(subdir_path):
                    if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                        self.image_paths.append(os.path.join(subdir_path, fname))

        if len(self.image_paths) == 0:
            raise ValueError("提供的目录中未找到任何图片文件。")
        
        self.img_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                            torchvision.transforms.Normalize((0.5, 0.5, 0.5),
                                                                                             (0.5, 0.5, 0.5))])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        batch = {}
        batch['image'] = Image.open(img_path).convert('RGB')
        
        # if self.transform:
        
        batch = functional_crop(batch, self.cfg.IMAGE.CROP)
        batch['image'] = self.img_transform(batch['image']).unsqueeze(0)
        
        return batch