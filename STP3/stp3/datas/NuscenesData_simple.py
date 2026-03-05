import os
from PIL import Image

import numpy as np
import cv2
import torch
import torch.utils.data
import torch.nn.functional as F
import torchvision

from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes, NuScenesExplorer
from nuscenes.can_bus.can_bus_api import NuScenesCanBus
from nuscenes.utils.splits import create_splits_scenes
from nuscenes.utils.data_classes import Box
from nuscenes.eval.common.utils import quaternion_yaw
from STP3.stp3.utils.tools import ( gen_dx_bx, get_nusc_maps)

from STP3.stp3.utils.geometry import (
    resize_and_crop_image,
    update_intrinsics,
    calculate_birds_eye_view_parameters,
    convert_egopose_to_matrix_numpy,
    pose_vec2mat,
    mat2pose_vec,
    invert_matrix_egopose_numpy,
    get_global_pose
)
from STP3.stp3.utils.instance import convert_instance_mask_to_center_and_offset_label
import STP3.stp3.utils.sampler as trajectory_sampler

def locate_message(utimes, utime):
    i = np.searchsorted(utimes, utime)
    if i == len(utimes) or (i > 0 and utime - utimes[i-1] < utimes[i] - utime):
        i -= 1
    return i

class FuturePredictionDataset(torch.utils.data.Dataset):
    SAMPLE_INTERVAL = 0.5 #SECOND
    def __init__(self, nusc, is_train, cfg, im_augmenter=None):
        self.nusc = nusc
        self.dataroot = self.nusc.dataroot
        self.nusc_exp = NuScenesExplorer(nusc)
        self.nusc_can = NuScenesCanBus(dataroot=self.dataroot)
        self.is_train = is_train
        self._im_augmenter = im_augmenter
        self._batch_read_number = 0
        self.cfg = cfg

        if self.is_train == 0:
            self.mode = 'train'
        elif self.is_train == 1:
            self.mode = 'val'
        elif self.is_train == 2:
            self.mode = 'test'
        else:
            raise NotImplementedError

        self.sequence_length = self.cfg.TIME_RECEPTIVE_FIELD + self.cfg.N_FUTURE_FRAMES
        self.receptive_field = self.cfg.TIME_RECEPTIVE_FIELD

        self.speed_factor = 12.0

        # 所有含CAN_BUS的场景
        self.scenes = self.get_scenes()
        # self.scenes中的所有sample
        self.ixes = self.prepro()
        # 获取每组past+future的sample的index,同时保证一组past+future在同一个scene中
        self.indices, self.scene_index = self.get_indices()


        # Normalising input images for cilrs
        self.normalise_img = torchvision.transforms.Compose(
            [torchvision.transforms.Resize([450,800]),
             torchvision.transforms.RandomCrop((450, 800)),
             torchvision.transforms.ToTensor(),
             torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )
        #  torchvision.transforms.Resize([225, 400]),
        #  torchvision.transforms.RandomCrop((225, 400)),


    def get_scenes(self):
        # filter by scene split
        split = {'v1.0-trainval': {0: 'train', 1: 'val', 2: 'test'},
                 'v1.0-mini': {0: 'mini_train', 1: 'mini_val'},}[
            self.nusc.version
        ][self.is_train]

        # 去除没有CAN_BUS信息的scene
        blacklist = [419] + self.nusc_can.can_blacklist  # # scene-0419 does not have vehicle monitor data
        blacklist = ['scene-' + str(scene_no).zfill(4) for scene_no in blacklist]

        scenes = create_splits_scenes()[split][:]
        for scene_no in blacklist:
            if scene_no in scenes:
                scenes.remove(scene_no)

        return scenes

    def prepro(self):
        # 去除未包含在self.scenes的sample，self.get_scene()方法去掉了没有CAN_BUS信息的scene
        samples = [samp for samp in self.nusc.sample]

        # remove samples that aren't in this split
        samples = [samp for samp in samples if self.nusc.get('scene', samp['scene_token'])['name'] in self.scenes]

        # sort by scene, timestamp (only to make chronological viz easier)
        samples.sort(key=lambda x: (x['scene_token'], x['timestamp']))

        return samples

    def get_indices(self):
        indices = []
        scene_begin_index = [0]
        for index in range(len(self.ixes)):
            is_valid_data = True
            previous_rec = None
            current_indices = []
            for t in range(self.sequence_length):
                index_t = index + t
                # Going over the dataset size limit.
                if index_t >= len(self.ixes):
                    is_valid_data = False
                    break
                rec = self.ixes[index_t]
                # Check if scene is the same
                if (previous_rec is not None) and (rec['scene_token'] != previous_rec['scene_token']):
                    is_valid_data = False
                    if index_t not in scene_begin_index:
                        scene_begin_index.append(index_t)
                    break

                current_indices.append(index_t)
                previous_rec = rec

            if is_valid_data:
                indices.append(current_indices)

        return np.asarray(indices), np.array(scene_begin_index)


    def get_gt_trajectory(self, rec, ref_index):
        n_output = self.cfg.N_FUTURE_FRAMES
        # n_output = 4
        gt_trajectory = np.zeros((n_output, 2), np.float64)

        egopose_cur = get_global_pose(rec, self.nusc, inverse=True)

        for i in range(n_output):
            index = ref_index + i +1
            if index < len(self.ixes):
                rec_future = self.ixes[index]

                egopose_future = get_global_pose(rec_future, self.nusc, inverse=False)

                egopose_future = egopose_cur.dot(egopose_future)
                # theta = quaternion_yaw(Quaternion(matrix=egopose_future))

                origin = np.array(egopose_future[:3, 3])
                # origin为(y,x,z),y正为车辆右，x正为车辆正前
                #gt_trajectory[i, :] = [origin[0], origin[1], theta]
                gt_trajectory[i, :] = [origin[1], origin[0]]

        if gt_trajectory[-1][1] >= 2:
            command = 2
        elif gt_trajectory[-1][1] <= -2:
            command = 1
        else:
            command = 4

        # VOID = -1
        # LEFT = 1
        # RIGHT = 2
        # STRAIGHT = 3
        # LANEFOLLOW = 4
        # CHANGELANELEFT = 5
        # CHANGELANERIGHT = 6

        #assert command in [0, 1, 2, 3, 4, 5]
        command_onehot = list(np.zeros(6,dtype=np.int32))
        command_onehot[command-1]=1

        return gt_trajectory, command_onehot, torch.tensor([command-1], dtype=torch.int8)

    def get_speed(self, rec, sample_index, gt_traj=None, is_bus=True):
        '''
        in m/s
        '''
        scene_index = self.scene_index[sample_index>=self.scene_index][-1]
        scene_time = (sample_index-scene_index)*0.5
        if is_bus:
            # process CAN_BUS velocity information
            scene = self.nusc.get('scene', rec['scene_token'])
            veh_speed = self.nusc_can.get_messages(scene['name'], 'vehicle_monitor')
            veh_speed = np.array([(m['utime'], m['vehicle_speed']) for m in veh_speed])
            veh_speed[:, 1] *= 1 / 3.6
            veh_speed[:, 0] = (veh_speed[:, 0] - veh_speed[0, 0]) / 1e6

            if any(abs(veh_speed[:,0]-scene_time)<0.1):
                speed = veh_speed[abs(veh_speed[:,0]-scene_time)<0.1][0][-1]
            elif any(abs(veh_speed[:, 0] - scene_time + 0.5) < 0.1) and any(abs(veh_speed[:, 0] - scene_time - 0.5) < 0.1):
                speed_past = veh_speed[abs(veh_speed[:, 0] - scene_time + 0.5) < 0.1][0][-1]
                speed_future = veh_speed[abs(veh_speed[:, 0] - scene_time - 0.5) < 0.1][0][-1]
                speed = (speed_past+speed_future)/2
            else:
                speed = None

        else:
            # gt_traj实现
            speed = 0

        if speed is None:
            # gt_traj实现,保证有值
            speed = 0

        return speed/self.speed_factor

    def get_target_gps_ref(self, rec, sample_index):
        '''
        in vehicle ref coordinate  in meter
        每个scene的10s和20s的车辆位置作为target_gps节点
        '''
        # egopose里包含了位置和朝向信息
        egopose_cur = get_global_pose(rec, self.nusc, inverse=True)
        target_gps_ref = np.zeros((4), np.float64)

        scene_index = self.scene_index[sample_index >= self.scene_index][-1]
        if any(sample_index < self.scene_index):
            next_scene_index = self.scene_index[sample_index < self.scene_index][0]
            scene_last_index = next_scene_index - 1
        else:
            scene_last_index = len(self.ixes)-1

        scene_mid_index = (scene_last_index-scene_index)//2 + scene_index

        if sample_index < scene_mid_index:
            target_rec = self.ixes[scene_mid_index]
            egopose_target = get_global_pose(target_rec, self.nusc, inverse=False)

            egopose_target_dot = egopose_cur.dot(egopose_target)
            origin = np.array(egopose_target_dot[:3, 3])
            
            if np.sqrt(origin[1]**2+origin[0]**2)<10:
                target_rec = self.ixes[scene_last_index]
                egopose_target = get_global_pose(target_rec, self.nusc, inverse=False)

        else:
            target_rec = self.ixes[scene_last_index]
            egopose_target = get_global_pose(target_rec, self.nusc, inverse=False)

        egopose_target_dot = egopose_cur.dot(egopose_target)
        theta = quaternion_yaw(Quaternion(matrix=egopose_target_dot))

        origin = np.array(egopose_target_dot[:3, 3])
        # origin为(y,x,z),y正为车辆右，x正为车辆正前
        # gt_trajectory[i, :] = [origin[0], origin[1], theta]
        target_gps_ref[:] = [origin[1], origin[0], origin[2], theta]

        return target_gps_ref

    def get_img(self, rec):
        sensor_list = ['CAM_FRONT']
        img_list = []
        for sensor in sensor_list:
            cam_data = self.nusc.get('sample_data', rec['data'][sensor])
            calibrated = self.nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
            # Load image
            image_filename = os.path.join(self.dataroot, cam_data['filename'])
            img = Image.open(image_filename)
            # Resize, crop, Normalise image
            normalised_img = self.normalise_img(img)
            img_list.append(normalised_img)

        return torch.stack(img_list, dim=1), calibrated

    def get_cilrs_state(self, rec, sample_index, command=None):
        speed = self.get_speed(rec, sample_index, None)
        target_points = self.get_target_gps_ref(rec, sample_index)
        if command is None:
            command = [0, 0, 0, 1, 0, 0]

        state_list = []
        state_list.append(speed)
        state_list.append(target_points[0])
        state_list.append(target_points[1])
        state_list += command

        return torch.tensor(state_list, dtype=torch.float32)




    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        data = {'indices':[]}
        # Loop over all the frames in the sequence.
        for i, index_t in enumerate(self.indices[index]):
            rec = self.ixes[index_t]

            if i < self.receptive_field:
                img, data['calibrated'] = self.get_img(rec)
                if self._im_augmenter is not None:
                    # pass
                    img = self._im_augmenter(self._batch_read_number).augment_image(img)
                # img = torch.stack([img], dim=1)
                data['img'] = img

                # images, intrinsics, extrinsics, depths = self.get_input_data(rec)
                # data['intrinsics'].append(intrinsics)
                # data['extrinsics'].append(extrinsics)

                self._batch_read_number += 1

            data['indices'].append(index_t)

            if i == self.cfg.TIME_RECEPTIVE_FIELD-1:
                gt_trajectory, command_onehot, command = self.get_gt_trajectory(rec, index_t)
                data['gt_trajectory'] = torch.from_numpy(gt_trajectory).float()
                data['state'] = self.get_cilrs_state(rec, index_t, command_onehot)
                data['target_point'] = self.get_target_gps_ref(rec, index_t)
                data['command'] = command

        return data