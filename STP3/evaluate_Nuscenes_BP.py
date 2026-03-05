import copy

import torch
from argparse import ArgumentParser

from stp3.planning_metrics import PlanningMetric_3, PlanningMetric_Cilrs, gen_dx_bx
from stp3.metrics import IntersectionOverUnion, PanopticMetric, PlanningMetric
from stp3.datas.NuscenesData_cilrs import FuturePredictionDataset
from stp3.utils.network import preprocess_batch
from stp3.utils.geometry import cumulative_warp_features_reverse, cumulative_warp_features
from stp3.trainer import TrainingModule
from skimage.draw import polygon

from nuscenes.nuscenes import NuScenes
from tqdm import tqdm
import pickle

import logging
import os
from pathlib import Path
import subprocess
import numpy as np
from agents.cilrs.cilrs_agent_unit import CilrsAgent
from agents.cilrs.models.utils.dataset_traj import get_dataloader
from UNIT.myutils import get_all_data_loaders, get_all_data_loaders_cilrs
import torch as th
import cv2

log = logging.getLogger(__name__)

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

#_visual_dir = Path('Visual_Check')
#_visual_dir.mkdir(parents=True, exist_ok=True)

#device = 'cuda'


class Eval():
    def __init__(self):
        self.WIDTH = 1.85
        self.HEIGHT = 4.084

        self.X_BOUND = [-50.0, 50.0, 0.2]  #  Forward
        self.Y_BOUND = [-30.0, 30.0, 0.2]  # Sides
        self.Z_BOUND = [-10.0, 10.0, 20.0]
        self.dx, self.bx, self.nx = gen_dx_bx(self.X_BOUND, self.Y_BOUND, self.Z_BOUND)
        self.dx, self.bx = self.dx[:2].numpy(), self.bx[:2].numpy()

        self.pts = np.array([
            [-self.HEIGHT / 2. + 0.5, self.WIDTH / 2.],
            [self.HEIGHT / 2. + 0.5, self.WIDTH / 2.],
            [self.HEIGHT / 2. + 0.5, -self.WIDTH / 2.],
            [-self.HEIGHT / 2. + 0.5, -self.WIDTH / 2.],
        ])
        # 获取ego车辆在BEV的像素位置
        self.pts = (self.pts - self.bx) / (self.dx)
        # pts[:, [0, 1]] = pts[:, [1, 0]]
        # rc是ego车辆的BEV占据像素找到
        # rr, cc = polygon(pts[:, 0], pts[:, 1])
        # rc = np.concatenate([rr[:, None], cc[:, None]], axis=-1)

        self.center_point = np.array([np.mean(self.pts[:, 0]), np.mean(self.pts[:, 1])])
        self.center_point_visual = np.array([np.mean(self.pts[:, 0]), np.mean(self.pts[:, 1])], dtype=np.int)

        self.spatial_extent = (self.X_BOUND[1], self.Y_BOUND[1])

        # fov可视化
        # 以横向为扩展基准，其实fov_w_l画的是右侧的，因为BEV的坐标系是左为正，和opencv正好相反
        self.fov = 70
        self.pix_len_width = self.nx[1].cpu().numpy() / 2 * 0.8
        self.fov_w_l = (self.center_point_visual[1] - self.pix_len_width).astype(np.int)
        self.fov_w_r = (self.center_point_visual[1] + self.pix_len_width).astype(np.int)
        # fov_w = np.linspace(center_bev[1]-self.nx[1]/2*0.6,center_bev[1]+self.nx[1]/2*0.6,num=50,dtype=np.int)
        self.fov_h = (self.center_point_visual[0] + self.pix_len_width * np.tan(np.pi / 2 - self.fov / 2 / 180 * np.pi)).astype(np.int)
        


    def evaluate(self, checkpoint_path=None, dataroot=None):
        self.N_FUTURE_FRAMES = 4
        self.TIME_RECEPTIVE_FIELD = 1

        device = torch.device('cuda:0')
        # device = torch.device('cuda:0' if online else 'cpu')
        metric_planning_val = []
        # real encoder结果记录
        metric_planning_val.append(PlanningMetric_Cilrs(self.N_FUTURE_FRAMES).to(device))
        # sim encoder结果记录
        metric_planning_val.append(PlanningMetric_Cilrs(self.N_FUTURE_FRAMES).to(device))

        agent = CilrsAgent('/home/liye/carla-roach-unit/outputs/2024-07-13/17-36-34/config_agent.yaml')
        policy = agent._policy
        policy = policy.to(device)
        policy = policy.eval()

        trainer = TrainingModule.load_from_checkpoint(checkpoint_path, strict=True)
        trainer.eval()
        trainer.to(device)
        model = trainer.model
        cfg = model.cfg
        cfg.GPUS = "[1]"
        cfg.BATCHSIZE = 1
        cfg.LIFT.GT_DEPTH = False
        cfg.DATASET.DATAROOT = dataroot
        cfg.DATASET.MAP_FOLDER = dataroot
        dataroot = cfg.DATASET.DATAROOT
        nworkers = 8

        cfg.LIFT.X_BOUND = self.X_BOUND
        cfg.LIFT.Y_BOUND = self.Y_BOUND

        cfg.TIME_RECEPTIVE_FIELD = self.TIME_RECEPTIVE_FIELD  # 3
        cfg.N_FUTURE_FRAMES = self.N_FUTURE_FRAMES  # 6

        cfg.DATASET.VERSION = 'mini'
        nusc = NuScenes(version='v1.0-{}'.format(cfg.DATASET.VERSION), dataroot=dataroot, verbose=False)
        valdata = FuturePredictionDataset(nusc, 0, cfg)

        valloader = torch.utils.data.DataLoader(
            valdata, batch_size=cfg.BATCHSIZE, shuffle=False, num_workers=nworkers, pin_memory=True, drop_last=False
        )
        for index, batch in enumerate(tqdm(valloader)):
            preprocess_batch(batch, device)
            # image的维度(B,C,T,H,W)
            # state的维度(B,9)
            '''
            image = batch['image']
            intrinsics = batch['intrinsics']
            extrinsics = batch['extrinsics']
            future_egomotion = batch['future_egomotion']  # 朝向
            trajs = batch['sample_trajectory']
            speed = batch['speed']
            command = batch['command']
            target_points = batch['target_point']
            '''
            state = batch['state']
            img = batch['img']
            calibrated = batch['calibrated']
            labels = self.prepare_future_plan_labels(batch)
            occupancy = torch.logical_or(labels['segmentation'][:, self.TIME_RECEPTIVE_FIELD:].squeeze(2),
                                         labels['pedestrian'][:, self.TIME_RECEPTIVE_FIELD:].squeeze(2))

            # visualize, opencv(H,W,C)的维度顺序和Tensor(C,H,W)的不太一样
            # occupancy(1,6,200,200)是未来时刻的
            occupancy_visual = np.zeros([self.nx[0], self.nx[1], 3])
            occupancy_cur = torch.logical_or(labels['segmentation'][:, self.TIME_RECEPTIVE_FIELD - 1].squeeze(2),
                                             labels['pedestrian'][:, self.TIME_RECEPTIVE_FIELD - 1].squeeze(2))
            occupancy_ = occupancy_cur.cpu().numpy()
            # occupancy_ = np.sum(occupancy_, axis=1)>0
            occupancy_ = occupancy_[0, 0] > 0
            occupancy_visual[:, :, 0] = occupancy_

            occupancy_visual_future = np.zeros([self.nx[0], self.nx[1], 3])
            occupancy_future = torch.logical_or(labels['segmentation'][:, -1].squeeze(2),
                                             labels['pedestrian'][:, -1].squeeze(2))
            occupancy_future = occupancy_future.cpu().numpy()
            # occupancy_ = np.sum(occupancy_, axis=1)>0
            occupancy_future = occupancy_future[0, 0] > 0
            occupancy_visual_future[:, :, 0] = occupancy_future
            # occupancy_visual[np.linspace(1,20,20,dtype=np.int),np.linspace(1,80,20,dtype=np.int),1] = 1

            # img_ = img.cpu().numpy()
            # img_.resize([3, 450, 800])
            # img_ = img_.transpose(1, 2, 0)
            # img_ = ((img_[:, :, ::-1]*0.5+0.5)).astype(np.float64)
            # img_ = cv2.resize(img_, (800, 450))

            # img(B,3,N,256,256), target_points(B,4), command(B,6), speed(B,)
            #print(state[:,3:].argmax())
            # print(f'img.shape is {img.shape}, state.shape is {state.shape}')

            outputs_real = policy.forward_real_BP(img, state)
            # outputs_real = policy.forward_sim(img, state)
            final_traj_real = outputs_real['pred_wp'].detach()
            final_traj_theta_real = torch.zeros([final_traj_real.shape[0], final_traj_real.shape[1], final_traj_real.shape[2] + 1],
                                           device=device)
            final_traj_xyz_real = torch.zeros([final_traj_real.shape[0], final_traj_real.shape[1], final_traj_real.shape[2] + 1],
                                           device=device)

            # final_traj_sim = outputs_sim['pred_wp'].detach()
            # final_traj_theta_sim = torch.zeros([final_traj_sim.shape[0], final_traj_sim.shape[1], final_traj_sim.shape[2] + 1],
            #                                device=device)
            # 填充计算theta
            '''
            for B in range(final_traj.shape[0]):
                for P in range(final_traj.shape[1] - 1):
                    if P == 0:
                        final_traj_theta[B, P, :2] = final_traj[B, P]
                        vec = final_traj[B, P + 1] - torch.tensor([0, 0], device=device)
                        final_traj_theta[B, P, 2] = torch.atan2(vec[0], vec[1])
                    else:
                        final_traj_theta[B, P, :2] = final_traj[B, P]
                        vec = final_traj[B, P + 1] - final_traj[B, P - 1]
                        final_traj_theta[B, P, 2] = torch.atan2(vec[0], vec[1])
            '''
            final_traj_theta_real[:,:,:2] = final_traj_real[:,:,:2]
            final_traj_theta_real[:,:,2] = 0
            
            final_traj_xyz_real[:, :, :2] = final_traj_real[:, :, :2]
            final_traj_xyz_real[:, :, 2] = -1.6

            # final_traj_theta_sim[:,:,:2] = final_traj_sim[:,:,:2]
            # final_traj_theta_sim[:,:,2] = 0


            metric_planning_val[0](final_traj_theta_real[:, :self.N_FUTURE_FRAMES].to(device),
                                labels['gt_trajectory'][:, :self.N_FUTURE_FRAMES].to(device),
                                occupancy[:, :self.N_FUTURE_FRAMES].to(device))
            # metric_planning_val[1](final_traj_theta_sim[:, :self.N_FUTURE_FRAMES].to(device),
            #                     labels['gt_trajectory'][:, :self.N_FUTURE_FRAMES].to(device),
            #                     occupancy[:, :self.N_FUTURE_FRAMES].to(device))

            # traj_bev, target_bev, gt_traj_bev = self.plot_traj_on_bev(final_traj_real[0], state[0], labels['gt_trajectory'][0])
            # self.show_bev(occupancy_visual, self.center_point_visual, target_bev, traj_bev, gt_traj_bev, img_,'real',(0,0,255),self.fov)

            #traj_bev, target_bev, gt_traj_bev = self.plot_traj_on_bev(final_traj_sim[0], state[0],labels['gt_trajectory'][0])
            #self.show_bev(occupancy_visual, self.center_point_visual, target_bev, traj_bev, gt_traj_bev, img_,'sim', (0,100,200),self.fov)
            #self.show_bev(occupancy_visual_future, self.center_point_visual, target_bev, traj_bev, gt_traj_bev, img_,'future')
            cv2.waitKey(50)

        results = {}

        scores_real = metric_planning_val[0].compute()
        for i in range(self.N_FUTURE_FRAMES):
            for key, value in scores_real.items():
                results['plan_' + key + '_real' + '_{}s'.format((i + 1) * 0.5)] = value[i]

        # scores_sim = metric_planning_val[1].compute()
        # for i in range(self.N_FUTURE_FRAMES):
        #     for key, value in scores_sim.items():
        #         results['plan_' + key + '_sim' + '_{}s'.format((i + 1) * 0.5)] = value[i]

        for key, value in results.items():
            print(f'{key} : {value.item()}')


    def prepare_future_plan_labels(self, batch):
        labels = {}

        segmentation_labels = batch['segmentation']
        future_egomotion = batch['future_egomotion']
        gt_trajectory = batch['gt_trajectory']

        # gt trajectory
        labels['gt_trajectory'] = gt_trajectory

        # Warp labels to present's reference frame
        segmentation_labels_past = cumulative_warp_features(
            segmentation_labels[:, :self.TIME_RECEPTIVE_FIELD].float(),
            future_egomotion[:, :self.TIME_RECEPTIVE_FIELD],
            mode='nearest', spatial_extent=self.spatial_extent,
        ).long().contiguous()[:, :-1]
        segmentation_labels = cumulative_warp_features_reverse(
            segmentation_labels[:, (self.TIME_RECEPTIVE_FIELD - 1):].float(),
            future_egomotion[:, (self.TIME_RECEPTIVE_FIELD - 1):],
            mode='nearest', spatial_extent=self.spatial_extent,
        ).long().contiguous()
        labels['segmentation'] = torch.cat([segmentation_labels_past, segmentation_labels], dim=1)

        pedestrian_labels = batch['pedestrian']
        pedestrian_labels_past = cumulative_warp_features(
            pedestrian_labels[:, :self.TIME_RECEPTIVE_FIELD].float(),
            future_egomotion[:, :self.TIME_RECEPTIVE_FIELD],
            mode='nearest', spatial_extent=self.spatial_extent,
        ).long().contiguous()[:, :-1]
        pedestrian_labels = cumulative_warp_features_reverse(
            pedestrian_labels[:, (self.TIME_RECEPTIVE_FIELD - 1):].float(),
            future_egomotion[:, (self.TIME_RECEPTIVE_FIELD - 1):],
            mode='nearest', spatial_extent=self.spatial_extent,
        ).long().contiguous()
        labels['pedestrian'] = torch.cat([pedestrian_labels_past, pedestrian_labels], dim=1)
        return labels


    def plot_traj_on_bev(self, traj, state, gt_traj):
        # 为简化程序，Dataset不单独记录target_gps，从state里获取
        target = state[1:3]
        # traj的坐标系是右侧为y正，BEV的坐标系是左侧为y正，故对横向位移乘(-1)
        traj = traj[:, :2] * torch.tensor([1, -1], device=traj.device)
        gt_traj = gt_traj[:, :2] * torch.tensor([1, -1], device=gt_traj.device)
        target = target[:2] * torch.tensor([1, -1], device=target.device)

        traj = traj.cpu()
        target = target.cpu()
        gt_traj = gt_traj.cpu()

        n_future, _ = traj.shape

        traj_bev = (traj / self.dx).numpy() + self.center_point
        gt_traj_bev = (gt_traj / self.dx).numpy() + self.center_point
        target_bev = (target / self.dx).numpy() + self.center_point
        traj_bev = traj_bev.astype(np.int32)
        gt_traj_bev = gt_traj_bev.astype(np.int32)
        target_bev = target_bev.astype(np.int32)

        return traj_bev, target_bev, gt_traj_bev

    def show_bev(self, occupancy_visual, center_bev, target_bev, traj_bev, gt_traj_bev, img_, name, color ,fov=None):
        # opencv画点线是(width,hight)顺序
        #occupancy_visual = copy.deepcopy(occupancy_visual)
        cv2.circle(occupancy_visual, tuple(center_bev[::-1]), 3, (0, 160, 0), -1)
        cv2.circle(occupancy_visual, tuple(target_bev[::-1]), 3, (0, 255, 255), -1)
        for wp in traj_bev:
            cv2.circle(occupancy_visual, tuple(wp[::-1]), 2, color, -1)
        for wp in gt_traj_bev:
            cv2.circle(occupancy_visual, tuple(wp[::-1]), 2, (160, 160, 0), -1)
        if fov is not None:
            cv2.line(occupancy_visual, tuple(self.center_point_visual[::-1]), (self.fov_w_l, self.fov_h), (0, 255, 255), 1)
            cv2.line(occupancy_visual, tuple(self.center_point_visual[::-1]), (self.fov_w_r, self.fov_h), (0, 255, 255), 1)

        # BEV的坐标系是y正为左方,x正为前方，和opencv的坐标系正好相反，故x,y均要相反
        occupancy_visual = occupancy_visual[::-1, ::-1, :]
        cv2.imshow('img', img_)
        cv2.imshow(f'bev_{name}', occupancy_visual * 255)


    def run(self):
        self.evaluate(checkpoint_path='STP3/ckpt/STP3_plan.ckpt', 
                      dataroot='/data/liye/Nuscenes/nuscenes')



if __name__ == '__main__':
    Eval().run()
