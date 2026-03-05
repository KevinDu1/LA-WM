"""
Copyright (C) 2017 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
from networks import AdaINGen, MsImageDis, VAEGen
from utils import weights_init, get_model_list, vgg_preprocess, load_vgg16, get_scheduler
from torch.autograd import Variable
import torch
import torch.nn as nn
import numpy as np
import os
from bev_networks import BEVGen
from mile.losses import SegmentationLoss, SpatialRegressionLoss, RegressionLoss
from mile.utils.network_utils import pack_sequence_dim, unpack_sequence_dim, remove_past
from mile.models.mile import Mile
from STP3.stp3.utils.geometry import cumulative_warp_features_reverse, cumulative_warp_features
from STP3.stp3.config import get_cfg
from STP3.stp3.planning_metrics import PlanningMetric_3, PlanningMetric_Cilrs, gen_dx_bx

import torch.nn.functional as F
from torchmetrics.image import StructuralSimilarityIndexMeasure
from mile.models.evaluate_iou import iou

class MILE_Trainer(nn.Module):
    def __init__(self, hyperparameters):
        super(MILE_Trainer, self).__init__()
        lr = hyperparameters['lr']
        # self.bev_dongzhu = hyperparameters['bev_dongzhu']
        self.cfg = hyperparameters
        self.cfg2 = get_cfg()
        self.spatial_extent = (self.cfg2.LIFT.X_BOUND[1], self.cfg2.LIFT.Y_BOUND[1])
        self.receptive_field = 6
        self.future_field = 6
        # Initiate the networks
        self.gen_a = VAEGen(hyperparameters['input_dim_a'], hyperparameters['gen'])  # auto-encoder for domain a
        self.gen_b = VAEGen(hyperparameters['input_dim_b'], hyperparameters['gen'])  # auto-encoder for domain b
        # self.dis_a = MsImageDis(hyperparameters['input_dim_a'], hyperparameters['dis'])  # discriminator for domain a
        # self.dis_b = MsImageDis(hyperparameters['input_dim_b'], hyperparameters['dis'])  # discriminator for domain b
        self.bevgen = BEVGen(hyperparameters)
        self.worldmodel = Mile()

        self.instancenorm = nn.InstanceNorm2d(512, affine=False)

        # Setup the optimizers
        beta1 = hyperparameters['beta1']
        beta2 = hyperparameters['beta2']
        # dis_params = list(self.dis_a.parameters()) + list(self.dis_b.parameters())
        gen_params = list(self.gen_b.parameters()) + list(self.bevgen.parameters()) + list(self.worldmodel.parameters())
        # self.dis_opt = torch.optim.Adam([p for p in dis_params if p.requires_grad],
        #                                 lr=lr, betas=(beta1, beta2), weight_decay=hyperparameters['weight_decay'])
        self.gen_opt = torch.optim.Adam([p for p in gen_params if p.requires_grad],
                                        lr=lr, betas=(beta1, beta2), weight_decay=hyperparameters['weight_decay'])
        # self.dis_scheduler = get_scheduler(self.dis_opt, hyperparameters)
        self.gen_scheduler = get_scheduler(self.gen_opt, hyperparameters)

        # Network weight initialization
        # self.apply(weights_init(hyperparameters['init']))
        # self.dis_a.apply(weights_init('gaussian'))
        # self.dis_b.apply(weights_init('gaussian'))

        # Load VGG model if needed
        # if 'vgg_w' in hyperparameters.keys() and hyperparameters['vgg_w'] > 0:
        #     self.vgg = load_vgg16('model_vgg/')
        #     self.vgg.eval()
        #     for param in self.vgg.parameters():
        #         param.requires_grad = False

        self.segmentation_loss = SegmentationLoss(
                use_top_k=hyperparameters['SEMANTIC.SEG_USE_TOP_K'],
                top_k_ratio=hyperparameters['SEMANTIC.SEG_TOP_K_RATIO'],
                use_weights=hyperparameters['SEMANTIC.SEG_USE_WEIGHTS'],
                )

        self.center_loss = SpatialRegressionLoss(norm=2)
        self.offset_loss = SpatialRegressionLoss(norm=1, ignore_index=hyperparameters['INSTANCE_SEG.IGNORE_INDEX'])
        self.predict_location_loss = RegressionLoss(norm=1)
        
        self.X_BOUND = [-50.0, 50.0, 0.2]  #  Forward
        self.Y_BOUND = [-30.0, 30.0, 0.2]  # Sides
        self.Z_BOUND = [-10.0, 10.0, 20.0]
        self.dx, self.bx, self.nx = gen_dx_bx(self.X_BOUND, self.Y_BOUND, self.Z_BOUND)

    def recon_criterion(self, input, target):
        return torch.mean(torch.abs(input - target))
    
    def dongzhu(self):
        for param in self.gen_b.parameters():
            param.requires_grad = False
            
        if self.bev_dongzhu:
            for param in self.bevgen.parameters():
                param.requires_grad = False

        # for param in self.dis_a.parameters():
        #     param.requires_grad = False

        # for param in self.dis_b.parameters():
        #     param.requires_grad = False

    def forward(self, x_a, x_b):
        self.eval()
        h_a, _ = self.gen_a.encode(x_a)
        h_b, _ = self.gen_b.encode(x_b)
        x_ba = self.gen_a.decode(h_b)
        x_ab = self.gen_b.decode(h_a)
        self.train()
        return x_ab, x_ba

    def __compute_kl(self, mu):
        # def _compute_kl(self, mu, sd):
        # mu_2 = torch.pow(mu, 2)
        # sd_2 = torch.pow(sd, 2)
        # encoding_loss = (mu_2 + sd_2 - torch.log(sd_2)).sum() / mu_2.size(0)
        # return encoding_loss
        mu_2 = torch.pow(mu, 2)
        encoding_loss = torch.mean(mu_2)
        return encoding_loss

    def gen_update(self, batch_a, batch_b, hyperparameters):
        self.gen_opt.zero_grad()
        # encode
        x_a = pack_sequence_dim(batch_a['image'])
        x_b = pack_sequence_dim(batch_b['image'])
        h_a, n_a = self.gen_a.encode(x_a)
        h_b, n_b = self.gen_b.encode(x_b)
        # decode (within domain)
        x_a_recon = self.gen_a.decode(h_a + n_a)
        x_b_recon = self.gen_b.decode(h_b + n_b)
        # decode (cross domain)
        x_ba = self.gen_a.decode(h_b + n_b)
        x_ab = self.gen_b.decode(h_a + n_a)
        # encode again
        h_b_recon, n_b_recon = self.gen_a.encode(x_ba)
        h_a_recon, n_a_recon = self.gen_b.encode(x_ab)
        # decode again (if needed)
        x_aba = self.gen_a.decode(h_a_recon + n_a_recon) if hyperparameters['recon_x_cyc_w'] > 0 else None
        x_bab = self.gen_b.decode(h_b_recon + n_b_recon) if hyperparameters['recon_x_cyc_w'] > 0 else None

        # decode bev
        bev_a = self.bevgen.forward_bev(batch_a, h_a + n_a)
        bev_cycle_a = self.bevgen.forward_bev(batch_a, h_a_recon + n_a_recon)

        # reconstruction loss
        self.loss_gen_recon_x_a = self.recon_criterion(x_a_recon, x_a)
        self.loss_gen_recon_x_b = self.recon_criterion(x_b_recon, x_b)
        self.loss_gen_recon_kl_a = self.__compute_kl(h_a)
        self.loss_gen_recon_kl_b = self.__compute_kl(h_b)
        self.loss_gen_cyc_x_a = self.recon_criterion(x_aba, x_a)
        self.loss_gen_cyc_x_b = self.recon_criterion(x_bab, x_b)
        self.loss_gen_recon_kl_cyc_aba = self.__compute_kl(h_a_recon)
        self.loss_gen_recon_kl_cyc_bab = self.__compute_kl(h_b_recon)
        #bev loss
        self.loss_gen_bev_a = self.cal_bev_loss(bev_a, batch_a, 'bev_carla') #1
        self.loss_gen_bev_cycle_a = self.cal_bev_loss(bev_cycle_a, batch_a, 'bev_cycle_carla')
        # GAN loss
        self.loss_gen_adv_a = self.dis_a.calc_gen_loss(x_ba)
        self.loss_gen_adv_b = self.dis_b.calc_gen_loss(x_ab)
        # domain-invariant perceptual loss
        self.loss_gen_vgg_a = self.compute_vgg_loss(self.vgg, x_ba, x_b) if hyperparameters['vgg_w'] > 0 else 0
        self.loss_gen_vgg_b = self.compute_vgg_loss(self.vgg, x_ab, x_a) if hyperparameters['vgg_w'] > 0 else 0
        # total loss
        self.loss_gen_total = hyperparameters['gan_w'] * self.loss_gen_adv_a + \
                              hyperparameters['gan_w'] * self.loss_gen_adv_b + \
                              hyperparameters['recon_x_w'] * self.loss_gen_recon_x_a + \
                              hyperparameters['recon_kl_w'] * self.loss_gen_recon_kl_a + \
                              hyperparameters['recon_x_w'] * self.loss_gen_recon_x_b + \
                              hyperparameters['recon_kl_w'] * self.loss_gen_recon_kl_b + \
                              hyperparameters['recon_x_cyc_w'] * self.loss_gen_cyc_x_a + \
                              hyperparameters['recon_kl_cyc_w'] * self.loss_gen_recon_kl_cyc_aba + \
                              hyperparameters['recon_x_cyc_w'] * self.loss_gen_cyc_x_b + \
                              hyperparameters['recon_kl_cyc_w'] * self.loss_gen_recon_kl_cyc_bab + \
                              hyperparameters['vgg_w'] * self.loss_gen_vgg_a + \
                              hyperparameters['vgg_w'] * self.loss_gen_vgg_b + \
                              self.loss_gen_bev_a + \
                              self.loss_gen_bev_cycle_a
        self.loss_gen_total.backward()
        self.gen_opt.step()

    def bev_update(self, batch_a, batch_b, hyperparameters):
        self.gen_opt.zero_grad()
        # encode
        x_a = pack_sequence_dim(batch_a['image'])
        x_b = pack_sequence_dim(batch_b['image'])
        h_a, n_a = self.gen_a.encode(x_a)
        h_b, n_b = self.gen_b.encode(x_b)
        # decode (cross domain)
        x_ab = self.gen_b.decode(h_a + n_a)
        # encode again
        h_a_recon, n_a_recon = self.gen_b.encode(x_ab)

        # carla decode bev
        bev_a = self.bevgen.forward_bev(batch_a, h_a + n_a)
        bev_cycle_a = self.bevgen.forward_bev(batch_a, h_a_recon + n_a_recon)

        # nusc latent bev
        emb = self.bevgen.encoder_bev(batch_b, h_b + n_b)
        traj_b = self.worldmodel.forward(batch_b, emb)

        #bev loss
        self.loss_gen_bev_a = self.cal_bev_loss(bev_a, batch_a, 'bev_carla') #1
        self.loss_gen_bev_cycle_a = self.cal_bev_loss(bev_cycle_a, batch_a, 'bev_cycle_carla')
        self.loss_gen_traj_b = self.predict_location_loss(traj_b, batch_b['gt_trajectory'])
        
        # total loss
        self.loss_gen_total = self.loss_gen_bev_a + self.loss_gen_bev_cycle_a + 10*self.loss_gen_traj_b
        self.loss_gen_total.backward()
        self.gen_opt.step()
        
    def traj_update(self, batch_b, hyperparameters, intr, extr):
        self.gen_opt.zero_grad()
        # encode
        # x_a = pack_sequence_dim(batch_a['image'])
        x_b = pack_sequence_dim(batch_b['image'])
        # h_a, n_a = self.gen_a.encode(x_a)
        h_b, n_b = self.gen_b.encode(x_b)
        # decode (cross domain)
        # x_ab = self.gen_b.decode(h_a + n_a)
        # encode again
        # h_a_recon, n_a_recon = self.gen_b.encode(x_ab)

        # carla decode bev
        emb = self.bevgen.forward_bev_b(batch_b, h_b + n_b, intr, extr)

        # nusc latent bev
        traj_b = self.worldmodel.forward(batch_b, emb)

        #bev loss
        # self.loss_gen_bev_a = self.cal_bev_loss(bev_a, batch_a, 'bev_carla') #1
        # self.loss_gen_bev_cycle_a = self.cal_bev_loss(bev_cycle_a, batch_a, 'bev_cycle_carla')
        self.loss_gen_traj_b = self.predict_location_loss(traj_b, batch_b['gt_trajectory'][:,:,:-1])
        
        # total loss
        # self.loss_gen_total = self.loss_gen_bev_a + self.loss_gen_bev_cycle_a + 10*self.loss_gen_traj_b
        self.loss_gen_traj_b.backward()
        self.gen_opt.step()

    def compute_vgg_loss(self, vgg, img, target):
        img_vgg = vgg_preprocess(img)
        target_vgg = vgg_preprocess(target)
        img_fea = vgg(img_vgg)
        target_fea = vgg(target_vgg)
        return torch.mean((self.instancenorm(img_fea) - self.instancenorm(target_fea)) ** 2)
    
    def cal_bev_loss(self, output, batch, zifu):
        loss_bev = dict()
        for downsampling_factor in [1, 2, 4]:
            bev_segmentation_loss = self.segmentation_loss(
                prediction=output[f'bev_segmentation_{downsampling_factor}'],
                target=batch[f'birdview_label_{downsampling_factor}'],
            )
            discount = 1/downsampling_factor
            loss_bev[f'loss_gen_{zifu}_bev_segmentation_{downsampling_factor}'] = discount * self.cfg['LOSSES.WEIGHT_SEGMENTATION'] * \
                                                                bev_segmentation_loss

            center_loss = self.center_loss(
                prediction=output[f'bev_instance_center_{downsampling_factor}'],
                target=batch[f'center_label_{downsampling_factor}']
            )
            offset_loss = self.offset_loss(
                prediction=output[f'bev_instance_offset_{downsampling_factor}'],
                target=batch[f'offset_label_{downsampling_factor}']
            )

            center_loss = self.cfg['INSTANCE_SEG.CENTER_LOSS_WEIGHT'] * center_loss
            offset_loss = self.cfg['INSTANCE_SEG.OFFSET_LOSS_WEIGHT'] * offset_loss

            loss_bev[f'loss_gen_{zifu}_bev_center_{downsampling_factor}'] = discount * self.cfg['LOSSES.WEIGHT_INSTANCE'] * center_loss
            # Offset are already discounted in the labels
            loss_bev[f'loss_gen_{zifu}_bev_offset_{downsampling_factor}'] = self.cfg['LOSSES.WEIGHT_INSTANCE'] * offset_loss
            total_loss = sum([x for x in loss_bev.values()])
        for key, value in loss_bev.items():
            setattr(self, key, value)
        return total_loss

    def sample(self, x_a, x_b):
        self.eval()
        x_a = pack_sequence_dim(x_a)
        x_b = pack_sequence_dim(x_b)
        x_a_recon, x_b_recon, x_ba, x_ab = [], [], [], []
        for i in range(x_a.size(0)):
            h_a, _ = self.gen_a.encode(x_a[i].unsqueeze(0))
            h_b, _ = self.gen_b.encode(x_b[i].unsqueeze(0))
            x_a_recon.append(self.gen_a.decode(h_a))
            x_b_recon.append(self.gen_b.decode(h_b))
            x_ba.append(self.gen_a.decode(h_b))
            x_ab.append(self.gen_b.decode(h_a))
        x_a_recon, x_b_recon = torch.cat(x_a_recon), torch.cat(x_b_recon)
        x_ba = torch.cat(x_ba)
        x_ab = torch.cat(x_ab)
        self.train()
        return x_a, x_a_recon, x_ab, x_b, x_b_recon, x_ba
    
    def sample_test_image(self, batch_a, batch_b):
        self.eval()
        #encode
        x_a = pack_sequence_dim(batch_a['image'])
        x_b = pack_sequence_dim(batch_b['image'])
        h_a, _ = self.gen_a.encode(x_a)
        h_b, _ = self.gen_b.encode(x_b)
        # decode (within domain)
        x_a_recon = self.gen_a.decode(h_a)
        x_b_recon = self.gen_b.decode(h_b)
        # decode (cross domain)
        x_ba = self.gen_a.decode(h_b)
        x_ab = self.gen_b.decode(h_a)
        # encode again
        h_b_recon, _ = self.gen_a.encode(x_ba)
        h_a_recon, _ = self.gen_b.encode(x_ab)
        # decode again (if needed)
        x_aba = self.gen_a.decode(h_a_recon)
        x_bab = self.gen_b.decode(h_b_recon)
        
        mse_a_recon = compute_mse(x_a_recon,x_a)
        ssim_a_recon = compute_ssim(x_a_recon,x_a)
        mse_b_recon = compute_mse(x_b_recon,x_b)
        ssim_b_recon = compute_ssim(x_b_recon,x_b)
        mse_a_cycle = compute_mse(x_aba,x_a)
        ssim_a_cycle = compute_ssim(x_aba,x_a)
        mse_b_cycle = compute_mse(x_bab,x_b)
        ssim_b_cycle = compute_ssim(x_bab,x_b)

        return [mse_a_recon, ssim_a_recon, mse_b_recon, ssim_b_recon, mse_a_cycle, ssim_a_cycle, mse_b_cycle, ssim_b_cycle]

    def sample_test_bev(self, batch_a, batch_b):
        self.eval()
        #encode
        x_a = pack_sequence_dim(batch_a['image'])
        # x_b = pack_sequence_dim(batch_b['image'])
        h_a, _ = self.gen_a.encode(x_a)
        # h_b, _ = self.gen_b.encode(x_b)
        # decode (within domain)
        # x_a_recon = self.gen_a.decode(h_a)
        # x_b_recon = self.gen_b.decode(h_b)
        # decode (cross domain)
        # x_ba = self.gen_a.decode(h_b)
        x_ab = self.gen_b.decode(h_a)
        # encode again
        # h_b_recon, _ = self.gen_a.encode(x_ba)
        h_a_recon, _ = self.gen_b.encode(x_ab)
        # decode again (if needed)
        # x_aba = self.gen_a.decode(h_a_recon)
        # x_bab = self.gen_b.decode(h_b_recon)
        
        # decode bev
        bev_a = self.bevgen.forward_bev(batch_a, h_a)
        bev_cycle_a = self.bevgen.forward_bev(batch_a, h_a_recon)
        
        bev_a_result = iou(batch_a['birdview_label_1'], torch.argmax(bev_a['bev_segmentation_1'].detach(), dim=-3))
        bev_cycle_a_result = iou(batch_a['birdview_label_1'],torch.argmax(bev_cycle_a['bev_segmentation_1'].detach(), dim=-3))
        
        return bev_a_result, bev_cycle_a_result
        
    def sample_traj(self, batch_b, hyperparameters, intr, extr):
        self.eval()
        # encode
        # x_a = pack_sequence_dim(batch_a['image'])
        x_b = pack_sequence_dim(batch_b['image'])
        # print(x_b.shape)
        # h_a, n_a = self.gen_a.encode(x_a)
        h_b, n_b = self.gen_b.encode(x_b)
        # decode (cross domain)
        # x_ab = self.gen_b.decode(h_a + n_a)
        # encode again
        # h_a_recon, n_a_recon = self.gen_b.encode(x_ab)

        # carla decode bev
        emb = self.bevgen.forward_bev_b(batch_b, h_b, intr, extr)

        # nusc latent bev
        traj_b = self.worldmodel.forward(batch_b, emb)

        self.loss_gen_traj_b = self.predict_location_loss(traj_b, batch_b['gt_trajectory'][:,:,:-1])
        print(self.loss_gen_traj_b)
        
        return traj_b
        
    def sample_traj_duibi(self, batch_b, hyperparameters, intr, extr):
        self.eval()
        # encode
        # x_a = pack_sequence_dim(batch_a['image'])
        x_b = pack_sequence_dim(batch_b['image'])
        # h_a, n_a = self.gen_a.encode(x_a)
        h_b, n_b = self.gen_a.encode(x_b)
        # decode (cross domain)
        # x_ab = self.gen_b.decode(h_a + n_a)
        # encode again
        # h_a_recon, n_a_recon = self.gen_b.encode(x_ab)

        # carla decode bev
        emb = self.bevgen.forward_bev_b(batch_b, h_b, intr, extr)

        # nusc latent bev
        traj_b = self.worldmodel.forward(batch_b, emb)

        self.loss_gen_traj_b = self.predict_location_loss(traj_b, batch_b['gt_trajectory'][:,:,:-1])
        print(self.loss_gen_traj_b)
        
        return traj_b

        
    def dis_update(self, batch_a, batch_b, hyperparameters):
        self.dis_opt.zero_grad()
        # encode
        x_a = pack_sequence_dim(batch_a['image'])
        x_b = pack_sequence_dim(batch_b['image'])
        h_a, n_a = self.gen_a.encode(x_a)
        h_b, n_b = self.gen_b.encode(x_b)
        # decode (cross domain)
        x_ba = self.gen_a.decode(h_b + n_b)
        x_ab = self.gen_b.decode(h_a + n_a)
        # D loss
        self.loss_dis_a = self.dis_a.calc_dis_loss(x_ba.detach(), x_a)
        self.loss_dis_b = self.dis_b.calc_dis_loss(x_ab.detach(), x_b)
        self.loss_dis_total = hyperparameters['gan_w'] * self.loss_dis_a + hyperparameters['gan_w'] * self.loss_dis_b
        self.loss_dis_total.backward()
        self.dis_opt.step()

    def update_learning_rate(self):
        # if self.dis_scheduler is not None:
        #     self.dis_scheduler.step()
        if self.gen_scheduler is not None:
            self.gen_scheduler.step()

    def resume(self, checkpoint_dir, hyperparameters):
        # Load generators
        last_model_name = get_model_list(checkpoint_dir, "gen")
        print(last_model_name)
        state_dict = torch.load(last_model_name)
        self.gen_a.load_state_dict(state_dict['a'])
        self.gen_b.load_state_dict(state_dict['b'])
        last_model_name = get_model_list(checkpoint_dir, "bev")
        print(last_model_name)
        state_dict = torch.load(last_model_name)
        self.bevgen.load_state_dict(state_dict['bev'])
        last_model_name = get_model_list(checkpoint_dir, "traj")
        print(last_model_name)
        state_dict = torch.load(last_model_name)
        self.worldmodel.load_state_dict(state_dict['traj'])
        # iterations = int(last_model_name[-11:-3])
        # Load discriminators
        # last_model_name = get_model_list(checkpoint_dir, "dis")
        # print(last_model_name)
        # state_dict = torch.load(last_model_name)
        # self.dis_a.load_state_dict(state_dict['a'])
        # self.dis_b.load_state_dict(state_dict['b'])
        # Load optimizers
        # state_dict = torch.load(os.path.join(checkpoint_dir, 'optimizer.pt'))
        # self.dis_opt.load_state_dict(state_dict['dis'])
        # self.gen_opt.load_state_dict(state_dict['gen'])
        # # Reinitilize schedulers
        # self.dis_scheduler = get_scheduler(self.dis_opt, hyperparameters, iterations)
        # self.gen_scheduler = get_scheduler(self.gen_opt, hyperparameters, iterations)
        # print('Resume from iteration %d' % iterations)
        iterations = 0
        return iterations

    def save(self, snapshot_dir, iterations):
        # Save generators, discriminators, and optimizers
        gen_name = os.path.join(snapshot_dir, 'gen_%08d.pt' % (iterations + 1))
        # dis_name = os.path.join(snapshot_dir, 'dis_%08d.pt' % (iterations + 1))
        bev_name = os.path.join(snapshot_dir, 'bev_%08d.pt' % (iterations + 1))
        traj_name = os.path.join(snapshot_dir, 'traj_%08d.pt' % (iterations + 1))
        opt_name = os.path.join(snapshot_dir, 'optimizer.pt')
        # torch.save({'b': self.gen_b.state_dict()}, gen_name)
        # torch.save({'a': self.dis_a.state_dict(), 'b': self.dis_b.state_dict()}, dis_name)
        if self.bev_dongzhu == False:
            torch.save({'bev': self.bevgen.state_dict()}, bev_name)
        torch.save({'traj': self.worldmodel.state_dict()}, traj_name)
        torch.save({'gen': self.gen_opt.state_dict()}, opt_name)
    
    def prepare_future_labels(self, batch):
        labels = {}

        segmentation_labels = batch['segmentation']
        # hdmap_labels = batch['hdmap']
        future_egomotion = batch['future_egomotion']
        gt_trajectory = batch['gt_trajectory']

        # present frame hd map gt
        # labels['hdmap'] = hdmap_labels[:, self.model.receptive_field - 1].long().contiguous()

        # gt trajectory
        labels['gt_trajectory'] = gt_trajectory

        # Warp labels to present's reference frame
        segmentation_labels_past = cumulative_warp_features(
            segmentation_labels[:, :self.receptive_field].float(),
            future_egomotion[:, :self.receptive_field],
            mode='nearest', spatial_extent=self.spatial_extent,
        ).long().contiguous()[:, :-1]
        segmentation_labels = cumulative_warp_features_reverse(
            segmentation_labels[:, (self.receptive_field - 1):].float(),
            future_egomotion[:, (self.receptive_field - 1):],
            mode='nearest', spatial_extent=self.spatial_extent,
        ).long().contiguous()
        labels['segmentation'] = torch.cat([segmentation_labels_past, segmentation_labels], dim=1)

        if self.cfg2.SEMANTIC_SEG.PEDESTRIAN.ENABLED:
            pedestrian_labels = batch['pedestrian']
            pedestrian_labels_past = cumulative_warp_features(
                pedestrian_labels[:, :self.receptive_field].float(),
                future_egomotion[:, :self.receptive_field],
                mode='nearest', spatial_extent=self.spatial_extent,
            ).long().contiguous()[:, :-1]
            pedestrian_labels = cumulative_warp_features_reverse(
                pedestrian_labels[:, (self.receptive_field - 1):].float(),
                future_egomotion[:, (self.receptive_field - 1):],
                mode='nearest', spatial_extent=self.spatial_extent,
            ).long().contiguous()
            labels['pedestrian'] = torch.cat([pedestrian_labels_past, pedestrian_labels], dim=1)

        # Warp instance labels to present's reference frame

        return labels
    
    def caculate_traj_loss(self, batch, labels, pre):
        metric_planning_val = []
        # real encoder结果记录
        metric_planning_val.append(PlanningMetric_Cilrs(self.future_field).to(pre.device))
        # sim encoder结果记录
        metric_planning_val.append(PlanningMetric_Cilrs(self.future_field).to(pre.device))
        occupancy = torch.logical_or(labels['segmentation'][:, self.receptive_field:].squeeze(2),
                                        labels['pedestrian'][:, self.receptive_field:].squeeze(2)).cpu()

        # visualize, opencv(H,W,C)的维度顺序和Tensor(C,H,W)的不太一样
        # occupancy(1,6,200,200)是未来时刻的
        # occupancy_visual = np.zeros([self.nx[0], self.nx[1], 3])
        # occupancy_cur = torch.logical_or(labels['segmentation'][:, self.receptive_field - 1].squeeze(2),
        #                                     labels['pedestrian'][:, self.receptive_field - 1].squeeze(2))
        # occupancy_ = occupancy_cur.cpu().numpy()
        # # occupancy_ = np.sum(occupancy_, axis=1)>0
        # occupancy_ = occupancy_[0, 0] > 0
        # occupancy_visual[:, :, 0] = occupancy_

        # occupancy_visual_future = np.zeros([self.nx[0], self.nx[1], 3])
        # occupancy_future = torch.logical_or(labels['segmentation'][:, -1].squeeze(2),
        #                                     labels['pedestrian'][:, -1].squeeze(2))
        # occupancy_future = occupancy_future.cpu().numpy()
        # # occupancy_ = np.sum(occupancy_, axis=1)>0
        # occupancy_future = occupancy_future[0, 0] > 0
        # occupancy_visual_future[:, :, 0] = occupancy_future
        # occupancy_visual[np.linspace(1,20,20,dtype=np.int),np.linspace(1,80,20,dtype=np.int),1] = 1
                    
        # img_ = batch['image'].cpu().numpy()
        # img_.resize([3, 450, 800])
        # img_ = img_.transpose(1, 2, 0)
        # img_ = ((img_[:, :, ::-1]*0.5+0.5)).astype(np.float64)
        # img_ = cv2.resize(img_, (800, 450))

        # img(B,3,N,256,256), target_points(B,4), command(B,6), speed(B,)
        #print(state[:,3:].argmax())

        #outputs_real = policy.forward_real(img, state)
        outputs_real = pre
        final_traj_real = outputs_real.detach().cpu()
        final_traj_theta_real = torch.zeros([final_traj_real.shape[0], final_traj_real.shape[1], final_traj_real.shape[2] + 1],
                                        device=final_traj_real.device)
        final_traj_xyz_real = torch.zeros([final_traj_real.shape[0], final_traj_real.shape[1], final_traj_real.shape[2] + 1],
                                        device=final_traj_real.device)

        #final_traj_sim = outputs_sim['pred_wp'].detach()
        #final_traj_theta_sim = torch.zeros([final_traj_sim.shape[0], final_traj_sim.shape[1], final_traj_sim.shape[2] + 1],
        #                               device=device)
        
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

        #final_traj_theta_sim[:,:,:2] = final_traj_sim[:,:,:2]
        #final_traj_theta_sim[:,:,2] = 0


        metric_planning_val[0](final_traj_theta_real[:, :self.future_field].to(final_traj_real.device),
                            labels['gt_trajectory'][:, :self.future_field].to(final_traj_real.device),
                            occupancy[:, :self.future_field].to(final_traj_real.device))
        #metric_planning_val[1](final_traj_theta_sim[:, :self.N_FUTURE_FRAMES].to(device),
        #                    labels['gt_trajectory'][:, :self.N_FUTURE_FRAMES].to(device),
        #                    occupancy[:, :self.N_FUTURE_FRAMES].to(device))

        #if index%1==0:
        #    traj_bev, target_bev, gt_traj_bev, point_project_img, gt_point_project_img = \
        #    self.plot_traj_on_bev(final_traj_xyz_real[0], state[0], labels['gt_trajectory'][0], calibrated)
        #    self.show_bev(occupancy_visual, self.center_point_visual, target_bev, traj_bev, gt_traj_bev, img_, point_project_img, gt_point_project_img, index,(0,0,255),self.fov)
        #traj_bev, target_bev, gt_traj_bev = self.plot_traj_on_bev(final_traj_sim[0], state[0],labels['gt_trajectory'][0])
        #self.show_bev(occupancy_visual, self.center_point_visual, target_bev, traj_bev, gt_traj_bev, img_,'sim', (0,100,200),self.fov)
        #cv2.waitKey(50)

        results = {}

        scores_real = metric_planning_val[0].compute()
        for i in range(self.future_field):
            for key, value in scores_real.items():
                results['plan_' + key + '_real' + '_{}s'.format((i + 1) * 0.5)] = value[i]

        #scores_sim = metric_planning_val[1].compute()
        #for i in range(self.N_FUTURE_FRAMES):
        #    for key, value in scores_sim.items():
        #        results['plan_' + key + '_sim' + '_{}s'.format((i + 1) * 0.5)] = value[i]

        # for key, value in results.items():
        #     print(f'{key} : {value.item()}')
        return results
        
def compute_mse(img1: torch.Tensor, img2: torch.Tensor) -> float:
    """
    计算两幅图像的均方误差（MSE）。
    
    参数:
    img1 (torch.Tensor): 第一张图像, 形状为 (C, H, W) 或 (N, C, H, W)
    img2 (torch.Tensor): 第二张图像, 形状需与 img1 相同
    
    返回:
    float: 计算得到的 MSE 值
    """
    assert img1.shape == img2.shape, "输入的两张图像形状必须相同"
    mse = F.mse_loss(img1, img2, reduction='mean')
    return mse.item()

def compute_ssim(img1: torch.Tensor, img2: torch.Tensor) -> float:
    """
    计算两幅图像的结构相似性（SSIM）。
    
    参数:
    img1 (torch.Tensor): 第一张图像, 形状为 (C, H, W) 或 (N, C, H, W)
    img2 (torch.Tensor): 第二张图像, 形状需与 img1 相同
    
    返回:
    float: 计算得到的 SSIM 值（范围为 [-1,1]）
    """
    assert img1.shape == img2.shape, "输入的两张图像形状必须相同"
    
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(img1.device)
    ssim_value = ssim_metric(img1.unsqueeze(0), img2.unsqueeze(0)) if img1.dim() == 3 else ssim_metric(img1, img2)
    
    return ssim_value.item()


