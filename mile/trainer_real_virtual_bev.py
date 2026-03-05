import os

import pytorch_lightning as pl
import torch
import torch.nn.init as init
from torchmetrics import JaccardIndex
import math
from mile.config import get_cfg
from mile.constants import BIRDVIEW_COLOURS
from mile.losses import SegmentationLoss, KLLoss, RegressionLoss, SpatialRegressionLoss
from mile.models.mile_bev import Mile_UINT
from mile.models.gen_networks import WorldGen
from mile.models.dis_networks import MsImageDis
from mile.models.preprocess import PreProcess
from torch.optim import lr_scheduler
import time
import matplotlib.pyplot as plt


class GANTrainer(pl.LightningModule):
    def __init__(self, hparams, path_to_conf_file=None, pretrained_path=None):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = get_cfg(cfg_dict=hparams)
        self.save_dir = os.path.join(
        self.cfg.LOG_DIR, time.strftime('%d%B%Yat%H:%M:%S%Z') + '_validate_save'
    )
        # self.automatic_optimization = False
        if path_to_conf_file:
            self.cfg.merge_from_file(path_to_conf_file)
        if pretrained_path:
            self.cfg.PRETRAINED.PATH = pretrained_path
        # print(self.cfg)
        self.preprocess = PreProcess(self.cfg)
        
        # Model
        self.model = Mile_UINT(self.cfg)
        
        # self.PRETRAINED.RESUME = True
        # self.PRETRAINED.PATH = '/home/hello/Dujiatong/transfer-mile/first_round_65.ckpt'
        # self.model.load_pretrained_weights()
        self.load_pretrained_weights_val()

        self.rgb_loss = SpatialRegressionLoss(norm=1)
        self.probabilistic_loss = KLLoss(alpha=self.cfg.LOSSES.KL_BALANCING_ALPHA)

    def load_pretrained_weights_val(self):
        if self.cfg.PRETRAINED.RESUME:
            # print(111,self.PRETRAINED.PATH)
            #self.cfg.PRETRAINED.PATH =  '/home/hello/Dujiatong/mile-main-0414/model-resume.ckpt'
            if os.path.isfile(self.cfg.PRETRAINED.PATH):
                checkpoint = torch.load(self.cfg.PRETRAINED.PATH, map_location='cpu')['state_dict']
                # checkpoint = {key[6:]: value for key, value in checkpoint.items() if key[:5] == 'gen_a' or 'gen_b' or 'dis_a' or 'dis_b'}
                # for name in checkpoint.keys():
                #     print(name)

                self.model.load_state_dict(checkpoint, strict=False)

                print('all is ok')
            else:
                raise FileExistsError(self.cfg.PRETRAINED.PATH)
        else:
            print('do not need to reload')

    def training_step(self, batch, batch_idx, optimizer_idx):
        batch = self.preprocess(batch, 'real_virtual')

        #############train discriminator###################
        if optimizer_idx == 0:
            # print('discriminator')
            output, losses_dis = self.model.forward_dis(batch)
            self.logging_and_visualisation(batch, output, losses_dis, batch_idx, prefix='train')
            return self.loss_reducing(losses_dis)

        ################train generator####################
        elif optimizer_idx == 1:
            # print('generator')
            output, losses_gen = self.model.forward_gen(batch)
            self.logging_and_visualisation(batch, output, losses_gen, batch_idx, prefix='train')
            return self.loss_reducing(losses_gen)      
    
    def validation_step(self, batch, batch_idx):
        batch = self.preprocess(batch, 'real_virtual')
        self.model.forward_val(batch, batch_idx)
        #

    
    def loss_reducing(self, loss):
        total_loss = sum([x for x in loss.values()])
        return total_loss
    
    def logging_and_visualisation(self, batch, output, loss, batch_idx, prefix='train'):
        # Logging
        self.log('-global_step', -self.global_step)
        # print(f'self.global_step is {self.global_step}')
        for key, value in loss.items():
            self.log(f'{prefix}_{key}', value)

        # if prefix == 'train':
        #     visualisation_criteria = self.global_step % self.cfg.VAL_CHECK_INTERVAL == 0
        # else:
        #     visualisation_criteria = batch_idx == 0
        # if visualisation_criteria:
        #     self.visualise_rgb(batch, output, batch_idx, prefix=prefix)

    # def visualise(self, batch, output, batch_idx, prefix='train'):
    #     if not self.cfg.SEMANTIC_SEG.ENABLED:
    #         return

    #     target = batch['birdview_label'][:, :, 0]
    #     pred = torch.argmax(output['bev_segmentation_1'].detach(), dim=-3)

    #     colours = torch.tensor(BIRDVIEW_COLOURS, dtype=torch.uint8, device=pred.device)

    #     target = colours[target]
    #     pred = colours[pred]

    #     # Move channel to third position
    #     target = target.permute(0, 1, 4, 2, 3)
    #     pred = pred.permute(0, 1, 4, 2, 3)

    #     visualisation_video = torch.cat([target, pred], dim=-1).detach()

    #     # Rotate for visualisation
    #     visualisation_video = torch.rot90(visualisation_video, k=1, dims=[3, 4])

    #     name = f'{prefix}_outputs'
    #     if prefix == 'val':
    #         name = name + f'_{batch_idx}'
    #     self.logger.experiment.add_video(name, visualisation_video, global_step=self.global_step, fps=2)

    def visualise_rgb(self, batch, output, batch_idx, prefix='train'):
        #存储量太大了，暂时割弃了，还不能运行
        if not self.cfg.SEMANTIC_SEG.ENABLED:
            return

        target = batch['data_carla']['image'][:, :, 0]
        pred = torch.argmax(output['rgb_label_1'].detach(), dim=-3)

        visualisation_video = torch.cat([target, pred], dim=-1).detach()

        name = f'{prefix}_outputs'
        if prefix == 'val':
            name = name + f'_{batch_idx}'
        self.logger.experiment.add_video(name, visualisation_video, global_step=self.global_step, fps=2)

    def configure_optimizers(self):
        lr = self.cfg.OPTIMIZER.LR
        beta1 = 0.5
        beta2 = 0.999
        dis_params = list(self.model.dis_a.parameters()) + list(self.model.dis_b.parameters())
        gen_params = list(self.model.gen_a.parameters()) + list(self.model.gen_b.parameters()) + list(self.model.bevgen.parameters())
        self.dis_opt = torch.optim.Adam([p for p in dis_params if p.requires_grad],
                                        lr=lr, betas=(beta1, beta2), weight_decay=0.0001)
        self.gen_opt = torch.optim.Adam([p for p in gen_params if p.requires_grad],
                                        lr=lr, betas=(beta1, beta2), weight_decay=0.0001)
        # if self.cfg.SCHEDULER.NAME == 'none':
        #     lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda lr: 1)
        if self.cfg.SCHEDULER.NAME == 'OneCycleLR':
            self.dis_lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.dis_opt,
                max_lr=self.cfg.OPTIMIZER.LR,
                total_steps=self.cfg.STEPS//2,
                pct_start=self.cfg.SCHEDULER.PCT_START,
            )
            self.gen_lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.gen_opt,
                max_lr=self.cfg.OPTIMIZER.LR,
                total_steps=self.cfg.STEPS//2,
                pct_start=self.cfg.SCHEDULER.PCT_START,
            )
            self.scheduler_d = {'scheduler': self.dis_lr_scheduler, 'interval': 'step'}
            self.scheduler_g = {'scheduler': self.gen_lr_scheduler, 'interval': 'step'}
        return [self.dis_opt, self.gen_opt], [self.scheduler_d, self.scheduler_g]
        # return [self.dis_opt, self.gen_opt]
        




