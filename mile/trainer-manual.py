import os

import pytorch_lightning as pl
import torch
import torch.nn.init as init
from torchmetrics import JaccardIndex
import math
from mile.config import get_cfg
from mile.constants import BIRDVIEW_COLOURS
from mile.losses import SegmentationLoss, KLLoss, RegressionLoss, SpatialRegressionLoss
from mile.models.mile import Mile
from mile.models.gen_networks import WorldGen
from mile.models.dis_networks import MsImageDis
from mile.models.preprocess import PreProcess
from torch.optim import lr_scheduler


class GANTrainer(pl.LightningModule):
    def __init__(self, hparams, path_to_conf_file=None, pretrained_path=None):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = get_cfg(cfg_dict=hparams)
        self.automatic_optimization = False
        if path_to_conf_file:
            self.cfg.merge_from_file(path_to_conf_file)
        if pretrained_path:
            self.cfg.PRETRAINED.PATH = pretrained_path
        # print(self.cfg)
        self.preprocess = PreProcess(self.cfg)
        
        # Model
        # self.model_first = Mile_GAN(self.cfg)
        self.gen_a = WorldGen(self.cfg)
        self.gen_b = WorldGen(self.cfg)
        self.dis_a = MsImageDis(self.cfg)
        self.dis_b = MsImageDis(self.cfg)

        self.dis_a.apply(weights_init('gaussian'))
        self.dis_b.apply(weights_init('gaussian'))

        self.rgb_loss = SpatialRegressionLoss(norm=1)
        self.probabilistic_loss = KLLoss(alpha=self.cfg.LOSSES.KL_BALANCING_ALPHA)

    def training_step(self, batch, batch_idx):
        batch = self.preprocess(batch)
        batch_a = batch['data_carla']
        batch_b = batch['data_nusc']

        #############train discriminator###################
        self.dis_opt.zero_grad()
        h_a, mu_a, sigma_a = self.gen_a.gen_encoder(batch_a, batch_a['image'])
        h_b, mu_b, sigma_b = self.gen_b.gen_encoder(batch_b, batch_b['image'])
        x_ba = self.gen_a.gen_decoder(h_b)
        x_ab = self.gen_b.gen_decoder(h_a)
        #loss_dis
        losses_dis = dict()
        losses_dis['dis_a_ba'] = self.cfg.DIS.GAN_WEIGHT * self.dis_a.calc_dis_loss(x_ba['rgb_label_1'].detach(), batch_a['image'])
        losses_dis['dis_b_ab'] = self.cfg.DIS.GAN_WEIGHT * self.dis_b.calc_dis_loss(x_ab['rgb_label_1'].detach(), batch_b['image'])
        self.logging_and_visualisation(losses_dis, batch_idx, prefix='train')
        dis_loss = self.loss_reducing(losses_dis)
        print(f'dis_loss is {dis_loss}')
        dis_loss.backward()
        self.dis_opt.step()
        self.dis_lr_scheduler.step()

        ################train generator####################
        self.gen_opt.zero_grad()
        h_a, mu_a, sigma_a = self.gen_a.gen_encoder(batch_a, batch_a['image'])
        h_b, mu_b, sigma_b = self.gen_b.gen_encoder(batch_b, batch_b['image'])
        #within domain
        x_a_recon = self.gen_a.gen_decoder(h_a)
        x_b_recon = self.gen_b.gen_decoder(h_b)
        #cross domain
        x_ba = self.gen_a.gen_decoder(h_b)
        x_ab = self.gen_b.gen_decoder(h_a)
        #encode again
        h_b_recon, mu_ba, sigma_ba, = self.gen_a.gen_encoder(batch_b, x_ba['rgb_label_1'])
        h_a_recon, mu_ab, sigma_ab, = self.gen_b.gen_encoder(batch_a, x_ab['rgb_label_1'])
        #decode again
        x_aba = self.gen_a.gen_decoder(h_a_recon)
        x_bab = self.gen_b.gen_decoder(h_b_recon)
        
        #loss
        losses_gen = dict()
        #重建loss
        losses_gen.update(self.cal_rgb_loss(x_a_recon, batch_a, 'x_a_recon'))
        losses_gen.update(self.cal_rgb_loss(x_b_recon, batch_b, 'x_b_recon'))
        #循环重建loss
        losses_gen.update(self.cal_rgb_loss(x_aba, batch_a, 'x_aba_recon'))
        losses_gen.update(self.cal_rgb_loss(x_bab, batch_b, 'x_bab_recon'))
        #分布的kl散度，同一张图片的不同风格迁移让他们分布相同
        h_recon_probabilistic_loss = self.probabilistic_loss(mu_b, sigma_b, mu_ba, sigma_ba)
        losses_gen['h_b_probabilistic'] = self.cfg.LOSSES.WEIGHT_PROBABILISTIC * h_recon_probabilistic_loss
        h_probabilistic_loss = self.probabilistic_loss(mu_a, sigma_a, mu_ab, sigma_ab)
        losses_gen['h_a_probabilistic'] = self.cfg.LOSSES.WEIGHT_PROBABILISTIC * h_probabilistic_loss
        # GAN loss
        losses_gen['loss_gen_adv_a'] = self.cfg.DIS.GAN_WEIGHT * self.dis_a.calc_gen_loss(x_ba['rgb_label_1'])
        losses_gen['loss_gen_adv_b'] = self.cfg.DIS.GAN_WEIGHT *self.dis_b.calc_gen_loss(x_ab['rgb_label_1'])
        
        self.logging_and_visualisation(losses_gen, batch_idx, prefix='train')
        gen_loss = self.loss_reducing(losses_gen)
        print(f'gen_loss is {gen_loss}')
        gen_loss.backward()
        self.gen_opt.step()
        self.gen_lr_scheduler.step()
        


    def cal_rgb_loss(self, x, batch, zifu):
        loss_rgb = dict()
        for downsampling_factor in [1, 2, 4]:
            rgb_weight = 0.1
            discount = 1 / downsampling_factor
            rgb_loss = self.rgb_loss(
                prediction=x[f'rgb_label_{downsampling_factor}'],
                target=batch[f'rgb_label_{downsampling_factor}'],
            )
            loss_rgb[f'{zifu}_rencon_rgb_{downsampling_factor}'] = rgb_weight * discount * rgb_loss
        return loss_rgb
    
    def loss_reducing(self, loss):
        total_loss = sum([x for x in loss.values()])
        return total_loss
    
    def logging_and_visualisation(self, loss, batch_idx, prefix='train'):
        # Logging
        self.log('-global_step', -self.global_step)
        for key, value in loss.items():
            self.log(f'{prefix}_{key}', value)

    def configure_optimizers(self):
        lr = self.cfg.OPTIMIZER.LR
        beta1 = 0.5
        beta2 = 0.999
        dis_params = list(self.dis_a.parameters()) + list(self.dis_b.parameters())
        gen_params = list(self.gen_a.parameters()) + list(self.gen_b.parameters())
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
                total_steps=self.cfg.STEPS,
                pct_start=self.cfg.SCHEDULER.PCT_START,
            )
            self.gen_lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.gen_opt,
                max_lr=self.cfg.OPTIMIZER.LR,
                total_steps=self.cfg.STEPS,
                pct_start=self.cfg.SCHEDULER.PCT_START,
            )
            # self.scheduler_d = {'scheduler': dis_lr_scheduler, 'interval': 'step'}
            # self.scheduler_g = {'scheduler': gen_lr_scheduler, 'interval': 'step'}
        #return [self.gen_opt, self.dis_opt], [scheduler_g, scheduler_d]
        return [self.dis_opt, self.gen_opt]
        


def weights_init(init_type='gaussian'):
    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') == 0 or classname.find('Linear') == 0) and hasattr(m, 'weight'):
            # print m.__class__.__name__
            if init_type == 'gaussian':
                init.normal_(m.weight.data, 0.0, 0.02)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'default':
                pass
            else:
                assert 0, "Unsupported initialization: {}".format(init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)

    return init_fun

