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
        # self.automatic_optimization = False
        if path_to_conf_file:
            self.cfg.merge_from_file(path_to_conf_file)
        if pretrained_path:
            self.cfg.PRETRAINED.PATH = pretrained_path
        # print(self.cfg)
        self.preprocess = PreProcess(self.cfg)
        
        # Model
        # self.model_first = Mile_GAN(self.cfg)
        self.model = WorldGen(self.cfg)

        # Losses
        self.action_loss = RegressionLoss(norm=1)
        if self.cfg.MODEL.TRANSITION.ENABLED:
            self.probabilistic_loss = KLLoss(alpha=self.cfg.LOSSES.KL_BALANCING_ALPHA)

        if self.cfg.SEMANTIC_SEG.ENABLED:
            self.segmentation_loss = SegmentationLoss(
                use_top_k=self.cfg.SEMANTIC_SEG.USE_TOP_K,
                top_k_ratio=self.cfg.SEMANTIC_SEG.TOP_K_RATIO,
                use_weights=self.cfg.SEMANTIC_SEG.USE_WEIGHTS,
                )

            self.center_loss = SpatialRegressionLoss(norm=2)
            self.offset_loss = SpatialRegressionLoss(norm=1, ignore_index=self.cfg.INSTANCE_SEG.IGNORE_INDEX)

            self.metric_iou_val = JaccardIndex(
                task='multiclass', num_classes=self.cfg.SEMANTIC_SEG.N_CHANNELS, average='none',
            )

        if self.cfg.EVAL.RGB_SUPERVISION:
            self.rgb_loss = SpatialRegressionLoss(norm=1)
        
    def training_step(self, batch, batch_idx):
        losses, output = self.shared_step(batch)

        self.logging_and_visualisation(batch, output, losses, batch_idx, prefix='train')

        return self.loss_reducing(losses)
    
    def validation_step(self, batch, batch_idx):
        loss, output = self.shared_step(batch)

        self.logging_and_visualisation(batch, output, loss, batch_idx, prefix='val')

        return {'val_loss': self.loss_reducing(loss)}

    def shared_step(self, batch):
        output = self.forward(batch)

        losses = dict()

        # action_weight = self.cfg.LOSSES.WEIGHT_ACTION

        #trajectory loss
        # losses['throttle_brake'] = action_weight * self.action_loss(output['throttle_brake'],
        #                                                             batch['throttle_brake'])
        # losses['steering'] = 2*action_weight * self.action_loss(output['steering'], batch['steering'])

        # if self.cfg.MODEL.TRANSITION.ENABLED:
        #     probabilistic_loss = self.probabilistic_loss(output['prior'], output['posterior'])

        #     losses['probabilistic'] = self.cfg.LOSSES.WEIGHT_PROBABILISTIC * probabilistic_loss

        if self.cfg.SEMANTIC_SEG.ENABLED:
            for downsampling_factor in [1, 2, 4]:
                bev_segmentation_loss = self.segmentation_loss(
                    prediction=output[f'bev_segmentation_{downsampling_factor}'],
                    target=batch[f'birdview_label_{downsampling_factor}'],
                )
                discount = 1/downsampling_factor
                losses[f'bev_segmentation_{downsampling_factor}'] = discount * self.cfg.LOSSES.WEIGHT_SEGMENTATION * \
                                                                    bev_segmentation_loss

                center_loss = self.center_loss(
                    prediction=output[f'bev_instance_center_{downsampling_factor}'],
                    target=batch[f'center_label_{downsampling_factor}']
                )
                offset_loss = self.offset_loss(
                    prediction=output[f'bev_instance_offset_{downsampling_factor}'],
                    target=batch[f'offset_label_{downsampling_factor}']
                )

                center_loss = self.cfg.INSTANCE_SEG.CENTER_LOSS_WEIGHT * center_loss
                offset_loss = self.cfg.INSTANCE_SEG.OFFSET_LOSS_WEIGHT * offset_loss

                losses[f'bev_center_{downsampling_factor}'] = discount * self.cfg.LOSSES.WEIGHT_INSTANCE * center_loss
                # Offset are already discounted in the labels
                losses[f'bev_offset_{downsampling_factor}'] = self.cfg.LOSSES.WEIGHT_INSTANCE * offset_loss

        # if self.cfg.EVAL.RGB_SUPERVISION:
        #     for downsampling_factor in [1, 2, 4]:
        #         rgb_weight = 0.1
        #         discount = 1 / downsampling_factor
        #         rgb_loss = self.rgb_loss(
        #             prediction=output[f'rgb_{downsampling_factor}'],
        #             target=batch[f'rgb_label_{downsampling_factor}'],
        #         )
        #         losses[f'rgb_{downsampling_factor}'] = rgb_weight * discount * rgb_loss

        return losses, output
    
    def forward(self, batch, deployment=False):
        batch = self.preprocess(batch, 'virtual')
        output = self.model.forward_bev(batch, deployment=deployment)
        return output
    
    def configure_optimizers(self):
        #  Do not decay batch norm parameters and biases
        # https://discuss.pytorch.org/t/weight-decay-in-the-optimizers-is-a-bad-idea-especially-with-batchnorm/16994/2
        def add_weight_decay(model, weight_decay=0.01, skip_list=[]):
            no_decay = []
            decay = []
            for name, param in model.named_parameters():
                if not param.requires_grad:
                    continue
                if len(param.shape) == 1 or any(x in name for x in skip_list):
                    no_decay.append(param)
                else:
                    decay.append(param)
            return [
                {'params': no_decay, 'weight_decay': 0.},
                {'params': decay, 'weight_decay': weight_decay},
            ]

        parameters = add_weight_decay(
            self.model,
            self.cfg.OPTIMIZER.WEIGHT_DECAY,
            skip_list=['relative_position_bias_table'],
        )
        weight_decay = 0.
        optimizer = torch.optim.AdamW(parameters, lr=self.cfg.OPTIMIZER.LR, weight_decay=weight_decay)

        # scheduler
        if self.cfg.SCHEDULER.NAME == 'none':
            lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda lr: 1)
        elif self.cfg.SCHEDULER.NAME == 'OneCycleLR':
            lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.cfg.OPTIMIZER.LR,
                total_steps=self.cfg.STEPS,
                pct_start=self.cfg.SCHEDULER.PCT_START,
            )

        return [optimizer], [{'scheduler': lr_scheduler, 'interval': 'step'}]

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
        # self.log('loss', total_loss)
        return total_loss
    
    def logging_and_visualisation(self, batch, output, loss, batch_idx, prefix='train'):
        # Logging
        self.log('-global_step', -self.global_step)
        # print(f'self.global_step is {self.global_step}')
        for key, value in loss.items():
            self.log(f'{prefix}_{key}', value)

        if prefix == 'train':
            visualisation_criteria = self.global_step % self.cfg.VAL_CHECK_INTERVAL == 0
        else:
            visualisation_criteria = batch_idx == 0
        if visualisation_criteria:
            self.visualise(batch, output, batch_idx, prefix=prefix)

    def visualise(self, batch, output, batch_idx, prefix='train'):
        if not self.cfg.SEMANTIC_SEG.ENABLED:
            return

        target = batch['birdview_label'][:, :, 0]
        pred = torch.argmax(output['bev_segmentation_1'].detach(), dim=-3)

        colours = torch.tensor(BIRDVIEW_COLOURS, dtype=torch.uint8, device=pred.device)

        target = colours[target]
        pred = colours[pred]

        # Move channel to third position
        target = target.permute(0, 1, 4, 2, 3)
        pred = pred.permute(0, 1, 4, 2, 3)

        visualisation_video = torch.cat([target, pred], dim=-1).detach()

        # Rotate for visualisation
        visualisation_video = torch.rot90(visualisation_video, k=1, dims=[3, 4])

        name = f'{prefix}_outputs'
        if prefix == 'val':
            name = name + f'_{batch_idx}'
        self.logger.experiment.add_video(name, visualisation_video, global_step=self.global_step, fps=2)
        


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

