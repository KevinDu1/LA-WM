import torch
import torch.nn as nn
import timm
import time
import os
import math
import numpy as np

from mile.constants import CARLA_FPS, DISPLAY_SEGMENTATION
from mile.utils.network_utils import pack_sequence_dim, unpack_sequence_dim, remove_past
from mile.models.common import BevDecoder, Decoder, RouteEncode, Policy
from mile.models.frustum_pooling import FrustumPooling
from mile.layers.layers import BasicBlock
from mile.models.transition import RSSM
from mile.models.gen_networks import WorldGen
from mile.models.dis_networks import MsImageDis
import torch.nn.init as init
import matplotlib.pyplot as plt
from mile.losses import SegmentationLoss, KLLoss, RegressionLoss, SpatialRegressionLoss
import torchfile
import torch.nn.functional as F
from torch.autograd import Variable
from mile.models.bev_networks import BEVGen
from mile.constants import BIRDVIEW_COLOURS


class Mile_UINT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.gen_a = WorldGen(self.cfg)
        self.gen_b = WorldGen(self.cfg)
        self.dis_a = MsImageDis(self.cfg)
        self.dis_b = MsImageDis(self.cfg)
        self.bevgen = BEVGen(self.cfg)
        # self.load_pretrained_weights()
        self.save_dir = os.path.join(
        self.cfg.LOG_DIR, 'validate_save'
    )

        self.gen_a.rgb_decoder.apply(weights_init('kaiming'))
        self.gen_b.rgb_decoder.apply(weights_init('kaiming'))
        self.dis_a.apply(weights_init('gaussian'))
        self.dis_b.apply(weights_init('gaussian'))
        # self.image_std = cfg.IMAGE.IMAGENET_STD
        # self.image_mean = cfg.IMAGE.IMAGENET_MEAN

        self.rgb_loss = SpatialRegressionLoss(norm=1)
        self.probabilistic_loss = KLLoss(alpha=self.cfg.LOSSES.KL_BALANCING_ALPHA)
        self.segmentation_loss = SegmentationLoss(
                use_top_k=self.cfg.SEMANTIC_SEG.USE_TOP_K,
                top_k_ratio=self.cfg.SEMANTIC_SEG.TOP_K_RATIO,
                use_weights=self.cfg.SEMANTIC_SEG.USE_WEIGHTS,
                )

        self.center_loss = SpatialRegressionLoss(norm=2)
        self.offset_loss = SpatialRegressionLoss(norm=1, ignore_index=self.cfg.INSTANCE_SEG.IGNORE_INDEX)

        self.vgg = load_vgg16('model_vgg/')
        self.vgg.eval()
        for param in self.vgg.parameters():
            param.requires_grad = False
        self.instancenorm = nn.InstanceNorm2d(512, affine=False)

    def load_pretrained_weights(self):
        if self.cfg.PRETRAINED.RESUME:
            # print(111,self.PRETRAINED.PATH)
            #self.cfg.PRETRAINED.PATH =  '/home/hello/Dujiatong/mile-main-0414/model-resume.ckpt'
            if os.path.isfile(self.cfg.PRETRAINED.PATH):
                checkpoint = torch.load(self.cfg.PRETRAINED.PATH, map_location='cpu')['state_dict']
                checkpoint = {key[6:]: value for key, value in checkpoint.items() if key[:5] == 'model'}
                # for name in checkpoint.keys():
                #     print(name)

                self.gen_a.load_state_dict(checkpoint, strict=True)
                for name, param in self.gen_a.named_parameters():
                    if 'rgb_decoder' not in name:  # 冻结除了 rgb_decoder 以外的所有参数
                        param.requires_grad = False
                print(f'Loaded weights from: {self.cfg.PRETRAINED.PATH} and virtual is freezed')
                for name, param in self.gen_a.named_parameters():
                    if not param.requires_grad:
                        print(f"Froze: {name}")
            else:
                raise FileExistsError(self.cfg.PRETRAINED.PATH)
        else:
            print('do not need to reload')
    
    def forward_dis(self, batch):
        batch_a = batch['data_carla']
        batch_b = batch['data_nusc']
        output = {}
        h_a, noise_a = self.gen_a.gen_encoder(batch_a, batch_a['image'])
        h_b, noise_b = self.gen_b.gen_encoder(batch_b, batch_b['image'])
        x_ba = self.gen_a.gen_decoder(h_b)
        x_ab = self.gen_b.gen_decoder(h_a)
        #loss_dis
        losses_dis = dict()
        losses_dis['dis_a_ba'] = self.cfg.DIS.GAN_WEIGHT * self.dis_a.calc_dis_loss(x_ba['rgb_label_1'].detach(), batch_a['image'])
        losses_dis['dis_b_ab'] = self.cfg.DIS.GAN_WEIGHT * self.dis_b.calc_dis_loss(x_ab['rgb_label_1'].detach(), batch_b['image'])
        output['ba'] = x_ba
        output['ab'] = x_ab
        return output, losses_dis
    
    def forward_gen(self, batch):
        batch_a = batch['data_carla']
        batch_b = batch['data_nusc']
        output = {}
        h_a, noise_a = self.gen_a.gen_encoder(batch_a, batch_a['image'])
        h_b, noise_b = self.gen_b.gen_encoder(batch_b, batch_b['image'])
        #within domain
        x_a_recon = self.gen_a.gen_decoder(h_a + noise_a)
        x_b_recon = self.gen_b.gen_decoder(h_b + noise_b)
        #cross domain
        x_ba = self.gen_a.gen_decoder(h_b + noise_b)
        x_ab = self.gen_b.gen_decoder(h_a + noise_a)
        #encode again
        h_b_recon, noise_b_recon = self.gen_a.gen_encoder(batch_b, x_ba['rgb_label_1'])
        h_a_recon, noise_a_recon = self.gen_b.gen_encoder(batch_a, x_ab['rgb_label_1'])
        #decode again
        x_aba = self.gen_a.gen_decoder(h_a_recon + noise_a_recon)
        x_bab = self.gen_b.gen_decoder(h_b_recon + noise_b_recon)
        #decode bev
        bev_a = self.bevgen.forward_bev(batch_a, h_a + noise_a)
        bev_cycle_a = self.bevgen.forward_bev(batch_a, h_a_recon + noise_a_recon)
        # bev_cycle_a = self.gen_a.decoder_bev(batch_a, h_a_recon)
        
        #loss
        losses_gen = dict()
        #重建loss
        losses_gen.update(self.cal_rgb_loss(x_a_recon, batch_a, 'x_a_recon')) #10
        losses_gen.update(self.cal_rgb_loss(x_b_recon, batch_b, 'x_b_recon'))
        losses_gen['kl_b'] = 0.01 * self.__compute_kl(h_b) #0.01
        losses_gen['kl_a'] = 0.01 * self.__compute_kl(h_a) #0.01
        #循环重建loss
        losses_gen.update(self.cal_rgb_loss(x_aba, batch_a, 'x_a_cycle'))
        losses_gen.update(self.cal_rgb_loss(x_bab, batch_b, 'x_b_cycle'))
        losses_gen['kl_b_recon'] = 0.01 * self.__compute_kl(h_b_recon) #0.01
        losses_gen['kl_a_recon'] = 0.01 * self.__compute_kl(h_a_recon) #0.01
        #bev loss
        losses_gen.update(self.cal_bev_loss(bev_a, batch_a, 'bev_carla')) #1
        losses_gen.update(self.cal_bev_loss(bev_cycle_a, batch_a, 'bev_cycle_carla'))
        #分布的kl散度，同一张图片的不同风格迁移让他们分布相同
        # h_recon_probabilistic_loss = self.probabilistic_loss(mu_b, sigma_b, mu_ba, sigma_ba)
        # losses_gen['h_b_probabilistic'] = self.cfg.LOSSES.WEIGHT_PROBABILISTIC * h_recon_probabilistic_loss #0.01  2
        # h_probabilistic_loss = self.probabilistic_loss(mu_a, sigma_a, mu_ab, sigma_ab)
        # losses_gen['h_a_probabilistic'] = self.cfg.LOSSES.WEIGHT_PROBABILISTIC * h_probabilistic_loss
        # GAN loss
        losses_gen['loss_gen_adv_a'] = self.cfg.DIS.GAN_WEIGHT * self.dis_a.calc_gen_loss(x_ba['rgb_label_1']) #0.1 1
        losses_gen['loss_gen_adv_b'] = self.cfg.DIS.GAN_WEIGHT *self.dis_b.calc_gen_loss(x_ab['rgb_label_1'])
        # VGG loss
        losses_gen['loss_gen_vgg_b'] = 0.1 * self.compute_vgg_loss(self.vgg, x_ba['rgb_label_1'], x_b_recon['rgb_label_1'])
        losses_gen['loss_gen_vgg_a'] = 0.1 * self.compute_vgg_loss(self.vgg, x_ab['rgb_label_1'], x_a_recon['rgb_label_1'])
        output['a_recon'] = x_a_recon
        output['b_recon'] = x_b_recon
        output['ba'] = x_ba
        output['aba'] = x_aba
        output['bab'] = x_bab
        return output, losses_gen
    
    def forward_val(self, batch, batch_idx):
        # batch_a = batch['data_carla']
        batch_a = batch['data_carla']
        batch_b = batch['data_nusc']

        h_a, noise_a = self.gen_a.gen_encoder(batch_a, batch_a['image'])
        h_b, noise_b = self.gen_b.gen_encoder(batch_b, batch_b['image'])
        #within domain
        x_a_recon = self.gen_a.gen_decoder(h_a)
        x_b_recon = self.gen_b.gen_decoder(h_b)
        #cross domain
        x_ba = self.gen_a.gen_decoder(h_b)
        x_ab = self.gen_b.gen_decoder(h_a)
        #encode again
        h_b_recon, noise_b_recon = self.gen_a.gen_encoder(batch_b, x_ba['rgb_label_1'])
        h_a_recon, noise_a_recon = self.gen_b.gen_encoder(batch_a, x_ab['rgb_label_1'])
        #decode again
        x_aba = self.gen_a.gen_decoder(h_a_recon)
        x_bab = self.gen_b.gen_decoder(h_b_recon)
        #decode bev
        bev_a = self.bevgen.forward_bev(batch_a, h_a)
        bev_cycle_a = self.bevgen.forward_bev(batch_a, h_a_recon)

        self.squezze_and_split_save(batch_b['image'], 'image_b', batch_idx)
        self.squezze_and_split_save(batch_a['image'], 'image_a', batch_idx)
        self.squezze_and_split_save(x_b_recon['rgb_label_1'], 'x_b_recon', batch_idx)
        self.squezze_and_split_save(x_a_recon['rgb_label_1'], 'x_a_recon', batch_idx)
        self.squezze_and_split_save(x_ba['rgb_label_1'], 'x_ba', batch_idx)
        self.squezze_and_split_save(x_ab['rgb_label_1'], 'x_ab', batch_idx)
        self.squezze_and_split_save(x_aba['rgb_label_1'], 'x_aba', batch_idx)
        self.squezze_and_split_save(x_bab['rgb_label_1'], 'x_bab', batch_idx)
        # print(bev_a['bev_segmentation_1'].shape)
        # self.squezze_and_split_save(bev_a['bev_segmentation_1'], 'bev_a', batch_idx)
        # self.squezze_and_split_save(bev_cycle_a['bev_segmentation_1'], 'bev_cycle_a', batch_idx)
        self.squezze_and_split_save_bev(bev_cycle_a['bev_segmentation_1'], 'bev_cycle_a_bev', batch_idx)
        self.squezze_and_split_save_bev(bev_a['bev_segmentation_1'], 'bev_a_bev', batch_idx)
        
    def __compute_kl(self, mu):
        # def _compute_kl(self, mu, sd):
        # mu_2 = torch.pow(mu, 2)
        # sd_2 = torch.pow(sd, 2)
        # encoding_loss = (mu_2 + sd_2 - torch.log(sd_2)).sum() / mu_2.size(0)
        # return encoding_loss
        mu_2 = torch.pow(mu, 2)
        encoding_loss = torch.mean(mu_2)
        return encoding_loss
    
    def squezze_and_split_save(self, img, str, batch_idx):
        tensor_squeezed = img.squeeze(1)  # 形状变为 (b, 1, 3, 320, 768)
        # 将 Tensor 拆分成两个独立的 Tensor
        for b in range(self.cfg.BATCHSIZE):
            image = tensor_squeezed[b].permute(1, 2, 0).cpu().numpy()  #  (3, 320, 768)
            # tensor = tensor * self.image_std + self.image_mean
            # image = np.clip(tensor, 0, 1)
            plt.imsave(os.path.join(self.save_dir, str + '_' + f'{b}' + f'{batch_idx}' + '.png'), image)

    def squezze_and_split_save_bev(self, img, str, batch_idx):
        # print(img.shape)
        tensor_squeezed = img.squeeze(1)  # 形状为 (b, 1, 3, 320, 768)
        # 将 Tensor 拆分成两个独立的 Tensor
        for b in range(self.cfg.BATCHSIZE):
            pred = tensor_squeezed[b]
            pred = torch.argmax(pred.detach(), dim=-3)
            colours = torch.tensor(BIRDVIEW_COLOURS, dtype=torch.uint8, device=pred.device)
            pred = colours[pred]
            # pred = pred.permute(0, 1, 4, 2, 3)
            # print(pred.shape)
            visualisation_video = torch.rot90(pred, k=1, dims=[0, 1]).cpu().numpy()
            # image = tensor_squeezed[b].permute(1, 2, 0).cpu().numpy()  #  (3, 320, 768)
            # tensor = tensor * self.image_std + self.image_mean
            # image = np.clip(tensor, 0, 1)
            plt.imsave(os.path.join(self.save_dir, str + '_' + f'{b}' + f'{batch_idx}' + '.png'), visualisation_video)

    def squezze_and_split_save_recon(self, img, str, batch_idx):
        tensor_squeezed = img.squeeze(1)  # 形状变为 (b, 1, 3, 320, 768)
        # 将 Tensor 拆分成两个独立的 Tensor
        for b in range(self.cfg.BATCHSIZE):
            tensor = tensor_squeezed[b].permute(1, 2, 0).cpu().numpy()  #  (3, 320, 768)
            # tensor = tensor * self.image_std + self.image_mean
            image = np.clip(tensor, 0, 1)
            plt.imsave(os.path.join(self.save_dir, str + '_' + f'{b}' + f'{batch_idx}' + '.png'), image)
    
    def cal_rgb_loss(self, x, batch, zifu):
        loss_rgb = dict()
        downsampling_factor = 1
        # for downsampling_factor in [1, 2, 4]:
        rgb_weight = 10
        discount = 1 / downsampling_factor
        rgb_loss = self.rgb_loss(
            prediction=x[f'rgb_label_{downsampling_factor}'],
            target=batch['image'],
        )
        loss_rgb[f'{zifu}_rencon_rgb_{downsampling_factor}'] = rgb_weight * discount * rgb_loss
        return loss_rgb
    
    def cal_bev_loss(self, output, batch, zifu):
        loss_bev = dict()
        for downsampling_factor in [1, 2, 4]:
            bev_segmentation_loss = self.segmentation_loss(
                prediction=output[f'bev_segmentation_{downsampling_factor}'],
                target=batch[f'birdview_label_{downsampling_factor}'],
            )
            discount = 1/downsampling_factor
            loss_bev[f'{zifu}_bev_segmentation_{downsampling_factor}'] = discount * self.cfg.LOSSES.WEIGHT_SEGMENTATION * \
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

            loss_bev[f'{zifu}_bev_center_{downsampling_factor}'] = discount * self.cfg.LOSSES.WEIGHT_INSTANCE * center_loss
            # Offset are already discounted in the labels
            loss_bev[f'{zifu}_bev_offset_{downsampling_factor}'] = self.cfg.LOSSES.WEIGHT_INSTANCE * offset_loss
        return loss_bev
    
    def compute_vgg_loss(self, vgg, img, target):
        img = pack_sequence_dim(img)
        target = pack_sequence_dim(target)
        img_vgg = vgg_preprocess(img)
        target_vgg = vgg_preprocess(target)
        img_fea = vgg(img_vgg)
        target_fea = vgg(target_vgg)
        return torch.mean((self.instancenorm(img_fea) - self.instancenorm(target_fea)) ** 2)


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

def load_vgg16(model_dir):
    """ Use the model from https://github.com/abhiskk/fast-neural-style/blob/master/neural_style/utils.py """
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    if not os.path.exists(os.path.join(model_dir, 'vgg16.weight')):
        if not os.path.exists(os.path.join(model_dir, 'vgg16.t7')):
            os.system('wget https://www.dropbox.com/s/76l3rt4kyi3s8x7/vgg16.t7?dl=1 -O ' + os.path.join(model_dir, 'vgg16.t7'))
        vgglua = torchfile.load(os.path.join(model_dir, 'vgg16.t7'))
        vgg = Vgg16()
        for (src, dst) in zip(vgglua.parameters()[0], vgg.parameters()):
            dst.data[:] = src
        torch.save(vgg.state_dict(), os.path.join(model_dir, 'vgg16.weight'))
    vgg = Vgg16()
    vgg.load_state_dict(torch.load(os.path.join(model_dir, 'vgg16.weight')))
    return vgg

class Vgg16(nn.Module):
    def __init__(self):
        super(Vgg16, self).__init__()
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

    def forward(self, X):
        h = F.relu(self.conv1_1(X), inplace=True)
        h = F.relu(self.conv1_2(h), inplace=True)
        # relu1_2 = h
        h = F.max_pool2d(h, kernel_size=2, stride=2)

        h = F.relu(self.conv2_1(h), inplace=True)
        h = F.relu(self.conv2_2(h), inplace=True)
        # relu2_2 = h
        h = F.max_pool2d(h, kernel_size=2, stride=2)

        h = F.relu(self.conv3_1(h), inplace=True)
        h = F.relu(self.conv3_2(h), inplace=True)
        h = F.relu(self.conv3_3(h), inplace=True)
        # relu3_3 = h
        h = F.max_pool2d(h, kernel_size=2, stride=2)

        h = F.relu(self.conv4_1(h), inplace=True)
        h = F.relu(self.conv4_2(h), inplace=True)
        h = F.relu(self.conv4_3(h), inplace=True)
        # relu4_3 = h

        h = F.relu(self.conv5_1(h), inplace=True)
        h = F.relu(self.conv5_2(h), inplace=True)
        h = F.relu(self.conv5_3(h), inplace=True)
        relu5_3 = h

        return relu5_3
    
def vgg_preprocess(batch):
    tensortype = type(batch)
    (r, g, b) = torch.chunk(batch, 3, dim = 1)
    batch = torch.cat((b, g, r), dim = 1) # convert RGB to BGR
    batch = (batch + 1) * 255 * 0.5 # [-1, 1] -> [0, 255]
    mean = tensortype(batch.size()).cuda()
    mean[:, 0, :, :] = 103.939
    mean[:, 1, :, :] = 116.779
    mean[:, 2, :, :] = 123.680
    batch = batch.sub(Variable(mean)) # subtract mean
    return batch