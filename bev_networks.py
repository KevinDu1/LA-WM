import torch
import torch.nn as nn
import timm
from torch.autograd import Variable
import torch.nn.functional as F

from mile.constants import CARLA_FPS, DISPLAY_SEGMENTATION
from mile.utils.network_utils import pack_sequence_dim, unpack_sequence_dim, remove_past
from mile.models.common import BevDecoder, Decoder, RouteEncode, Policy, RGBDecoder
from mile.models.frustum_pooling import FrustumPooling
from mile.layers.layers import BasicBlock
from mile.models.transition import RSSM, RepresentationModel

from mile.models.gen_networks import Conv2dBlock, ResBlocks


class BEVGen(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        # self.receptive_field = cfg.RECEPTIVE_FIELD
        state_dim = 512
        self.bev_decoder = BevDecoder(latent_n_channels=state_dim, semantic_n_channels=cfg['SEMANTIC.SEG_N_CHANNELS'])

        self.feat_decoder = Conv2dBlock(256 ,64, 5, 1, 2, norm='bn', activation='relu')
        self.depth_decoder = Conv2dBlock(256 ,64, 5, 1, 2, norm='bn', activation='relu')

        # Frustum pooling
        bev_downsample = cfg['BEV.FEATURE_DOWNSAMPLE']
        self.frustum_pooling = FrustumPooling(
            size=(cfg['BEV.SIZE'][0] // bev_downsample, cfg['BEV.SIZE'][1] // bev_downsample),
            scale=cfg['BEV.RESOLUTION'] * bev_downsample,
            offsetx=cfg['BEV.OFFSET_FORWARD'] / bev_downsample,
            dbound=cfg['BEV.FRUSTUM_POOL.D_BOUND'],
            downsample=8,
        )

        self.depth = nn.Conv2d(64, self.frustum_pooling.D, kernel_size=1)
        self.sparse_depth = False

        #换成自己的降维的
        # self.backbone_bev = timm.create_model(
        #     cfg.MODEL.BEV.BACKBONE,
        #     in_chans=64,
        #     pretrained=True,
        #     features_only=True,
        #     out_indices=[3],
        # )
        self.backbone_bev = DownDecoder(n_upsample=2, n_res=3, dim=64)

        # feature_info_bev = self.backbone_bev.feature_info.get_dicts(keys=['num_chs', 'reduction'])
        embedding_n_channels = 512
        # embedding_n_feature = 3*embedding_n_channels
        self.final_state_conv = nn.Sequential(
            BasicBlock(256, embedding_n_channels, stride=2, downsample=True),
            BasicBlock(embedding_n_channels, embedding_n_channels),
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(start_dim=1),
        )


    def encoder_bev(self, batch, sample):
        b, s = batch['image'].shape[:2]
        state = sample
        # state = pack_sequence_dim(sample)

        x = self.feat_decoder(state)

            # Depth distribution
        intrinsics = pack_sequence_dim(batch['intrinsics'])
        extrinsics = pack_sequence_dim(batch['extrinsics'])
        depth = self.depth(self.depth_decoder(state)).softmax(dim=1)

        if self.sparse_depth:
            # only lift depth for topk most likely depth bins
            topk_bins = depth.topk(self.sparse_depth_count, dim=1)[1]
            depth_mask = torch.zeros(depth.shape, device=depth.device, dtype=torch.bool)
            depth_mask.scatter_(1, topk_bins, 1)
        else:
            depth_mask = torch.zeros(0, device=depth.device)
        x = (depth.unsqueeze(1) * x.unsqueeze(2)).type_as(x)  # outer product

        #  Add camera dimension
        x = x.unsqueeze(1)
        x = x.permute(0, 1, 3, 4, 5, 2) #b,1,37,40,96,64

        x = self.frustum_pooling(x, intrinsics.unsqueeze(1), extrinsics.unsqueeze(1), depth_mask)

        embedding = self.backbone_bev(x)
        embedding = self.final_state_conv(embedding)
        embedding = unpack_sequence_dim(embedding, b, s)
        return embedding
    
    def decoder_bev(self, batch, sample):
        b, s = batch['image'].shape[:2]

        output = dict()

        state = pack_sequence_dim(sample)
    
        bev_decoder_output = self.bev_decoder(state)
        bev_decoder_output = unpack_sequence_dim(bev_decoder_output, b, s)
        output = {**output, **bev_decoder_output}
        return output
    
    def forward_bev(self, batch, sample):
        emb = self.encoder_bev(batch, sample)
        return self.decoder_bev(batch, emb)
    
    def forward_bev_b(self, batch, sample, intr, extr):
        emb = self.encoder_bev_b(batch, sample, intr, extr)
        return emb
    
    def encoder_bev_b(self, batch, sample, intr, extr):
        b, s = batch['image'].shape[:2]
        state = sample
        # state = pack_sequence_dim(sample)

        x = self.feat_decoder(state)

            # Depth distribution
        # intrinsics = pack_sequence_dim(batch['intrinsics'])
        # extrinsics = pack_sequence_dim(batch['extrinsics'])
        intrinsics = intr.unsqueeze(0).repeat(b*s, 1, 1)
        extrinsics = extr.unsqueeze(0).repeat(b*s, 1, 1)
        depth = self.depth(self.depth_decoder(state)).softmax(dim=1)

        if self.sparse_depth:
            # only lift depth for topk most likely depth bins
            topk_bins = depth.topk(self.sparse_depth_count, dim=1)[1]
            depth_mask = torch.zeros(depth.shape, device=depth.device, dtype=torch.bool)
            depth_mask.scatter_(1, topk_bins, 1)
        else:
            depth_mask = torch.zeros(0, device=depth.device)
        x = (depth.unsqueeze(1) * x.unsqueeze(2)).type_as(x)  # outer product

        #  Add camera dimension
        x = x.unsqueeze(1)
        x = x.permute(0, 1, 3, 4, 5, 2) #b,1,37,40,96,64

        x = self.frustum_pooling(x, intrinsics.unsqueeze(1), extrinsics.unsqueeze(1), depth_mask)

        embedding = self.backbone_bev(x)
        embedding = self.final_state_conv(embedding)
        embedding = unpack_sequence_dim(embedding, b, s)
        return embedding
    
class DownDecoder(nn.Module):
    def __init__(self, n_upsample, n_res, dim, res_norm='in', activ='relu', pad_type='zero'):
        super(DownDecoder, self).__init__()

        self.model = []
        # AdaIN residual blocks
        # self.model += [ResBlocks(n_res, dim, res_norm, activ, pad_type=pad_type)]
        # updim = 512
        # self.model += [Conv2dBlock(dim, updim, 5, 1, 2, norm='ln', activation=activ, pad_type=pad_type)]
        # dim = updim
        self.model += [ResBlocks(n_res, dim, res_norm, activ, pad_type=pad_type)]
        # upsampling blocks
        for i in range(n_upsample):
            self.model += [Conv2dBlock(dim, 2 * dim, 8, 4, 2, norm='bn', activation=activ, pad_type=pad_type)]
            dim *= 2

        # self.model += [ResBlocks(n_res, dim, res_norm, activ, pad_type=pad_type)]
        
        # use reflection padding in the last conv layer
        # self.model += [nn.Upsample(size=(320, 768), mode='bilinear', align_corners=False)]  # (768x768 -> 320x768)
        # self.model += [Conv2dBlock(dim, output_dim, 7, 1, 3, norm='none', activation='sigmoid', pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)
 

    def forward(self, x):
        return self.model(x)