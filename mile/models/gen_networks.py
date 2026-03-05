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


class WorldGen(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.receptive_field = cfg.RECEPTIVE_FIELD

        # Image feature encoder
        # if self.cfg.MODEL.ENCODER.NAME == 'resnet18':
        #     self.encoder = timm.create_model(
        #         cfg.MODEL.ENCODER.NAME, pretrained=True, features_only=True, out_indices=[2, 3, 4],
        #     )
        #     feature_info = self.encoder.feature_info.get_dicts(keys=['num_chs', 'reduction'])

        # self.feat_decoder = Decoder(feature_info, self.cfg.MODEL.ENCODER.OUT_CHANNELS)

        self.contenc = ContentEncoder(3, 4, 3, 64, 'in', 'relu', pad_type='reflect')

        # if not self.cfg.EVAL.NO_LIFTING:
        #     # Frustum pooling
        #     bev_downsample = cfg.BEV.FEATURE_DOWNSAMPLE
        #     self.frustum_pooling = FrustumPooling(
        #         size=(cfg.BEV.SIZE[0] // bev_downsample, cfg.BEV.SIZE[1] // bev_downsample),
        #         scale=cfg.BEV.RESOLUTION * bev_downsample,
        #         offsetx=cfg.BEV.OFFSET_FORWARD / bev_downsample,
        #         dbound=cfg.BEV.FRUSTUM_POOL.D_BOUND,
        #         downsample=8,
        #     )

        #     # mono depth head
        #     self.depth_decoder = Decoder(feature_info, self.cfg.MODEL.ENCODER.OUT_CHANNELS)
        #     self.depth = nn.Conv2d(self.depth_decoder.out_channels, self.frustum_pooling.D, kernel_size=1)
        #     # only lift argmax of depth distribution for speed
        #     self.sparse_depth = cfg.BEV.FRUSTUM_POOL.SPARSE
        #     self.sparse_depth_count = cfg.BEV.FRUSTUM_POOL.SPARSE_COUNT

        # backbone_bev_in_channels = self.cfg.MODEL.ENCODER.OUT_CHANNELS

        # Bev network
        # self.backbone_bev = timm.create_model(
        #     cfg.MODEL.BEV.BACKBONE,
        #     in_chans=backbone_bev_in_channels,
        #     pretrained=True,
        #     features_only=True,
        #     out_indices=[3],
        # )

        # feature_info_bev = self.backbone_bev.feature_info.get_dicts(keys=['num_chs', 'reduction'])
        # embedding_n_channels = self.cfg.MODEL.EMBEDDING_DIM
        # # embedding_n_feature = 3*embedding_n_channels
        # self.final_state_conv = nn.Sequential(
        #     BasicBlock(feature_info_bev[-1]['num_chs'], embedding_n_channels, stride=2, downsample=True),
        #     BasicBlock(embedding_n_channels, embedding_n_channels),
        #     nn.AdaptiveAvgPool2d(output_size=(1, 1)),
        #     nn.Flatten(start_dim=1),
        # )

        # # Recurrent model
        # self.receptive_field = self.cfg.RECEPTIVE_FIELD
        # # if self.cfg.MODEL.TRANSITION.ENABLED:
        # #     state_dim = self.cfg.MODEL.TRANSITION.HIDDEN_STATE_DIM + self.cfg.MODEL.TRANSITION.STATE_DIM
        # # else:
        # #     state_dim = embedding_n_channels
        # state_dim = self.cfg.MODEL.TRANSITION.STATE_DIM
        # # Bird's-eye view semantic segmentation
        # if self.cfg.SEMANTIC_SEG.ENABLED:
        #     self.bev_decoder = BevDecoder(
        #         latent_n_channels=state_dim,
        #         semantic_n_channels=self.cfg.SEMANTIC_SEG.N_CHANNELS,
        #     )

        # RGB reconstruction
        if self.cfg.EVAL.RGB_SUPERVISION:
            # self.rgb_decoder = RGBDecoder(
            #     latent_n_channels=state_dim,
            #     semantic_n_channels=3,
            #     constant_size=(5, 12),
            #     is_segmentation=False,
            # )
            # self.rgb_decoder = UpsampleConvDecoder(latent_n_channels= 64)
            self.rgb_decoder = ResDecoder(n_upsample=3, n_res=4, dim=512, output_dim=3,)

        # Used during deployment to save last state
        self.last_h = None
        self.last_sample = None
        self.last_action = None
        self.count = 0

        # self.posterior = RepresentationModel(
        #     in_channels=embedding_n_channels,
        #     latent_dim=self.cfg.MODEL.TRANSITION.STATE_DIM,
        # )

        # self.dec = Decoder(n_downsample, n_res, 512, input_dim, res_norm='in', activ=activ, pad_type=pad_type)

    def gen_encoder(self, batch, img):
        embedding = self.encode(batch, img)
        # state_mu, state_sigma = self.posterior(embedding)
        # sample = self.sample_from_distribution(state_mu, state_sigma, use_sample=True)
        noise = Variable(torch.randn(embedding.size()).cuda(embedding.data.get_device()))
        # noise = Variable(torch.randn(embedding.size()))
        return embedding, noise
        # return embedding
    
    def gen_decoder(self, embedding):
        b, s = embedding.shape[:2]
        sam = pack_sequence_dim(embedding)
        rgb_decoder_output = self.rgb_decoder(sam)
        rgb_decoder_output = unpack_sequence_dim(rgb_decoder_output, b, s)
        return rgb_decoder_output
    
    # def gen_decoder(self, emebeddings):
    #     images = self.dec(emebeddings)
    #     return images


    def encode(self, batch, img):
        b, s = batch['image'].shape[:2]
        image = pack_sequence_dim(img)
        # speed = pack_sequence_dim(batch['speed'])

        # Image encoder, multiscale
        # xs = self.encoder(image)
        x = self.contenc(image)

        # # Lift features to bird's-eye view.
        # # Aggregate features to output resolution (H/8, W/8)
        # x = self.feat_decoder(xs)

        # if not self.cfg.EVAL.NO_LIFTING:
        #     # Depth distribution
        #     intrinsics = pack_sequence_dim(batch['intrinsics'])
        #     extrinsics = pack_sequence_dim(batch['extrinsics'])
        #     depth = self.depth(self.depth_decoder(xs)).softmax(dim=1)

        #     if self.sparse_depth:
        #         # only lift depth for topk most likely depth bins
        #         topk_bins = depth.topk(self.sparse_depth_count, dim=1)[1]
        #         depth_mask = torch.zeros(depth.shape, device=depth.device, dtype=torch.bool)
        #         depth_mask.scatter_(1, topk_bins, 1)
        #     else:
        #         depth_mask = torch.zeros(0, device=depth.device)
        #     x = (depth.unsqueeze(1) * x.unsqueeze(2)).type_as(x)  # outer product

        #     #  Add camera dimension
        #     x = x.unsqueeze(1)
        #     x = x.permute(0, 1, 3, 4, 5, 2) #b,1,37,40,96,64

        #     x = self.frustum_pooling(x, intrinsics.unsqueeze(1), extrinsics.unsqueeze(1), depth_mask)

        # embedding = self.backbone_bev(x)[-1]
        # embedding = self.final_state_conv(embedding)
        # embedding = unpack_sequence_dim(embedding, b, s)
        embedding = unpack_sequence_dim(x, b, s)
        return embedding
    
    def sample_from_distribution(self, mu, sigma, use_sample):
        sample = mu
        if use_sample:
            noise = torch.randn_like(sample)
            sample = sample + sigma * noise
        return sample
    
    def decoder_bev(self, batch, sample):
        b, s = batch['image'].shape[:2]

        output = dict()

        state = pack_sequence_dim(sample)
    
        bev_decoder_output = self.bev_decoder(state)
        bev_decoder_output = unpack_sequence_dim(bev_decoder_output, b, s)
        output = {**output, **bev_decoder_output}
        return output
    
    def forward_bev(self, batch, deployment):
        """
        Parameters
        ----------
            batch: dict of torch.Tensor
                keys:
                    image: (b, s, 3, h, w)
                    route_map: (b, s, 3, h_r, w_r)
                    speed: (b, s, 1)
                    intrinsics: (b, s, 3, 3)
                    extrinsics: (b, s, 4, 4)
                    throttle_brake: (b, s, 1)
                    steering: (b, s, 1)
        """
        # Encode RGB images, route_map, speed using intrinsics and extrinsics
        # to a 512 dimensional vector
        #print(batch['speed'].shape)
        sample, state_mu, state_sigma = self.gen_encoder(batch, batch['image'])
        b, s = batch['image'].shape[:2]

        output = dict()

        state = pack_sequence_dim(sample)
    

        if self.cfg.SEMANTIC_SEG.ENABLED:
            if (not deployment) or (deployment and DISPLAY_SEGMENTATION):
                bev_decoder_output = self.bev_decoder(state)
                bev_decoder_output = unpack_sequence_dim(bev_decoder_output, b, s)
                output = {**output, **bev_decoder_output}


        return output

class ResDecoder(nn.Module):
    def __init__(self, n_upsample, n_res, dim, output_dim, res_norm='in', activ='relu', pad_type='zero'):
        super(ResDecoder, self).__init__()

        self.model = []
        # AdaIN residual blocks
        # self.model += [ResBlocks(n_res, dim, res_norm, activ, pad_type=pad_type)]
        # updim = 512
        # self.model += [Conv2dBlock(dim, updim, 5, 1, 2, norm='ln', activation=activ, pad_type=pad_type)]
        # dim = updim
        self.model += [ResBlocks(n_res, dim, res_norm, activ, pad_type=pad_type)]
        # upsampling blocks
        for i in range(n_upsample):
            self.model += [nn.Upsample(scale_factor=2),
                           Conv2dBlock(dim, dim // 2, 5, 1, 2, norm='ln', activation=activ, pad_type=pad_type)]
            dim //= 2

        # self.model += [ResBlocks(n_res, dim, res_norm, activ, pad_type=pad_type)]
        
        # use reflection padding in the last conv layer
        # self.model += [nn.Upsample(size=(320, 768), mode='bilinear', align_corners=False)]  # (768x768 -> 320x768)
        self.model += [Conv2dBlock(dim, output_dim, 7, 1, 3, norm='none', activation='tanh', pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return {
            f'rgb_label_1': self.model(x),
        }

class UpsampleConvDecoder(nn.Module):
    def __init__(self, latent_n_channels, output_channels=3):
        super(UpsampleConvDecoder, self).__init__()

        # 上采样层 + 卷积层设计
        # 第一步：上采样（从 48x48 到 96x96）
        self.res1 = ResBlocks(4, latent_n_channels, 'in', 'relu', pad_type='zero')
        self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)  # (48x48 -> 96x96)
        self.conv1 = nn.Conv2d(latent_n_channels, 128, kernel_size=3, padding=1)

        # 第二步：上采样（从 96x96 到 192x192）
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)  # (96x96 -> 192x192)
        self.conv2 = nn.Conv2d(128, 256, kernel_size=3, padding=1)

        # 第三步：上采样（从 192x192 到 384x384）
        self.upsample3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)  # (192x192 -> 384x384)
        self.conv3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)

        # 第四步：上采样（从 384x384 到 768x768）
        self.upsample4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)  # (384x384 -> 768x768)
        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)

        self.res2 = ResBlocks(4, 256, 'in', 'relu', pad_type='zero')

        # 第五步：从 768x768 到 320x768
        self.upsample5 = nn.Upsample(size=(320, 768), mode='bilinear', align_corners=False)  # (768x768 -> 320x768)
        self.conv5 = nn.Conv2d(256, output_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.res1(x)
        x = self.upsample1(x)
        x = torch.relu(self.conv1(x))

        x = self.upsample2(x)
        x = torch.relu(self.conv2(x))

        x = self.upsample3(x)
        x = torch.relu(self.conv3(x))

        x = self.upsample4(x)
        x = torch.relu(self.conv4(x))

        x = self.res2(x)

        x = self.upsample5(x)
        x = self.conv5(x)  # 最后一层卷积输出图像，应用 sigmoid 或其他激活函数取决于任务
        
        return {
            f'rgb_label_1': torch.sigmoid(x),
        } # 如果是图像重建任务，可以使用 sigmoid 激活函数（归一化到 [0, 1]）

class ResBlocks(nn.Module):
    def __init__(self, num_blocks, dim, norm='in', activation='relu', pad_type='zero'):
        super(ResBlocks, self).__init__()
        self.model = []
        for i in range(num_blocks):
            self.model += [ResBlock(dim, norm=norm, activation=activation, pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)
    
class ResBlock(nn.Module):
    def __init__(self, dim, norm='in', activation='relu', pad_type='zero'):
        super(ResBlock, self).__init__()

        model = []
        model += [Conv2dBlock(dim ,dim, 3, 1, 1, norm=norm, activation=activation, pad_type=pad_type)]
        model += [Conv2dBlock(dim ,dim, 3, 1, 1, norm=norm, activation='none', pad_type=pad_type)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        residual = x
        out = self.model(x)
        out += residual
        return out
    
class Conv2dBlock(nn.Module):
    def __init__(self, input_dim ,output_dim, kernel_size, stride,
                 padding=0, norm='none', activation='relu', pad_type='zero'):
        super(Conv2dBlock, self).__init__()
        self.use_bias = True
        # initialize padding
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # initialize normalization
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'in':
            #self.norm = nn.InstanceNorm2d(norm_dim, track_running_stats=True)
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == 'ln':
            self.norm = LayerNorm(norm_dim)
        elif norm == 'adain':
            self.norm = AdaptiveInstanceNorm2d(norm_dim)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # initialize convolution
        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride, bias=self.use_bias)

    def forward(self, x):
        x = self.conv(self.pad(x))
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x

class AdaptiveInstanceNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(AdaptiveInstanceNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        # weight and bias are dynamically assigned
        self.weight = None
        self.bias = None
        # just dummy buffers, not used
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, x):
        assert self.weight is not None and self.bias is not None, "Please assign weight and bias before calling AdaIN!"
        b, c = x.size(0), x.size(1)
        running_mean = self.running_mean.repeat(b)
        running_var = self.running_var.repeat(b)

        # Apply instance norm
        x_reshaped = x.contiguous().view(1, b * c, *x.size()[2:])

        out = F.batch_norm(
            x_reshaped, running_mean, running_var, self.weight, self.bias,
            True, self.momentum, self.eps)

        return out.view(b, c, *x.size()[2:])

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.num_features) + ')'


class LayerNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super(LayerNorm, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps

        if self.affine:
            self.gamma = nn.Parameter(torch.Tensor(num_features).uniform_())
            self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        shape = [-1] + [1] * (x.dim() - 1)
        # print(x.size())
        if x.size(0) == 1:
            # These two lines run much faster in pytorch 0.4 than the two lines listed below.
            mean = x.view(-1).mean().view(*shape)
            std = x.view(-1).std().view(*shape)
        else:
            mean = x.view(x.size(0), -1).mean(1).view(*shape)
            std = x.view(x.size(0), -1).std(1).view(*shape)

        x = (x - mean) / (std + self.eps)

        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)
            x = x * self.gamma.view(*shape) + self.beta.view(*shape)
        return x
    
class ContentEncoder(nn.Module):
    def __init__(self, n_downsample, n_res, input_dim, dim, norm, activ, pad_type):
        super(ContentEncoder, self).__init__()
        self.model = []
        self.model += [Conv2dBlock(input_dim, dim, 7, 1, 3, norm=norm, activation=activ, pad_type=pad_type)]
        # downsampling blocks
        for i in range(n_downsample):
            self.model += [Conv2dBlock(dim, 2 * dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)]
            dim *= 2
        # residual blocks
        self.model += [ResBlocks(n_res, dim, norm=norm, activation=activ, pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)
        self.output_dim = dim

    def forward(self, x):
        return self.model(x)