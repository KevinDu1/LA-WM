import torch
import torch.nn as nn
import timm

from mile.constants import CARLA_FPS, DISPLAY_SEGMENTATION
from mile.utils.network_utils import pack_sequence_dim, unpack_sequence_dim, remove_past
from mile.models.common import BevDecoder, Decoder, RouteEncode, Policy, Ncppolicy
from mile.models.frustum_pooling import FrustumPooling
from mile.layers.layers import BasicBlock
from mile.models.transition import RSSM
from config import get_cfg_djt


class Mile(nn.Module):
    def __init__(self):
        super().__init__()
        cfg = get_cfg_djt()
        self.cfg = cfg
        self.receptive_field = cfg.RECEPTIVE_FIELD

        # Speed as input
        self.speed_enc = nn.Sequential(
            nn.Linear(1, cfg.MODEL.SPEED.CHANNELS),
            nn.ReLU(True),
            nn.Linear(cfg.MODEL.SPEED.CHANNELS, cfg.MODEL.SPEED.CHANNELS),
            nn.ReLU(True),
        )
        self.speed_normalisation = cfg.SPEED.NORMALISATION
        self.command_enc = nn.Sequential(
            nn.Linear(1, cfg.MODEL.SPEED.COMMAND),
            nn.ReLU(True),
            nn.Linear(cfg.MODEL.SPEED.COMMAND, cfg.MODEL.SPEED.COMMAND),
            nn.ReLU(True),
        )
        
        self.target_enc = nn.Sequential(
            nn.Linear(2, cfg.MODEL.SPEED.COMMAND),
            nn.ReLU(True),
            nn.Linear(cfg.MODEL.SPEED.COMMAND, cfg.MODEL.SPEED.COMMAND),
            nn.ReLU(True),
        )
        
        # self.speed_normalisation = cfg.SPEED.NORMALISATION
        
        # Recurrent model
        self.receptive_field = self.cfg.RECEPTIVE_FIELD
        self.future_field = self.cfg.FUTURE_HORIZON
        embedding_n_channels = 512 + 2*cfg.MODEL.SPEED.COMMAND + cfg.MODEL.SPEED.CHANNELS
        # if self.cfg.MODEL.TRANSITION.ENABLED:
        #     # Recurrent state sequence module
        #     self.rssm = RSSM(
        #         embedding_dim=embedding_n_channels,
        #         action_dim=self.cfg.MODEL.ACTION_DIM,
        #         hidden_state_dim=self.cfg.MODEL.TRANSITION.HIDDEN_STATE_DIM,
        #         state_dim=self.cfg.MODEL.TRANSITION.STATE_DIM,
        #         action_latent_dim=self.cfg.MODEL.TRANSITION.ACTION_LATENT_DIM,
        #         receptive_field=self.receptive_field,
        #         use_dropout=self.cfg.MODEL.TRANSITION.USE_DROPOUT,
        #         dropout_probability=self.cfg.MODEL.TRANSITION.DROPOUT_PROBABILITY,
        #     )

        # # Policy
        # if self.cfg.MODEL.TRANSITION.ENABLED:
        #     state_dim = self.cfg.MODEL.TRANSITION.HIDDEN_STATE_DIM + self.cfg.MODEL.TRANSITION.STATE_DIM
        # else:
        #     state_dim = embedding_n_channels
        

        self.policy = Ncppolicy(in_channels=embedding_n_channels, fut=self.future_field)


        # Used during deployment to save last state
        self.last_h = None
        self.last_sample = None
        self.last_action = None
        self.count = 0

    def forward(self, batch, emb):
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
        b, s = batch['image'].shape[:2]
        embedding = self.encode(batch, emb)  # dim (b, s, 512)

        # output = dict()

        # action = batch['gt_trajectory']

        # state_dict = self.rssm(embedding, action)

        # # output = {**output, **state_dict}
        # state = torch.cat([state_dict['posterior']['hidden_state'], state_dict['posterior']['sample']], dim=-1)

        # state = pack_sequence_dim(state)
        #print(state.shape) #[B*12,1024+512]
        output_policy = self.policy(embedding)
        # output_policy = unpack_sequence_dim(output_policy, b, s)
        #print(output_policy.shape) #[B*12,2]
        # output['traj'] = output_policy
        return output_policy

    def encode(self, batch, emb):
        b, s = batch['image'].shape[:2]
        speed = batch['speed'].float()
        command = batch['command'].float()
        target = batch['target_point'].float()

        speed_features = self.speed_enc(speed)
        command_features = self.command_enc(command).repeat(1, s, 1)
        target_features = self.target_enc(target).repeat(1, s, 1)

        x = torch.cat((emb, speed_features, command_features, target_features), -1)

        return x

    # def observe_and_imagine(self, batch, predict_action=False, future_horizon=None):
    #     """ This is only used for visualisation of future prediction"""
    #     assert self.cfg.MODEL.TRANSITION.ENABLED and self.cfg.SEMANTIC_SEG.ENABLED
    #     if future_horizon is None:
    #         future_horizon = self.cfg.FUTURE_HORIZON

    #     b, s = batch['image'].shape[:2]

    #     if not predict_action:
    #         assert batch['throttle_brake'].shape[1] == s + future_horizon
    #         assert batch['steering'].shape[1] == s + future_horizon

    #     # Observe past context
    #     output_observe = self.forward(batch)

    #     # Imagine future states
    #     output_imagine = {
    #         'action': [],
    #         'state': [],
    #         'hidden': [],
    #         'sample': [],
    #     }
    #     h_t = output_observe['posterior']['hidden_state'][:, -1]
    #     sample_t = output_observe['posterior']['sample'][:, -1]
    #     for t in range(future_horizon):
    #         if predict_action:
    #             action_t = self.policy(torch.cat([h_t, sample_t], dim=-1))
    #         else:
    #             action_t = torch.cat([batch['throttle_brake'][:, s+t], batch['steering'][:, s+t]], dim=-1)
    #         prior_t = self.rssm.imagine_step(
    #             h_t, sample_t, action_t, use_sample=True, policy=self.policy,
    #         )
    #         sample_t = prior_t['sample']
    #         h_t = prior_t['hidden_state']
    #         output_imagine['action'].append(action_t)
    #         output_imagine['state'].append(torch.cat([h_t, sample_t], dim=-1))
    #         output_imagine['hidden'].append(h_t)
    #         output_imagine['sample'].append(sample_t)

    #     for k, v in output_imagine.items():
    #         output_imagine[k] = torch.stack(v, dim=1)

    #     bev_decoder_output = self.bev_decoder(pack_sequence_dim(output_imagine['state']))
    #     bev_decoder_output = unpack_sequence_dim(bev_decoder_output, b, future_horizon)
    #     output_imagine = {**output_imagine, **bev_decoder_output}

    #     return output_observe, output_imagine

    # def imagine(self, batch, predict_action=False, future_horizon=None):
    #     """ This is only used for visualisation of future prediction"""
    #     assert self.cfg.MODEL.TRANSITION.ENABLED and self.cfg.SEMANTIC_SEG.ENABLED
    #     if future_horizon is None:
    #         future_horizon = self.cfg.FUTURE_HORIZON

    #     # Imagine future states
    #     output_imagine = {
    #         'action': [],
    #         'state': [],
    #         'hidden': [],
    #         'sample': [],
    #     }
    #     h_t = batch['hidden_state'] #(b, c)
    #     sample_t = batch['sample']  #(b, s)
    #     b = h_t.shape[0]
    #     for t in range(future_horizon):
    #         if predict_action:
    #             action_t = self.policy(torch.cat([h_t, sample_t], dim=-1))
    #         else:
    #             action_t = torch.cat([batch['throttle_brake'][:, t], batch['steering'][:, t]], dim=-1)
    #         prior_t = self.rssm.imagine_step(
    #             h_t, sample_t, action_t, use_sample=True, policy=self.policy,
    #         )
    #         sample_t = prior_t['sample']
    #         h_t = prior_t['hidden_state']
    #         output_imagine['action'].append(action_t)
    #         output_imagine['state'].append(torch.cat([h_t, sample_t], dim=-1))
    #         output_imagine['hidden'].append(h_t)
    #         output_imagine['sample'].append(sample_t)

    #     for k, v in output_imagine.items():
    #         output_imagine[k] = torch.stack(v, dim=1)

    #     bev_decoder_output = self.bev_decoder(pack_sequence_dim(output_imagine['state']))
    #     bev_decoder_output = unpack_sequence_dim(bev_decoder_output, b, future_horizon)
    #     output_imagine = {**output_imagine, **bev_decoder_output}

    #     return output_imagine

    # def deployment_forward(self, batch, is_dreaming):
    #     """
    #     Keep latent states in memory for fast inference.

    #     Parameters
    #     ----------
    #         batch: dict of torch.Tensor
    #             keys:
    #                 image: (b, s, 3, h, w)
    #                 route_map: (b, s, 3, h_r, w_r)
    #                 speed: (b, s, 1)
    #                 intrinsics: (b, s, 3, 3)
    #                 extrinsics: (b, s, 4, 4)
    #                 throttle_brake: (b, s, 1)
    #                 steering: (b, s, 1)
    #     """
    #     assert self.cfg.MODEL.TRANSITION.ENABLED
    #     b = batch['image'].shape[0]

    #     if self.count == 0:
    #         # Encode RGB images, route_map, speed using intrinsics and extrinsics
    #         # to a 512 dimensional vector
    #         s = batch['image'].shape[1]
    #         action_t = batch['action'][:, -2]  # action from t-1 to t
    #         batch = remove_past(batch, s)
    #         embedding_t = self.encode(batch)[:, -1]  # dim (b, 1, 512)

    #         # Recurrent state sequence module
    #         if self.last_h is None:
    #             h_t = action_t.new_zeros(b, self.cfg.MODEL.TRANSITION.HIDDEN_STATE_DIM)
    #             sample_t = action_t.new_zeros(b, self.cfg.MODEL.TRANSITION.STATE_DIM)
    #         else:
    #             h_t = self.last_h
    #             sample_t = self.last_sample

    #         if is_dreaming:
    #             rssm_output = self.rssm.imagine_step(
    #                 h_t, sample_t, action_t, use_sample=False, policy=self.policy,
    #             )
    #         else:
    #             rssm_output = self.rssm.observe_step(
    #                 h_t, sample_t, action_t, embedding_t, use_sample=False, policy=self.policy,
    #             )['posterior']
    #         sample_t = rssm_output['sample']
    #         h_t = rssm_output['hidden_state']

    #         self.last_h = h_t
    #         self.last_sample = sample_t

    #         game_frequency = CARLA_FPS
    #         model_stride_sec = self.cfg.DATASET.STRIDE_SEC
    #         n_image_per_stride = int(game_frequency * model_stride_sec)
    #         self.count = n_image_per_stride - 1
    #     else:
    #         self.count -= 1
    #     s = 1
    #     state = torch.cat([self.last_h, self.last_sample], dim=-1)
    #     output_policy = self.policy(state)
    #     throttle_brake, steering = torch.split(output_policy, 1, dim=-1)
    #     output = dict()
    #     output['throttle_brake'] = unpack_sequence_dim(throttle_brake, b, s)
    #     output['steering'] = unpack_sequence_dim(steering, b, s)

    #     output['hidden_state'] = self.last_h
    #     output['sample'] = self.last_sample

    #     if self.cfg.SEMANTIC_SEG.ENABLED and DISPLAY_SEGMENTATION:
    #         bev_decoder_output = self.bev_decoder(state)
    #         bev_decoder_output = unpack_sequence_dim(bev_decoder_output, b, s)
    #         output = {**output, **bev_decoder_output}

    #     return output
