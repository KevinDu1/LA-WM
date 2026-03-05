"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
from utils import get_all_data_loaders, prepare_sub_folder, write_html, write_loss, get_config, write_2images, Timer
import argparse
from torch.autograd import Variable
from trainer import MUNIT_Trainer, UNIT_Trainer
from trainer_mile_bev_test import MILE_Trainer
import torch.backends.cudnn as cudnn
import torch
try:
    from itertools import izip as zip
except ImportError: # will be 3.x series
    pass
import os
import sys
import tensorboardX
import shutil

from dataset_real_virtual import RealDataset, CarlaDataset
from mile.data.NuscenesData import FuturePredictionDataset
from nuscenes.nuscenes import NuScenes
from torch.utils.data import DataLoader, random_split
from config import get_cfg_djt
from mile.models.preprocess import PreProcess
from batch_a_intrin_extrin import calculate_geometry_from_config

os.environ['CUDA_VISIBLE_DEVICES'] = '2'
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='configs/uint_carla2real_bev.yaml', help='Path to the config file.')
parser.add_argument('--output_path', type=str, default='.', help="outputs path")
parser.add_argument("--resume", default=True, action="store_true")
# parser.add_argument('--trainer', type=str, default='MUNIT', help="MUNIT|UNIT")
opts = parser.parse_args()

cudnn.benchmark = True

# Load experiment setting
config = get_config(opts.config)
max_iter = config['max_iter']
display_size = config['display_size']
config['vgg_model_path'] = opts.output_path
batch_size = config['batch_size']
num_workers = config['num_workers']

# Setup model and data loader
# if opts.trainer == 'MUNIT':
#     trainer = MUNIT_Trainer(config)
# elif opts.trainer == 'UNIT':
#     trainer = UNIT_Trainer(config)
# else:
#     sys.exit("Only support MUNIT|UNIT")
trainer = MILE_Trainer(config)
trainer.cuda()
# train_loader_a, train_loader_b, test_loader_a, test_loader_b = get_all_data_loaders(config)
# dataset_carla = CarlaDataset(sequence_length=1)
# nusc = NuScenes(version='v1.0-mini', dataroot='/media/ps/data/dujiatong/nusc_mini', verbose=True)
# nusc = NuScenes(version='v1.0-trainval', dataroot='/media/ps/data/dujiatong/nusc', verbose=True)
# dataset_nusc = FuturePredictionDataset(nusc=nusc, is_train=0)
dataset_nusc = RealDataset()
# train_dataset_carla, test_dataset_carla = random_split(dataset_carla, [int(0.99 * len(dataset_carla)), len(dataset_carla) - int(0.99 * len(dataset_carla))])
train_dataset_nusc, test_dataset_nusc = random_split(dataset_nusc, [int(0.99 * len(dataset_nusc)), len(dataset_nusc) - int(0.99 * len(dataset_nusc))])
# train_loader_a = DataLoader(dataset=train_dataset_carla, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers)
train_loader_b = DataLoader(dataset=train_dataset_nusc, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers)
# test_loader_a = DataLoader(dataset=test_dataset_carla, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers)
# test_loader_b = DataLoader(dataset=test_dataset_nusc, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers)

# train_display_images_a = torch.stack([train_loader_a.dataset[i]['image'] for i in range(display_size)]).cuda()
# train_display_images_b = torch.stack([train_loader_b.dataset[i]['image'] for i in range(display_size)]).cuda()
# test_display_images_a = torch.stack([test_loader_a.dataset[i]['image'] for i in range(display_size)]).cuda()
# test_display_images_b = torch.stack([test_loader_b.dataset[i]['image'] for i in range(display_size)]).cuda()
# print(f'carla:{train_display_images_a.shape}')
# print(f'nusc:{train_display_images_b.shape}')

# Setup logger and output folders
model_name = os.path.splitext(os.path.basename(opts.config))[0]
train_writer = tensorboardX.SummaryWriter(os.path.join(opts.output_path + "/logs", model_name))
output_directory = os.path.join(opts.output_path + "/outputs", model_name)
checkpoint_directory, image_directory = prepare_sub_folder(output_directory)
shutil.copy(opts.config, os.path.join(output_directory, 'config.yaml')) # copy config file to output folder

cfg = get_cfg_djt()
preprocess = PreProcess(cfg)

# Start training
iterations = trainer.resume(checkpoint_directory, hyperparameters=config) if opts.resume else 0
trainer.dongzhu()
iterations = 0

intr, extr = calculate_geometry_from_config(cfg)
intr = torch.from_numpy(intr).cuda()
extr = torch.from_numpy(extr).cuda()
while True:
    for it, batch_b in enumerate(train_loader_b):
        trainer.update_learning_rate()
        # images_a, images_b = images_a.cuda().detach(), images_b.cuda().detach()
        # batch_a = preprocess.prepare_bev_labels(batch_a)
        # batch_b = preprocess.prepare_traj_labels(batch_b)
        # batch_a = {key: value.cuda().detach() if isinstance(value, torch.Tensor) else value for key, value in batch_a.items()}
        batch_b = {key: value.cuda().detach() if isinstance(value, torch.Tensor) else value for key, value in batch_b.items()}

        with Timer("Elapsed time in update: %f"):
            # Main training code
            # trainer.dis_update(batch_a, batch_b, config)
            # trainer.gen_update(batch_a, batch_b, config)
            # trainer.bev_update(batch_a, batch_b, config)
            trainer.traj_update(batch_b, config, intr, extr)
            torch.cuda.synchronize()

        # Dump training stats in log file
        if (iterations + 1) % config['log_iter'] == 0:
            print("Iteration: %08d/%08d" % (iterations + 1, max_iter))
            write_loss(iterations, trainer, train_writer)

        # Write images
        # if (iterations + 1) % config['image_save_iter'] == 0:
        #     with torch.no_grad():
        #         test_image_outputs = trainer.sample(test_display_images_a, test_display_images_b)
        #         train_image_outputs = trainer.sample(train_display_images_a, train_display_images_b)
        #     write_2images(test_image_outputs, display_size, image_directory, 'test_%08d' % (iterations + 1))
        #     write_2images(train_image_outputs, display_size, image_directory, 'train_%08d' % (iterations + 1))
        #     # HTML
        #     write_html(output_directory + "/index.html", iterations + 1, config['image_save_iter'], 'images')

        # if (iterations + 1) % config['image_display_iter'] == 0:
        #     with torch.no_grad():
        #         image_outputs = trainer.sample(train_display_images_a, train_display_images_b)
        #     write_2images(image_outputs, display_size, image_directory, 'train_current')

        # Save network weights
        if (iterations + 1) % config['snapshot_save_iter'] == 0:
            trainer.save(checkpoint_directory, iterations)

        iterations += 1
        if iterations >= max_iter:
            sys.exit('Finish training')
