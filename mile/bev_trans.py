from mile.data.NuscenesData import FuturePredictionDataset
from nuscenes.nuscenes import NuScenes
from torch.utils.data import DataLoader
import torch
from PIL import Image
import numpy as np
from torchvision import transforms
from mile.constants import BIRDVIEW_COLOURS

# tensor_ones = torch.ones(2, 7, 3)
# print(tensor_ones[:,:1,:].shape)
# relative_traj = tensor_ones - tensor_ones[:,:1,:]
# print(relative_traj.shape)
# print(relative_traj[:,1:,:].shape)

#测试nusc数据集
# nusc = NuScenes(version='v1.0-mini', dataroot='/media/ps/data/dujiatong/nusc_mini', verbose=True)
# dataset_nusc = FuturePredictionDataset(nusc=nusc, is_train=0)
# dataloader = DataLoader(dataset_nusc, batch_size=2, shuffle=True)

# # print('bye')
# for batch_idx, data in enumerate(dataloader):
#     print(data)
# #     print(1)
#     if batch_idx == 2:  # 只打印前3个批次数据
#         break
def integer_to_binary(integer_array, n_bits):
    """
    Parameters
    ----------
        integer_array: np.ndarray<int32> (n,)
        n_bits: int

    Returns
    -------
        binary_array: np.ndarray<float32> (n, n_bits)

    """
    return (((integer_array[:, None] & (1 << np.arange(n_bits)))) > 0).astype(np.float32)

def calculate_birdview_labels(birdview, n_classes, has_time_dimension=False):
    """
    Parameters
    ----------
        birdview: torch.Tensor<float> (C, H, W)
        n_classes: int
            number of total classes
        has_time_dimension: bool

    Returns
    -------
        birdview_label: (H, W)
    """
    # When a pixel contains two labels, argmax will output the first one that is encountered.
    # By reversing the order, we prioritise traffic lights over road.
    dim = 0
    if has_time_dimension:
        dim = 1
    birdview_label = torch.argmax(birdview.flip(dims=[dim]), dim=dim)
    # We then re-normalise the classes in the normal order.
    birdview_label = (n_classes - 1) - birdview_label
    return birdview_label

def trans_black_to_bev(birdview):
    pred = torch.argmax(birdview, dim=-3)
    colours = torch.tensor(BIRDVIEW_COLOURS, dtype=torch.uint8, device=pred.device)
    pred = colours[pred]
    # pred = pred.permute(0, 1, 4, 2, 3)
    # target = birdview[:, :, 0]
    # colours = torch.tensor(BIRDVIEW_COLOURS, dtype=torch.uint8, device=target.device)

    # target = colours[target]
    return pred

def bev_save(bev, image_directory, postfix):
    for index, value in enumerate(bev):
        # print(value.shape)
        m = value.cpu().numpy()
        img = Image.fromarray(m)
        img.save('%s/test/gen_%s_a2b_%s_%s.jpg' % (image_directory, postfix, index, 0))
        # for b in range(value.shape[0]):
        #     m = value[b].cpu().numpy()
        #     img = Image.fromarray(m)
        #     img.save('%s/test/gen_%s_a2b_%s_%s.jpg' % (image_directory, postfix, index, b))