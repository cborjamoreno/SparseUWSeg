##### spixel_utils #####
# This script contains utility functions including:
#
# -> find_mean_std: finds the mean and standard deviations for the Red, Green and Blue channel
# of an input image, such that the image can be normalized
#
# -> 

## IMPORTS ##
# Load necessary modules
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import math

from skimage.color import rgb2lab
from skimage.util import img_as_float

from scipy import interpolate

### Functions ###
class img2lab(object):
    def __call__(self, img):
        img = np.array(img)
        flt_img = img_as_float(img)
        lab_img = rgb2lab(flt_img)
        return (lab_img)
      
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, img):
        assert isinstance(img, np.ndarray)
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C x H x W
        img = img.transpose((2, 0, 1))
        return (torch.from_numpy(img))

class xylab(nn.Module):
    def __init__(self, color_scale, pos_scale_x, pos_scale_y):
        super(xylab, self).__init__()
        self.color_scale = color_scale
        self.pos_scale_x = pos_scale_x
        self.pos_scale_y = pos_scale_y

    def forward(self, Lab):
        ########## compute the XYLab features of the batch of images in Lab ########
        # 1. rgb2Lab
        # 2. create meshgrid of X, Y and expand it along the mini-batch dimension
        #
        # Lab:   tensor (shape = [N, 3, H, W]): the input image is already opened in LAB format via the Dataloader defined #        in "cityscapes.py" 
        # XY:    tensor (shape = [N, 2, H, W])
        # XYLab: tensor (shape = [N, 5, H, W])
        
        N = Lab.shape[0]
        H = Lab.shape[2]
        W = Lab.shape[3]
        
        # Create meshgrid on the same device as Lab and with explicit indexing to avoid warnings
        dev = Lab.device if isinstance(Lab, torch.Tensor) else None
        Y, X = torch.meshgrid(
            torch.arange(0, H, dtype=torch.float32, device=dev),
            torch.arange(0, W, dtype=torch.float32, device=dev),
            indexing="ij"
        )
        # print(Y.shape, X.shape)
        # print(Y, X)
        # print('X[None, None, :, :]', X[None, None, :, :].shape)
        # print('X[None, None, :, :].expand(N, -1, -1, -1)', X[None, None, :, :].expand(N, -1, -1, -1).shape)
        X = self.pos_scale_x *  X[None, None, :, :].expand(N, -1, -1, -1)                            # shape = [N, 1, H, W]
        # print(X)
        # print(X.shape)
        Y = self.pos_scale_y *  Y[None, None, :, :].expand(N, -1, -1, -1)                            # shape = [N, 1, H, W]
        Lab = self.color_scale * Lab.to(torch.float)                                               # requires casting as all input tensors to torch.cat must be of the same dtype

        # print(torch.cat((X, Y, Lab), dim = 1))
        # print(torch.cat((X, Y, Lab), dim = 1).shape)
        return torch.cat((X, Y, Lab), dim = 1), X, Y, Lab


def find_mean_std(img):
    # Finds the mean and standard deviation of each RGB channel of an input image

    total_pixel = img.shape[0] * img.shape[1]

    R_mean = np.sum(img[:,:,0]) / total_pixel
    G_mean = np.sum(img[:,:,1]) / total_pixel
    B_mean = np.sum(img[:,:,2]) / total_pixel

    R_std = math.sqrt( (np.sum((img[:, :, 0] - R_mean) ** 2)) / total_pixel)
    G_std = math.sqrt( (np.sum((img[:, :, 0] - G_mean) ** 2)) / total_pixel)
    B_std = math.sqrt( (np.sum((img[:, :, 0] - B_mean) ** 2)) / total_pixel)

    return [R_mean, G_mean, B_mean], [R_std, G_std, B_std]

def get_spixel_init(num_spixels, img_width, img_height):

    k = num_spixels
    k_w = int(np.floor(np.sqrt(k * img_width / img_height)))
    k_h = int(np.floor(np.sqrt(k * img_height / img_width)))

    # print(k_h,k_w)

    spixel_height = img_height / (1. * k_h)
    spixel_width = img_width / (1. * k_w)
    # print(spixel_width)

    # h_coords = np.arange(-spixel_height / 2., img_height + spixel_height - 1,
    #                      spixel_height)
    # w_coords = np.arange(-spixel_width / 2., img_width + spixel_width - 1,
    #                      spixel_width)


    h_coords = np.arange(-spixel_height / 2., img_height + spixel_height - 1,
                         spixel_height)
    w_coords = np.arange(-spixel_width / 2., img_width + spixel_width - 1,
                         spixel_width)
    
    # print(h_coords)
    # print(w_coords)
    
    spix_values = np.int32(np.arange(0, k_w * k_h).reshape((k_h, k_w)))
    spix_values = np.pad(spix_values, 1, 'symmetric')
    # print(spix_values)
    f = interpolate.RegularGridInterpolator((h_coords, w_coords), spix_values, method='nearest')

    all_h_coords = np.arange(0, img_height, 1)
    all_w_coords = np.arange(0, img_width, 1)
    all_grid = np.array(np.meshgrid(all_h_coords, all_w_coords, indexing = 'ij'))
    all_points = np.reshape(all_grid, (2, img_width * img_height)).transpose()
    # print(all_points)

    spixel_initmap = f(all_points).reshape((img_height,img_width))
    # print(spixel_initmap)

    feat_spixel_initmap = spixel_initmap
    return [spixel_initmap, feat_spixel_initmap, k_w, k_h]


def compute_init_spixel_feat(trans_feature, spixel_init, num_spixels):
    # initializes the (mean) features of each superpixel using the features encoded by the CNN "trans_feature"
    #
    # INPUTS:
    # 1) trans_feature:     (tensor of shape [B, C, H, W])
    # 2) spixel_init:       (tensor of shape [H, W])
    #
    # RETURNS:
    # 1) init_spixel_feat:  (tensor of shape [B, K, C])

    # Flatten and transpose trans_feature to shape [N, C, B]
    trans_feature = torch.flatten(trans_feature, start_dim=2).transpose(0, 2)  # shape = [N, C, B]

    # Flatten spixel_init to shape [N]
    spixel_init = spixel_init.flatten().long()  # shape = [N]


    # Create a tensor to hold the sum of features for each superpixel
    sum_spixel_feat = torch.zeros((num_spixels, trans_feature.size(1), trans_feature.size(2)), device=trans_feature.device)

    # Create a tensor to hold the count of elements in each superpixel
    count_spixel_feat = torch.zeros((num_spixels, 1, 1), device=trans_feature.device)

    # Scatter add the features to the corresponding superpixel
    sum_spixel_feat.index_add_(0, spixel_init, trans_feature)

    # Scatter add the counts to the corresponding superpixel
    ones = torch.ones_like(trans_feature[:, :1, :1])
    count_spixel_feat.index_add_(0, spixel_init, ones)

    # Compute the mean by dividing the sum by the count
    init_spixel_feat = torch.where(count_spixel_feat > 0, sum_spixel_feat / count_spixel_feat, torch.zeros_like(sum_spixel_feat))

    # Transpose the result to match the expected output shape [B, K, C]
    result = init_spixel_feat.transpose(0, 2).transpose(1, 2)
    return result   # shape = [B, K, C]
