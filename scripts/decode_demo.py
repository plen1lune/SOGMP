#!/usr/bin/env python
#
# file: $ISIP_EXP/SOGMP/scripts/decode_demo.py
#
# revision history: xzt
#  20220824 (TE): first version
#
# usage:
#  python decode_demo.py mdir mdl_path test_data
#
# arguments:
#  mdir: the directory where the output results are stored
#  mdl_path: the directory of training data
#  test_data: the directory of testing data
#
# This script decodes a SOGMP++ model and gives a result demo
#------------------------------------------------------------------------------

# import pytorch modules
#
import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm

# visualize:
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms as transforms
import torchvision
import matplotlib
from torchvision.utils import make_grid
matplotlib.style.use('ggplot')

# import modules
#
import sys
import os

# import the model and all of its variables/functions
#
from model import *
from local_occ_grid_map import LocalMap

#-----------------------------------------------------------------------------
#
# global variables are listed here
#
#-----------------------------------------------------------------------------

# general global values
#
NUM_ARGS = 3
IMG_SIZE = 64
SPACE = " "        
log_dir = '../model/model.pth'   

# Constants
NUM_CLASSES = 1
NUM_INPUT_CHANNELS = 1
NUM_LATENT_DIM = 512
NUM_OUTPUT_CHANNELS = NUM_CLASSES

# Init map parameters
P_prior = 0.5    # Prior occupancy probability
P_occ = 0.7      # Probability that cell is occupied with total confidence
P_free = 0.3     # Probability that cell is free with total confidence 
MAP_X_LIMIT = [0, 6.4]      # Map limits on the x-axis
MAP_Y_LIMIT = [-3.2, 3.2]   # Map limits on the y-axis
RESOLUTION = 0.1        # Grid resolution in [m]'
TRESHOLD_P_OCC = 0.8    # Occupancy threshold

# for reproducibility, we seed the rng
#
SEED1 = 42
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

set_seed(SEED1)        

#------------------------------------------------------------------------------
#
# the main program starts here
#
#------------------------------------------------------------------------------

# function: main
#
# arguments: none
#
# return: none
#
# This method is the main function.
#
def main(argv):
    # ensure we have the correct number of arguments:
    if(len(argv) != NUM_ARGS):
        print("usage: python decode_demo.py [ODIR] [MDL_PATH] [EVAL_SET]")
        exit(-1)

    # define local variables:
    odir = argv[0]
    mdl_path = argv[1]
    fImg = argv[2]

    # set the device to use GPU if available:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    eval_dataset = VaeTestDataset(fImg,'test')
    eval_dataloader = torch.utils.data.DataLoader(eval_dataset, batch_size=1, shuffle=False, drop_last=True)

    # instantiate a model:
    model = DiffusionModel(input_channels=NUM_INPUT_CHANNELS, latent_dim=NUM_LATENT_DIM, output_channels=NUM_OUTPUT_CHANNELS)
    model.to(device)

    # set the model to evaluate
    model.eval()

    # set the loss criterion:
    criterion = nn.MSELoss(reduction='sum')
    criterion.to(device)

    # load the weights
    checkpoint = torch.load(mdl_path, map_location=device)
    model.load_state_dict(checkpoint['model'])

    # for each batch in increments of batch size:
    counter = 0
    num_batches = int(len(eval_dataset) / eval_dataloader.batch_size)
    with torch.no_grad():
        for i, batch in tqdm(enumerate(eval_dataloader), total=num_batches):
            counter += 1
            scans = batch['scan'].to(device)
            positions = batch['position'].to(device)
            velocities = batch['velocity'].to(device)

            batch_size = scans.size(0)
            mask_gridMap = LocalMap(X_lim=MAP_X_LIMIT, Y_lim=MAP_Y_LIMIT, resolution=RESOLUTION, p=P_prior, size=[batch_size, SEQ_LEN], device=device)
            x_odom = torch.zeros(batch_size, SEQ_LEN).to(device)
            y_odom = torch.zeros(batch_size, SEQ_LEN).to(device)
            theta_odom = torch.zeros(batch_size, SEQ_LEN).to(device)
            distances = scans[:, SEQ_LEN:]
            angles = torch.linspace(-(135 * np.pi / 180), 135 * np.pi / 180, distances.shape[-1]).to(device)
            distances_x, distances_y = mask_gridMap.lidar_scan_xy(distances, angles, x_odom, y_odom, theta_odom)
            mask_binary_maps = mask_gridMap.discretize(distances_x, distances_y)
            mask_binary_maps = mask_binary_maps.unsqueeze(2)
            
            prediction_maps = torch.zeros(SEQ_LEN, 1, IMG_SIZE, IMG_SIZE).to(device)
            # multi-step prediction: 10 time steps:
            for j in range(SEQ_LEN):
                input_gridMap = LocalMap(X_lim=MAP_X_LIMIT, Y_lim=MAP_Y_LIMIT, resolution=RESOLUTION, p=P_prior, size=[batch_size, SEQ_LEN], device=device)
                obs_pos_N = positions[:, SEQ_LEN-1]
                vel_N = velocities[:, SEQ_LEN-1]
                T = j + 1
                noise_std = [0, 0, 0]
                pos_origin = input_gridMap.origin_pose_prediction(vel_N, obs_pos_N, T, noise_std)
                pos = positions[:, :SEQ_LEN]
                x_odom, y_odom, theta_odom = input_gridMap.robot_coordinate_transform(pos, pos_origin)
                distances = scans[:, :SEQ_LEN]
                angles = torch.linspace(-(135 * np.pi / 180), 135 * np.pi / 180, distances.shape[-1]).to(device)
                distances_x, distances_y = input_gridMap.lidar_scan_xy(distances, angles, x_odom, y_odom, theta_odom)
                input_binary_maps = input_gridMap.discretize(distances_x, distances_y)
                input_binary_maps = input_binary_maps.unsqueeze(2)

                num_samples = 32
                inputs_samples = input_binary_maps.repeat(num_samples, 1, 1, 1, 1)

                for t in range(T):
                    prediction = model(inputs_samples, torch.tensor([t] * num_samples).to(device))
                    prediction = prediction.reshape(-1, 1, 1, IMG_SIZE, IMG_SIZE)
                    inputs_samples = torch.cat([inputs_samples[:, 1:], prediction], dim=1)

                predictions = prediction.squeeze(1)
                pred_mean = torch.mean(predictions, dim=0, keepdim=True)
                prediction_maps[j, 0] = pred_mean.squeeze()

            fig = plt.figure(figsize=(8, 1))
            for m in range(SEQ_LEN):
                a = fig.add_subplot(1, 10, m + 1)
                mask = mask_binary_maps[0, m]
                input_grid = make_grid(mask.detach().cpu())
                input_image = input_grid.permute(1, 2, 0)
                plt.imshow(input_image)
                plt.xticks([])
                plt.yticks([])
                fontsize = 8
                input_title = "n=" + str(m + 1)
                a.set_title(input_title, fontdict={'fontsize': fontsize})
            input_img_name = os.path.join(odir, "mask" + str(i) + ".jpg")
            plt.savefig(input_img_name)

            fig = plt.figure(figsize=(8, 1))
            for m in range(SEQ_LEN):
                a = fig.add_subplot(1, 10, m + 1)
                pred = prediction_maps[m]
                input_grid = make_grid(pred.detach().cpu())
                input_image = input_grid.permute(1, 2, 0)
                plt.imshow(input_image)
                plt.xticks([])
                plt.yticks([])
                input_title = "n=" + str(m + 1)
                a.set_title(input_title, fontdict={'fontsize': fontsize})
            input_img_name = os.path.join(odir, "pred" + str(i) + ".jpg")
            plt.savefig(input_img_name)
            plt.show()

            print(i)

    return True

if __name__ == '__main__':
    main(sys.argv[1:])
