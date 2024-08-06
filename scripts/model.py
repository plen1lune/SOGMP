#!/usr/bin/env python
#
# file: $ISIP_EXP/SOGMP/scripts/model.py
#
# revision history: xzt
#  20220824 (TE): first version
#
# usage:
#
# This script hold the model architecture
#------------------------------------------------------------------------------

# import pytorch modules
#
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict
from convlstm import ConvLSTMCell

# import modules
#
import os
import random

# for reproducibility, we seed the rng
#
SEED1 = 1337
NEW_LINE = "\n"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#-----------------------------------------------------------------------------
#
# helper functions are listed here
#
#-----------------------------------------------------------------------------

# function: set_seed
#
# arguments: seed - the seed for all the rng
#
# returns: none
#
# this method seeds all the random number generators and makes
# the results deterministic
#
def set_seed(seed):
    #torch.manual_seed(seed)
    #torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    #random.seed(seed)
    #os.environ['PYTHONHASHSEED'] = str(seed)
#
# end of method


# function: get_data
#
# arguments: fp - file pointer
#            num_feats - the number of features in a sample
#
# returns: data - the signals/features
#          labels - the correct labels for them
#
# this method takes in a fp and returns the data and labels
POINTS = 1080
IMG_SIZE = 64 
SEQ_LEN = 10
class VaeTestDataset(torch.utils.data.Dataset):
    def __init__(self, img_path, file_name):
        # initialize the data and labels
        # read the names of image data:
        self.scan_file_names = []
        self.pos_file_names = []
        self.vel_file_names = []
        # open train.txt or dev.txt:
        fp_scan = open(img_path+'/scans/'+file_name+'.txt', 'r')
        fp_pos = open(img_path+'/positions/'+file_name+'.txt', 'r')
        fp_vel = open(img_path+'/velocities/'+file_name+'.txt', 'r')
        # for each line of the file:
        for line in fp_scan.read().split(NEW_LINE):
            if('.npy' in line): 
                self.scan_file_names.append(img_path+'/scans/'+line)
        for line in fp_pos.read().split(NEW_LINE):
            if('.npy' in line): 
                self.pos_file_names.append(img_path+'/positions/'+line)
        for line in fp_vel.read().split(NEW_LINE):
            if('.npy' in line): 
                self.vel_file_names.append(img_path+'/velocities/'+line)
        # close txt file:
        fp_scan.close()
        fp_pos.close()
        fp_vel.close()
        self.length = len(self.scan_file_names)

        print("dataset length: ", self.length)


    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # get the index of start point:
        scans = np.zeros((SEQ_LEN+SEQ_LEN, POINTS))
        positions = np.zeros((SEQ_LEN+SEQ_LEN, 3))
        vels = np.zeros((SEQ_LEN+SEQ_LEN, 2))
        # get the index of start point:
        if(idx+(SEQ_LEN+SEQ_LEN) < self.length): # train1:
            idx_s = idx
        else:
            idx_s = idx - (SEQ_LEN+SEQ_LEN)

        for i in range(SEQ_LEN+SEQ_LEN):
            # get the scan data:
            scan_name = self.scan_file_names[idx_s+i]
            scan = np.load(scan_name)
            scans[i] = scan
            # get the scan_ur data:
            pos_name = self.pos_file_names[idx_s+i]
            pos = np.load(pos_name)
            positions[i] = pos
            # get the velocity data:
            vel_name = self.vel_file_names[idx_s+i]
            vel = np.load(vel_name)
            vels[i] = vel
        
        # initialize:
        scans[np.isnan(scans)] = 20.
        scans[np.isinf(scans)] = 20.
        scans[scans==30] = 20.

        positions[np.isnan(positions)] = 0.
        positions[np.isinf(positions)] = 0.

        vels[np.isnan(vels)] = 0.
        vels[np.isinf(vels)] = 0.

        # transfer to pytorch tensor:
        scan_tensor = torch.FloatTensor(scans)
        pose_tensor = torch.FloatTensor(positions)
        vel_tensor =  torch.FloatTensor(vels)

        data = {
                'scan': scan_tensor,
                'position': pose_tensor,
                'velocity': vel_tensor, 
                }

        return data

#
# end of function


#------------------------------------------------------------------------------
#
# the model is defined here
#
#------------------------------------------------------------------------------

# define the PyTorch VAE model
#
# define a VAE
# Residual blocks: 
class Residual(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_hiddens):
        super(Residual, self).__init__()
        self._block = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=in_channels,
                      out_channels=num_residual_hiddens,
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_residual_hiddens),
            nn.ReLU(),
            nn.Conv2d(in_channels=num_residual_hiddens,
                      out_channels=num_hiddens,
                      kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(num_hiddens)
        )
    
    def forward(self, x):
        return x + self._block(x)

class ResidualStack(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(ResidualStack, self).__init__()
        self._num_residual_layers = num_residual_layers
        self._layers = nn.ModuleList([Residual(in_channels, num_hiddens, num_residual_hiddens)
                             for _ in range(self._num_residual_layers)])

    def forward(self, x):
        for i in range(self._num_residual_layers):
            x = self._layers[i](x)
        return F.relu(x)

# Encoder & Decoder Architecture:
# Encoder:
class Encoder(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(Encoder, self).__init__()
        self._conv_1 = nn.Sequential(*[
                                        nn.Conv2d(in_channels=in_channels,
                                                  out_channels=num_hiddens//2,
                                                  kernel_size=4,
                                                  stride=2, 
                                                  padding=1),
                                        nn.BatchNorm2d(num_hiddens//2),
                                        nn.ReLU()
                                    ])
        self._conv_2 = nn.Sequential(*[
                                        nn.Conv2d(in_channels=num_hiddens//2,
                                                  out_channels=num_hiddens,
                                                  kernel_size=4,
                                                  stride=2, 
                                                  padding=1),
                                        nn.BatchNorm2d(num_hiddens)
                                        #nn.ReLU()
                                    ])
        self._residual_stack = ResidualStack(in_channels=num_hiddens,
                                             num_hiddens=num_hiddens,
                                             num_residual_layers=num_residual_layers,
                                             num_residual_hiddens=num_residual_hiddens)

    def forward(self, inputs):
        x = self._conv_1(inputs)
        x = self._conv_2(x)
        x = self._residual_stack(x)
        return x

# Decoder:
class Decoder(nn.Module):
    def __init__(self, out_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(Decoder, self).__init__()
        
        self._residual_stack = ResidualStack(in_channels=num_hiddens,
                                             num_hiddens=num_hiddens,
                                             num_residual_layers=num_residual_layers,
                                             num_residual_hiddens=num_residual_hiddens)

        self._conv_trans_2 = nn.Sequential(*[
                                            nn.ReLU(),
                                            nn.ConvTranspose2d(in_channels=num_hiddens,
                                                              out_channels=num_hiddens//2,
                                                              kernel_size=4,
                                                              stride=2,
                                                              padding=1),
                                            nn.BatchNorm2d(num_hiddens//2),
                                            nn.ReLU()
                                        ])

        self._conv_trans_1 = nn.Sequential(*[
                                            nn.ConvTranspose2d(in_channels=num_hiddens//2,
                                                              out_channels=num_hiddens//2,
                                                              kernel_size=4,
                                                              stride=2,
                                                              padding=1),
                                            nn.BatchNorm2d(num_hiddens//2),
                                            nn.ReLU(),                  
                                            nn.Conv2d(in_channels=num_hiddens//2,
                                                      out_channels=out_channels,
                                                      kernel_size=3,
                                                      stride=1,
                                                      padding=1),
                                            nn.Sigmoid()
                                        ])

    def forward(self, inputs):
        x = self._residual_stack(inputs)
        x = self._conv_trans_2(x)
        x = self._conv_trans_1(x)
        return x

class VAE_Encoder(nn.Module):
    def __init__(self, input_channel):
        super(VAE_Encoder, self).__init__()
        # parameters:
        self.input_channels = input_channel
        # Constants
        num_hiddens = 128 #128
        num_residual_hiddens = 64 #32
        num_residual_layers = 2
        embedding_dim = 2 #64

        # encoder:
        in_channels = input_channel
        self._encoder = Encoder(in_channels, 
                                num_hiddens,
                                num_residual_layers, 
                                num_residual_hiddens)

        # z latent variable: 
        self._encoder_z_mu = nn.Conv2d(in_channels=num_hiddens, 
                                    out_channels=embedding_dim,
                                    kernel_size=1, 
                                    stride=1)
        self._encoder_z_log_sd = nn.Conv2d(in_channels=num_hiddens, 
                                    out_channels=embedding_dim,
                                    kernel_size=1, 
                                    stride=1)  
        
    def forward(self, x):
        # input reshape:
        x = x.reshape(-1, self.input_channels, IMG_SIZE, IMG_SIZE)
        # Encoder:
        encoder_out = self._encoder(x)
        # get `mu` and `log_var`:
        z_mu = self._encoder_z_mu(encoder_out)
        z_log_sd = self._encoder_z_log_sd(encoder_out)
        return z_mu, z_log_sd

# Example usage
input_channels = 1
latent_dim = 128
output_channels = 1
IMG_SIZE = 64
SEQ_LEN = 10
from convlstm import ConvLSTMCell
import numpy as np
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class UNet(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(UNet, self).__init__()
        self.in_channels = input_channels
        self.out_channels = output_channels

        self.enc1 = DoubleConv(input_channels, 64)
        self.enc2 = DoubleConv(64, 128)
        self.enc3 = DoubleConv(128, 256)
        self.enc4 = DoubleConv(256, 512)
        self.enc5 = DoubleConv(512, 1024)

        self.pool = nn.MaxPool2d(2)

        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(1024, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(128, 64)

        self.final_conv = nn.Conv2d(64, output_channels, kernel_size=1)

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))
        enc5 = self.enc5(self.pool(enc4))

        dec4 = self.up4(enc5)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.dec4(dec4)
        dec3 = self.up3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.dec3(dec3)
        dec2 = self.up2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.dec2(dec2)
        dec1 = self.up1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.dec1(dec1)

        return self.final_conv(dec1)
class MediumUNet(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(MediumUNet, self).__init__()
        self.in_channels = input_channels
        self.out_channels = output_channels

        self.enc1 = DoubleConv(input_channels, 32)
        self.enc2 = DoubleConv(32, 64)
        self.enc3 = DoubleConv(64, 128)

        self.pool = nn.MaxPool2d(2)

        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(128, 64)
        self.up2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(64, 32)
        self.final_conv = nn.Conv2d(32, output_channels, kernel_size=1)

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))

        dec1 = self.up1(enc3)
        dec1 = torch.cat((dec1, enc2), dim=1)
        dec1 = self.dec1(dec1)
        dec2 = self.up2(dec1)
        dec2 = torch.cat((dec2, enc1), dim=1)
        dec2 = self.dec2(dec2)

        return self.final_conv(dec2)
class ShallowUNet(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(ShallowUNet, self).__init__()
        self.in_channels = input_channels
        self.out_channels = output_channels

        self.enc1 = DoubleConv(input_channels, 32)
        self.enc2 = DoubleConv(32, 64)

        self.pool = nn.MaxPool2d(2)

        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(64, 32)
        self.final_conv = nn.Conv2d(32, output_channels, kernel_size=1)

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))

        dec1 = self.up1(enc2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.dec1(dec1)

        return self.final_conv(dec1)
    
class DiffusionModel(nn.Module):
    def __init__(self, input_channels, latent_dim, output_channels):
        super(DiffusionModel, self).__init__()
        # parameters:
        self.input_channels = input_channels
        self.latent_dim = latent_dim
        self.output_channels = output_channels
        self.z_w = int(np.sqrt(latent_dim // 2))

        # Constants
        num_hiddens = 128

        # prediction encoder:
        self._convlstm = ConvLSTMCell(input_dim=self.input_channels,
                                      hidden_dim=num_hiddens // 4,
                                      kernel_size=(3, 3),
                                      bias=True)

        # UNet-based noise predictor and decoder
        # self._noise_predictor = UNet(input_channels=(num_hiddens // 4) + 1, output_channels=num_hiddens // 4)
        # self._decoder = UNet(input_channels=num_hiddens // 4, output_channels=self.output_channels)
        self._noise_predictor = MediumUNet(input_channels=(num_hiddens // 4) + 1, output_channels=num_hiddens // 4)
        self._decoder = MediumUNet(input_channels=num_hiddens // 4, output_channels=self.output_channels)

    def forward(self, x, t, noise=None):
        """
        Forward pass input_img through the network
        """
        # reconstruction:
        # encode:
        # input reshape:
        x = x.reshape(-1, SEQ_LEN, 1, IMG_SIZE, IMG_SIZE)
        # find size of different input dimensions
        b, seq_len, c, h, w = x.size()
        # llc: b = batch size, seq_len = sequence length, c = channel, h = height, w = width

        # encode:
        # initialize hidden states
        h_enc, enc_state = self._convlstm.init_hidden(batch_size=b, image_size=(h, w))
        for t_step in range(seq_len):
            x_in = x[:, t_step]
            h_enc, enc_state = self._convlstm(input_tensor=x_in,
                                              cur_state=[h_enc, enc_state])
        # llc: this is output of the lstm, which is the input to the encoder
        enc_in = h_enc

        # add noise
        if noise is None:
            noise = torch.randn_like(enc_in)
        z_noisy = enc_in + noise

        # prepare time step encoding
        t = t.view(b, 1, 1, 1).repeat(1, 1, h, w)  # Repeat the time step for concatenation
        z_noisy = torch.cat([z_noisy, t], dim=1)

        # predict noise
        z_predicted = self._noise_predictor(z_noisy)

        # denoise
        z_denoised = enc_in - z_predicted

        # decode:
        prediction = self._decoder(z_denoised)
        prediction = torch.sigmoid(prediction)
        return prediction
