"""
Codes for our ICCV 2021 paper: Learning to Reduce Defocus Blur by Realistically
Modeling Dual-Pixel Data.
GitHub: https://github.com/Abdullah-Abuolaim/recurrent-defocus-deblurring-synth-dual-pixel
    
This is the configuration module has all the gobal variables and basic
libraries to be shared with other modules in the same project.

Copyright (c) 2021-present, Abdullah Abuolaim
This source code is licensed under the license found in the LICENSE file in
the root directory of this source tree.

Note: this code is adapted from the code of "Defocus Deblurring Using Dual-
Pixel Data" paper. Link to GitHub repository:
https://github.com/Abdullah-Abuolaim/defocus-deblurring-dual-pixel

Email: abuolaim@eecs.yorku.ca | abdullah.abuolaim@gmail.com
"""

import numpy as np
import os
import math
import cv2
import random
from skimage.metrics import structural_similarity
from sklearn.metrics import mean_absolute_error
import argparse

# parse args
parser = argparse.ArgumentParser(description='Keras UNet-convLSTM')
parser.add_argument('--op_phase', default='train', type=str,  help='operation phase: training or testing')
parser.add_argument('--test_model', default='RDPD+', type=str,  help='test model name')
parser.add_argument('--data_dir', default='../dataset_dp_img_vid/', type=str,  help='dataset directory')
parser.add_argument('--img_mini_b', default=8, type=int, help='image mini batch size')
parser.add_argument('--vid_mini_b', default=2, type=int, help='video mini batch size')
parser.add_argument('--num_frames', default=4, type=int, help='number of video frames')
parser.add_argument('--patch_size', default=512, type=int, help='patch size')
parser.add_argument('--epoch', default=140, type=int, help='number of train epoches')
parser.add_argument('--lr', default=5e-5, type=float, help='initial learning rate')
parser.add_argument('--schedule_lr_rate', default=40, type=int, help='after how many epochs you change learning rate')
parser.add_argument('--dropout_rate', default=0.4, type=float, help='dropout rate')
parser.add_argument('--ms_edge_loss', default=False, type=bool, help='use our edge loss or not')
parser.add_argument('--ms_edge_loss_weight_x', default=0.03, type=float, help='our edge loss x weight')
parser.add_argument('--ms_edge_loss_weight_y', default=0.02, type=float, help='our edge loss y weight')
args = parser.parse_args()

op_phase = args.op_phase
test_model = args.test_model

img_mini_b = args.img_mini_b
vid_mini_b = args.vid_mini_b
num_frames = args.num_frames

patch_h, patch_w = args.patch_size, args.patch_size

nb_epoch = args.epoch
init_lr = args.lr
scheduling_rate = args.schedule_lr_rate
dropout_rate = args.dropout_rate

ms_edge_loss=args.ms_edge_loss
ms_edge_loss_weight_x=args.ms_edge_loss_weight_x
ms_edge_loss_weight_y=args.ms_edge_loss_weight_y
#########################################################################

synth_only=False

# results and model name
if synth_only:
    res_model_name='RDPD_synth_l4_d0.4'
else:
    res_model_name='RDPD+_mix_l4_d_0.4'

test_img=True
test_vid=False
#########################################################################
# READ & WRITE DATA PATHS									            #
#########################################################################
continue_checkpoint=False

add_noise=False
linear_layer=True
radial_mask=True

depth_estimation=False

filter_patch=True
filter_num=3

kernel_type='_bw_rd' # _bw  _bw_rd   _log_div   _log_sub  _e_1  _e_2

sub_folder=['source/','target/']
sub_folder_vid=['source/','target/']

dataset_name='dp_img_vid'
img_camera = 'canon'

res_model_name+=kernel_type

if depth_estimation:
    res_model_name+='_depth'
    sub_folder_vid[1]='target_depth_g/'
    
if add_noise:
    res_model_name+='_noise'
    
if linear_layer:
    res_model_name+='_linear'
    
if ms_edge_loss:
    res_model_name+='_edge'
    
if radial_mask:
    res_model_name+='_radial'

if filter_patch:
    res_model_name+='_filter'

if op_phase == 'train':
    res_model_name='defocus_deblurring_dp_'+res_model_name
elif op_phase == 'test':
    res_model_name=test_model

# path_to_model
path_to_model='./ModelCheckpoints/'+res_model_name+'.hdf5'
# paths to read datasets
path_read = args.data_dir
# path to write results
path_write='./results/res_'+res_model_name+'/'

#dd_dp_dataset_canon
#dd_dp_dataset_synth_vid
#########################################################################
# NUMBER OF IMAGES IN THE TRAINING, VALIDATION, AND TESTING SETS		#
#########################################################################
# Preclaculated values, due to Colab running issues with reading files using os
total_nb_train = len([path_read + 'dd_dp_dataset_canon/train_c/' + sub_folder[0] + f for f
                in os.listdir(path_read + 'dd_dp_dataset_canon/train_c/' + sub_folder[0])
                if f.endswith(('.jpg','.JPG', '.png', '.PNG', '.TIF'))])

total_nb_val = len([path_read + 'dd_dp_dataset_canon/val_c/' + sub_folder[0] + f for f
                in os.listdir(path_read + 'dd_dp_dataset_canon/val_c/' + sub_folder[0])
                if f.endswith(('.jpg','.JPG', '.png', '.PNG', '.TIF'))])

total_nb_test = len([path_read + 'dd_dp_dataset_canon/test_c/' + sub_folder[0] + f for f
                in os.listdir(path_read + 'dd_dp_dataset_canon/test_c/' + sub_folder[0])
                if f.endswith(('.jpg','.JPG', '.png', '.PNG', '.TIF'))])

total_nb_train_vid = len([path_read + 'dd_dp_dataset_synth_vid/train_c/' + sub_folder[0] + sub_dir
                    for sub_dir in os.listdir(path_read  + 'dd_dp_dataset_synth_vid/train_c/' + sub_folder[0])
                    if os.path.isdir(os.path.join(path_read  + 'dd_dp_dataset_synth_vid/train_c/' + sub_folder[0], sub_dir))])

total_nb_val_vid = len([path_read + 'dd_dp_dataset_synth_vid/val_c/' + sub_folder[0] + sub_dir
                    for sub_dir in os.listdir(path_read  + 'dd_dp_dataset_synth_vid/val_c/' + sub_folder[0])
                    if os.path.isdir(os.path.join(path_read  + 'dd_dp_dataset_synth_vid/val_c/' + sub_folder[0], sub_dir))])

total_nb_test_vid = len([path_read + 'dd_dp_dataset_synth_vid/test_c/' + sub_folder[0] + sub_dir
                    for sub_dir in os.listdir(path_read  + 'dd_dp_dataset_synth_vid/test_c/' + sub_folder[0])
                    if os.path.isdir(os.path.join(path_read  + 'dd_dp_dataset_synth_vid/test_c/' + sub_folder[0], sub_dir))])

#########################################################################
# MODEL PARAMETERS & TRAINING SETTINGS									#
#########################################################################
# resize flag to resize input and output images
resize_flag=False

# input image size
img_w_real=1680
img_h_real=1120

img_w_synth=1920
img_h_synth=1080

# number of patches to extarct from a single image
rand_patch_num=8

# mean value pre-claculated
src_mean=0
trg_mean=0

# output patch size
patch_w_out = patch_w
patch_h_out = patch_h

# number of input channels for the two DP views
nb_ch_all = 6
# number of input channels for single image
nb_ch=3

# number of out channels
nb_ch_out=3

nb_max_pooling_layers=4

# color flag:"1" for 3-channel 8-bit image or "0" for 1-channel 8-bit grayscale
# or "-1" to read image as it including bit depth
color_flag=1

#for output 
color_flag_out=1

bit_depth=8

norm_val=(2**bit_depth)-1

img_iter_rate = 2
# number of training image batches
nb_train = total_nb_train * img_iter_rate
# number of validation image batches
nb_val = 50

# generate learning rate array
lr_=[]

'''Remember: smaller mini-batch size means smaller learning rate'''
lr_.append(init_lr) #initial learning rate
for i in range(int(nb_epoch/scheduling_rate)):
    lr_.append(lr_[i]*0.5)

train_set, val_set, test_set =  [], [], []
train_set_vid, val_set_vid, test_set_vid = [], [], []
train_set_vid_fr, val_set_vid_fr, test_set_vid_fr = [], [], []

size_set_img, size_set_vid = [], []

vid_padding=[]

mse_list, psnr_list, ssim_list, mae_list = [], [], [], []
mse_vid_list, psnr_vid_list, ssim_vid_list, mae_vid_list = [], [], [], []


gauss_noise_sigma=np.arange(0.05,0.55,0.05)

if radial_mask:
    img_real_rd_mask=(cv2.imread('real_radial_mask.png',-1)-src_mean)/norm_val
    img_synth_rd_mask=(cv2.imread('synth_radial_mask.png',-1)-src_mean)/norm_val
    nb_ch_all = 7

if depth_estimation:
    nb_ch_out=1
    color_flag_out=0