"""
Codes for our ICCV 2021 paper: Learning to Reduce Defocus Blur by Realistically
Modeling Dual-Pixel Data.
GitHub: https://github.com/Abdullah-Abuolaim/recurrent-defocus-deblurring-synth-dual-pixel

This module has all the functions used for data manipulation and generation,
scheduling the learning rate as well.

Copyright (c) 2021-present, Abdullah Abuolaim
This source code is licensed under the license found in the LICENSE file in
the root directory of this source tree.

This code imports the modules and starts the implementation based on the
configurations in config.py module.

Note: this code is adapted from the code of "Defocus Deblurring Using Dual-
Pixel Data" paper. Link to GitHub repository:
https://github.com/Abdullah-Abuolaim/defocus-deblurring-dual-pixel

Email: abuolaim@eecs.yorku.ca | abdullah.abuolaim@gmail.com
"""

from config import *
from metrics import *

def filter_shapness_measure(img_,kernelSize):
    convX = cv2.Sobel(img_,cv2.CV_64F,1,0,ksize=kernelSize)
    convY = cv2.Sobel(img_,cv2.CV_64F,0,1,ksize=kernelSize)
    tempArrX=convX*convX
    tempArrY=convY*convY
    tempSumXY=tempArrX+tempArrY
    tempSumXY=np.sqrt(tempSumXY)
    return np.sum(tempSumXY)

def check_dir(_path):
    if not os.path.exists(_path):
        try:
            os.makedirs(_path)
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
                
def schedule_learning_rate(epoch):
    lr=lr_[int(epoch/scheduling_rate)]
    return lr

def data_random_shuffling(temp_type):
    global train_set, val_set, test_set, train_set_vid, val_set_vid, test_set_vid, train_set_vid_fr, val_set_vid_fr, test_set_vid_fr
    
    """Get paths to all images and shuffle"""
    images_C_src = [path_read + 'dd_dp_dataset_canon/' + temp_type + '_c/' + sub_folder[0] + f for f
                    in os.listdir(path_read + 'dd_dp_dataset_canon/' + temp_type + '_c/' + sub_folder[0])
                    if f.endswith(('.jpg','.JPG', '.png', '.PNG', '.TIF'))]
    images_C_src.sort()
    
    images_C_trg = [path_read + 'dd_dp_dataset_canon/' + temp_type + '_c/' + sub_folder[1] + f for f
                    in os.listdir(path_read + 'dd_dp_dataset_canon/' + temp_type + '_c/' + sub_folder[1])
                    if f.endswith(('.jpg','.JPG', '.png', '.PNG', '.TIF'))]
    images_C_trg.sort()
    
    images_L_src = [path_read + 'dd_dp_dataset_canon/' + temp_type + '_l/' + sub_folder[0] + f for f
                    in os.listdir(path_read + 'dd_dp_dataset_canon/' + temp_type + '_l/' + sub_folder[0])
                    if f.endswith(('.jpg','.JPG', '.png', '.PNG', '.TIF'))]
    images_L_src.sort()
    
    images_R_src = [path_read + 'dd_dp_dataset_canon/' + temp_type + '_r/' + sub_folder[0] + f for f
                    in os.listdir(path_read + 'dd_dp_dataset_canon/' + temp_type + '_r/'  + sub_folder[0])
                    if f.endswith(('.jpg','.JPG', '.png', '.PNG', '.TIF'))]
    images_R_src.sort()
    
    len_imgs_list=len(images_C_src)
    
    if temp_type != 'test':
        # generate random shuffle index list for all list
        tempInd=np.arange(len_imgs_list)
        random.shuffle(tempInd)
        
        images_C_src=np.asarray(images_C_src)[tempInd]
        images_C_trg=np.asarray(images_C_trg)[tempInd]
        
        images_L_src=np.asarray(images_L_src)[tempInd]
        images_R_src=np.asarray(images_R_src)[tempInd]

    for i in range(len_imgs_list):
        if temp_type =='train':
            train_set.append([images_C_src[i],images_L_src[i],images_R_src[i],
                              images_C_trg[i]])
        elif temp_type =='val':
            val_set.append([images_C_src[i],images_L_src[i],images_R_src[i],
                            images_C_trg[i]])
        elif temp_type =='test':
            test_set.append([images_C_src[i],images_L_src[i],images_R_src[i],
                             images_C_trg[i]])
        else:
            raise NotImplementedError
    
    """Get paths to all video sequences and shuffle"""
    videos_C_src = [ sub_dir for sub_dir in os.listdir(path_read + 'dd_dp_dataset_synth_vid/' + temp_type + '_c/' + sub_folder[0])
                    if os.path.isdir(os.path.join(path_read + 'dd_dp_dataset_synth_vid/' + temp_type + '_c/' + sub_folder[0], sub_dir))]
    videos_C_src.sort()
    
    len_vids_list=len(videos_C_src)
    
    if temp_type != 'test':
        # generate random shuffle index list for all list
        tempInd=np.arange(len_vids_list)
        random.shuffle(tempInd)
        
        videos_C_src=np.asarray(videos_C_src)[tempInd]
    
    for i in range(len_vids_list):
        fr_names_list= [ fr_name for fr_name in os.listdir(path_read + 'dd_dp_dataset_synth_vid/' + temp_type + '_c/' + sub_folder[0] + videos_C_src[i] + '/')
                    if fr_name.endswith(('.jpg','.JPG', '.png', '.PNG', '.TIF'))]
        fr_names_list.sort()
        if temp_type =='train':
            train_set_vid.append(videos_C_src[i])
            train_set_vid_fr.append(fr_names_list)
        elif temp_type =='val':
            val_set_vid.append(videos_C_src[i])
            val_set_vid_fr.append(fr_names_list)
        elif temp_type =='test':
            test_set_vid.append(videos_C_src[i])
            test_set_vid_fr.append(fr_names_list)
        else:
            raise NotImplementedError

def test_generator_image(num_image):
    inImgTst = np.zeros((num_image, 1, img_h_real, img_w_real, nb_ch_all))
    outImgGt = np.zeros((num_image, 1, img_h_real, img_w_real, nb_ch_out))
    for i in range(num_image):
        print('Read image: ',i,num_image)
        if resize_flag:
            temp_img_1=cv2.imread(test_set[i][1],color_flag)
            size_set_img.append([temp_img_1.shape[1],temp_img_1.shape[0]])
            inImgTst[i, :,:,:,0:3]=cv2.resize((temp_img_1-src_mean)/norm_val,(img_w_real,img_h_real))
            inImgTst[i, :,:,:,3:6]=cv2.resize((cv2.imread(test_set[i][2],color_flag)-src_mean)/norm_val,(img_w_real,img_h_real))
            if radial_mask:
                inImgTst[i, :,:,:,6:7] = (cv2.resize(img_real_rd_mask,(img_w_real,img_h_real))).reshape((1,img_h_real,img_w_real,1))
            outImgGt[i, :]=(cv2.resize((cv2.imread(test_set[i][3],color_flag_out)-src_mean)/norm_val,(img_w_real,img_h_real))).reshape((img_h_real,img_w_real,nb_ch_out))
        else:
            inImgTst[i, :,:,:,0:3]=(cv2.imread(test_set[i][1],color_flag)-src_mean)/norm_val
            inImgTst[i, :,:,:,3:6]=(cv2.imread(test_set[i][2],color_flag)-src_mean)/norm_val
            if radial_mask:
                inImgTst[i, :,:,:,6:7] = img_real_rd_mask.reshape((1,img_real_rd_mask.shape[0],img_real_rd_mask.shape[1],1))
            outImgGt[i, :]=((cv2.imread(test_set[i][3],color_flag_out)-src_mean)/norm_val).reshape((img_h_real,img_w_real,nb_ch_out))
    return inImgTst, outImgGt

def test_generator_video(num_clips=5, test_samples=2, num_frames_=4): #num_clips=2, test_samples=1
    inImgTst = np.zeros((num_clips*test_samples, num_frames_, img_h_synth+8, img_w_synth, nb_ch_all))
    outImgGt = np.zeros((num_clips*test_samples, num_frames_, img_h_synth+8, img_w_synth, nb_ch_out))
    
    first_frame_list=np.load('vid_test_samples_list.npy')
    for _clip in range(num_clips*test_samples):        
        img_data_src_l_path = path_read + 'dd_dp_dataset_synth_vid/test_l/' + sub_folder[0] + test_set_vid[_clip//test_samples] + '/'
        img_data_src_r_path = path_read + 'dd_dp_dataset_synth_vid/test_r/' + sub_folder[0] + test_set_vid[_clip//test_samples] + '/'                   
        img_data_trg_path = path_read + 'dd_dp_dataset_synth_vid/test_c/' + sub_folder_vid[1] + test_set_vid[_clip//test_samples] + '/' 
        first_fr=first_frame_list[_clip]
        for fr in range(num_frames_):
            print('Read seq: '+test_set_vid[_clip//test_samples]+'    clip: '+str(_clip%test_samples)+'    first fr: '+str(first_fr))
            if resize_flag:
                temp_img_1=cv2.imread(img_data_src_l_path+test_set_vid_fr[_clip//test_samples][first_fr+fr],color_flag)
                size_set_vid.append([temp_img_1.shape[1],temp_img_1.shape[0]])
                inImgTst[_clip, fr,:,:,0:3]=cv2.resize((temp_img_1-src_mean)/norm_val,(img_w_synth,img_h_synth))
                inImgTst[_clip, fr,:,:,3:6]=cv2.resize((cv2.imread(img_data_src_r_path+test_set_vid_fr[_clip//test_samples][first_fr+fr],color_flag)-src_mean)/norm_val,(img_w_synth,img_h_synth))
                if radial_mask:
                        inImgTst[_clip, fr,:,:,6:7] = cv2.resize(img_synth_rd_mask,(img_w_synth,img_h_synth))
                outImgGt[_clip, fr, :]=cv2.resize((cv2.imread(img_data_trg_path+test_set_vid_fr[_clip//test_samples][first_fr+fr],color_flag_out)-trg_mean)/norm_val,(img_w_synth,img_h_synth))
                vid_padding.append([0,0,0,0])
            else:
                fr_l=(cv2.imread(img_data_src_l_path+test_set_vid_fr[_clip//test_samples][first_fr+fr],color_flag)-src_mean)/norm_val
                fr_r=(cv2.imread(img_data_src_r_path+test_set_vid_fr[_clip//test_samples][first_fr+fr],color_flag)-src_mean)/norm_val
                fr_g=(cv2.imread(img_data_trg_path+test_set_vid_fr[_clip//test_samples][first_fr+fr],color_flag_out)-trg_mean)/norm_val
                
                if img_w_synth % 2**nb_max_pooling_layers != 0 or img_h_synth % 2**nb_max_pooling_layers != 0:
                    # Check width
                    if img_w_synth % 2**nb_max_pooling_layers != 0:
                        padding_w=2**nb_max_pooling_layers-(img_w_synth % 2**nb_max_pooling_layers)
                        padding_l, padding_r = padding_w//2, padding_w//2
                        if padding_w % 2 != 0:
                            padding_l+=1
                    else:
                        padding_l, padding_r = 0, 0
                        
                    # Check height
                    if img_h_synth % 2**nb_max_pooling_layers != 0:
                        padding_h=2**nb_max_pooling_layers-(img_h_synth % 2**nb_max_pooling_layers)
                        padding_t, padding_b = padding_h//2, padding_h//2
                        if padding_h % 2 != 0:
                            padding_t+=1
                    else:
                        padding_t, padding_b = 0, 0
                    # print('W: ',padding_l,padding_r)
                    # print('H: ',padding_t,padding_b)
                    inImgTst[_clip, fr,:,:,0:3] = cv2.copyMakeBorder(fr_l, padding_t, padding_b, padding_l, padding_r, cv2.BORDER_REFLECT)
                    inImgTst[_clip, fr,:,:,3:6] = cv2.copyMakeBorder(fr_r, padding_t, padding_b, padding_l, padding_r, cv2.BORDER_REFLECT)
                    if radial_mask:
                        inImgTst[_clip, fr,:,:,6:7] = cv2.copyMakeBorder(img_synth_rd_mask, padding_t, padding_b, padding_l, padding_r, cv2.BORDER_REFLECT).reshape((img_h_synth+8,img_w_synth,1))
                    outImgGt[_clip, fr, :] = cv2.copyMakeBorder(fr_g, padding_t, padding_b, padding_l, padding_r, cv2.BORDER_REFLECT)
                    vid_padding.append([padding_t, padding_b, padding_l, padding_r])
                else:
                    inImgTst[_clip, fr,:,:,0:3] = fr_l
                    inImgTst[_clip, fr,:,:,3:6] = fr_r
                    if radial_mask:
                        inImgTst[_clip, fr,:,:,6:7] = img_synth_rd_mask.reshape((img_h_synth,img_w_synth,1))
                    outImgGt[_clip, fr, :] = fr_g
                    vid_padding.append([0,0,0,0])
                
    return inImgTst, outImgGt

def generator(phase_gen='train'):
    if phase_gen == 'train':       
        data_set_temp=train_set
        data_set_temp_vid=train_set_vid
        data_set_temp_vid_fr=train_set_vid_fr
        nb_total=total_nb_train
        nb_total_vid=total_nb_train_vid
    elif phase_gen == 'val':
        data_set_temp=val_set
        data_set_temp_vid=val_set_vid
        data_set_temp_vid_fr=val_set_vid_fr
        nb_total=total_nb_val
        nb_total_vid=total_nb_val_vid
    else:
        raise NotImplementedError
        
    img_count = 0
    vid_count = 0
    iter_loop = 0
    
    while True:
        if synth_only:
            iter_flag=True
        else:
            iter_flag= iter_loop % img_iter_rate
        if iter_flag:
            src_ims = np.zeros((vid_mini_b, num_frames, patch_h, patch_w, nb_ch_all))
            trg_ims = np.zeros((vid_mini_b, num_frames, patch_h_out, patch_w_out, nb_ch_out))
            
            vid_seq_num = data_set_temp_vid[vid_count % nb_total_vid]
            vid_seq_num_fr = data_set_temp_vid_fr[vid_count % nb_total_vid]
            
            img_data_src_l_path = path_read + 'dd_dp_dataset_synth_vid/' + phase_gen + '_l/' + sub_folder[0] + vid_seq_num + '/'
            img_data_src_r_path = path_read + 'dd_dp_dataset_synth_vid/' + phase_gen + '_r/' + sub_folder[0] + vid_seq_num + '/'                   
            img_data_trg_path = path_read + 'dd_dp_dataset_synth_vid/' + phase_gen + '_c/' + sub_folder_vid[1] + vid_seq_num + '/' 
            
            first_fr=random.randint(0,len(vid_seq_num_fr)-num_frames)
            
######################################################Rmove patches with lowest sharpness
            patch_sh=[]
            
            pts_patch_sp=[]
            std_list=[]
            
            if filter_patch:
                temp_vid_mini_b = vid_mini_b + filter_num
            else:
                temp_vid_mini_b = vid_mini_b
                
            for _pts in range(0, temp_vid_mini_b):
                s_p=[random.randint(0,img_h_synth-patch_h),random.randint(0,img_w_synth-patch_w)]
                e_p=[s_p[0]+patch_h,s_p[1]+patch_w]
                
                if filter_patch:
                    test_filt_img=((cv2.imread(img_data_trg_path+vid_seq_num_fr[first_fr],0)-trg_mean)/norm_val)[s_p[0]:e_p[0],s_p[1]:e_p[1]]
                    patch_sh.append(filter_shapness_measure(test_filt_img,5))
                
                pts_patch_sp.append(s_p)
                if add_noise:
                    std_list.append(gauss_noise_sigma[random.randint(0,len(gauss_noise_sigma)-1)])
            
            if filter_patch:
                for _pts in range(0, filter_num):
                    rmv_ind=np.argmin(patch_sh)
                    patch_sh.pop(rmv_ind)
                    pts_patch_sp.pop(rmv_ind)
                    if add_noise:
                        std_list.pop(rmv_ind)
######################################################
                
            for fr in range(num_frames):
                if resize_flag:
                    fr_l=cv2.resize((cv2.imread(img_data_src_l_path+vid_seq_num_fr[first_fr+fr],color_flag)-src_mean)/norm_val,(img_w_synth,img_h_synth))
                    fr_r=cv2.resize((cv2.imread(img_data_src_r_path+vid_seq_num_fr[first_fr+fr],color_flag)-src_mean)/norm_val,(img_w_synth,img_h_synth))
                    fr_g=cv2.resize((cv2.imread(img_data_trg_path+vid_seq_num_fr[first_fr+fr],color_flag_out)-trg_mean)/norm_val,(img_w_synth,img_h_synth))
                else:
                    fr_l=(cv2.imread(img_data_src_l_path+vid_seq_num_fr[first_fr+fr],color_flag)-src_mean)/norm_val
                    fr_r=(cv2.imread(img_data_src_r_path+vid_seq_num_fr[first_fr+fr],color_flag)-src_mean)/norm_val
                    fr_g=(cv2.imread(img_data_trg_path+vid_seq_num_fr[first_fr+fr],color_flag_out)-trg_mean)/norm_val
                for i in range(0, vid_mini_b):
                    s_p=pts_patch_sp[i]
                    e_p=[s_p[0]+patch_h,s_p[1]+patch_w]
                    src_ims[i, fr,:,:,0:3] = fr_l[s_p[0]:e_p[0],s_p[1]:e_p[1],:].reshape((patch_h, patch_w,nb_ch))
                    src_ims[i, fr,:,:,3:6] = fr_r[s_p[0]:e_p[0],s_p[1]:e_p[1],:].reshape((patch_h, patch_w,nb_ch))
                    if radial_mask:
                        src_ims[i, fr,:,:,6:7] = img_synth_rd_mask[s_p[0]:e_p[0],s_p[1]:e_p[1]].reshape((patch_h, patch_w,1))

                    trg_ims[i, fr, :] = fr_g[s_p[0]:e_p[0],s_p[1]:e_p[1]].reshape((patch_h_out, patch_w_out,nb_ch_out))
                        
                    if add_noise:
                        gauss_noise_l = np.zeros((patch_h,patch_w,nb_ch))
                        cv2.randn(gauss_noise_l, 0, std_list[i])
                        
                        gauss_noise_r = np.zeros((patch_h,patch_w,nb_ch))
                        cv2.randn(gauss_noise_r, 0, std_list[i])
                        
                        gauss_noise_c=(gauss_noise_l+gauss_noise_r)/2
                        
                        noise_fr_l = fr_l[s_p[0]:e_p[0],s_p[1]:e_p[1],:]
                        noise_fr_l = noise_fr_l + noise_fr_l * gauss_noise_l
                        noise_fr_l = np.clip(noise_fr_l, 0, 1)
                        src_ims[i, fr,:,:,0:3] = noise_fr_l.reshape((patch_h, patch_w,nb_ch))
                        
                        noise_fr_r=fr_r[s_p[0]:e_p[0],s_p[1]:e_p[1],:]
                        noise_fr_r=noise_fr_r + noise_fr_r * gauss_noise_r
                        noise_fr_r = np.clip(noise_fr_r, 0, 1)
                        src_ims[i, fr,:,:,3:6] = noise_fr_r.reshape((patch_h, patch_w,nb_ch))

                            
                        # noise_fr_c = fr_g[s_p[0]:e_p[0],s_p[1]:e_p[1],:]
                        # noise_fr_c = noise_fr_c + noise_fr_c * gauss_noise_c
                        # noise_fr_c = np.clip(noise_fr_c, 0, 1)
                        # trg_ims[i, fr, :] = noise_fr_c.reshape((patch_h_out, patch_w_out,nb_ch_out))

            yield (src_ims,trg_ims)
            vid_count += 1
            iter_loop += 1
            
        else:
            src_ims = np.zeros((img_mini_b, 1, patch_h, patch_w, nb_ch_all))
            trg_ims = np.zeros((img_mini_b, 1, patch_h_out, patch_w_out, nb_ch_out))
            
            img_data_src_l = data_set_temp[img_count % nb_total][1]
            img_data_src_r = data_set_temp[img_count % nb_total][2]
            img_data_trg = data_set_temp[img_count % nb_total][3]
            
            if resize_flag:
                img_iter_l=cv2.resize((cv2.imread(img_data_src_l,color_flag)-src_mean)/norm_val,(img_w_real,img_h_real))
                img_iter_r=cv2.resize((cv2.imread(img_data_src_r,color_flag)-src_mean)/norm_val,(img_w_real,img_h_real))
                img_iter_g=cv2.resize((cv2.imread(img_data_trg,color_flag_out)-src_mean)/norm_val,(img_w_real,img_h_real))
            else:
                img_iter_l=(cv2.imread(img_data_src_l,color_flag)-src_mean)/norm_val
                img_iter_r=(cv2.imread(img_data_src_r,color_flag)-src_mean)/norm_val
                img_iter_g=(cv2.imread(img_data_trg,color_flag_out)-src_mean)/norm_val

######################################################Rmove patches with lowest sharpness
            patch_sh=[]
            pts_patch_sp=[]
            if filter_patch:
                temp_img_mini_b = img_mini_b + filter_num
            else:
                temp_img_mini_b = img_mini_b
            
            for _pts in range(0, temp_img_mini_b):
                s_p=[random.randint(0,img_h_real-patch_h),random.randint(0,img_w_real-patch_w)]
                e_p=[s_p[0]+patch_h,s_p[1]+patch_w]
                
                if filter_patch:
                    test_filt_img=img_iter_g[s_p[0]:e_p[0],s_p[1]:e_p[1],1]
                    patch_sh.append(filter_shapness_measure(test_filt_img,5))
                
                pts_patch_sp.append(s_p)
                
            if filter_patch:
                for _pts in range(0, filter_num):
                    rmv_ind=np.argmin(patch_sh)
                    patch_sh.pop(rmv_ind)
                    pts_patch_sp.pop(rmv_ind)
######################################################

            for i in range(0, img_mini_b):
                s_p=pts_patch_sp[i]
                e_p=[s_p[0]+patch_h,s_p[1]+patch_w]
                src_ims[i, :,:,:,0:3] = img_iter_l[s_p[0]:e_p[0],s_p[1]:e_p[1],:].reshape((patch_h, patch_w,nb_ch))
                src_ims[i, :,:,:,3:6] = img_iter_r[s_p[0]:e_p[0],s_p[1]:e_p[1],:].reshape((patch_h, patch_w,nb_ch))
                if radial_mask:
                    src_ims[i, :,:,:,6:7] = img_real_rd_mask[s_p[0]:e_p[0],s_p[1]:e_p[1]].reshape((patch_h, patch_w,1))
                trg_ims[i, :] = img_iter_g[s_p[0]:e_p[0],s_p[1]:e_p[1]].reshape((patch_h_out, patch_w_out,nb_ch_out))

            yield (src_ims,trg_ims)
            img_count += 1
            iter_loop += 1

def save_eval_predictions_image(path_to_save,test_imgaes,predictions,gt_images):
    global mse_list, psnr_list, ssim_list, mae_list, test_set
    for i in range(len(test_imgaes)):
        if not depth_estimation:
            mse, psnr, ssim = MSE_PSNR_SSIM((gt_images[i,0]).astype(np.float64), (predictions[i,0]).astype(np.float64))
            mae = MAE((gt_images[i,0]).astype(np.float64), (predictions[i,0]).astype(np.float64))
            mse_list.append(mse)
            psnr_list.append(psnr)
            ssim_list.append(ssim)
            mae_list.append(mae)

        # temp_in_img=cv2.imread(test_set[i][0],color_flag)
        if bit_depth == 8:
            temp_out_img=((predictions[i,0]*norm_val)+src_mean).astype(np.uint8)
            temp_gt_img=((gt_images[i,0]*norm_val)+src_mean).astype(np.uint8)
        elif bit_depth == 16:
            temp_out_img=((predictions[i,0]*norm_val)+src_mean).astype(np.uint16)
            temp_gt_img=((gt_images[i,0]*norm_val)+src_mean).astype(np.uint16)
        img_name=((test_set[i][0]).split('/')[-1]).split('.')[0]
        if resize_flag:
            temp_out_img=cv2.resize(temp_out_img,(size_set_img[i][0],size_set_img[i][1]))
            temp_gt_img=cv2.resize(temp_gt_img,(size_set_img[i][0],size_set_img[i][1]))
        check_dir(path_to_save+'img_'+img_camera+'/')
        # cv2.imwrite(path_to_save+'img_'+img_camera+'/'+str(img_name)+'_i.png',temp_in_img)
        cv2.imwrite(path_to_save+'img_'+img_camera+'/'+str(img_name)+'_p.png',temp_out_img)
        # cv2.imwrite(path_to_save+'img_'+img_camera+'/'+str(img_name)+'_g.png',temp_gt_img)
        print('Write image: ',i,len(test_imgaes))
    
def save_eval_predictions_video(path_to_save,test_vid_fr,predictions_vid,gt_vid_fr):
    global mse_vid_list, psnr_vid_list, ssim_vid_list, mae_vid_list, test_set
    test_seq=test_vid_fr.shape[0]
    seq_fr=test_vid_fr.shape[1]
    for i in range(test_seq):
        for fr in range(seq_fr):
            h_s, h_e, w_s, w_e = vid_padding[i*seq_fr+fr][0], gt_vid_fr[i,fr].shape[0]-vid_padding[i*seq_fr+fr][1], vid_padding[i*seq_fr+fr][2], gt_vid_fr[i,fr].shape[1]-vid_padding[i*seq_fr+fr][3]
            mse, psnr, ssim = MSE_PSNR_SSIM((gt_vid_fr[i,fr,h_s:h_e,w_s:w_e,:]).astype(np.float64), (predictions_vid[i,fr,h_s:h_e,w_s:w_e,:]).astype(np.float64))
            mae = MAE((gt_vid_fr[i,fr,h_s:h_e,w_s:w_e,:]).astype(np.float64), (predictions_vid[i,fr,h_s:h_e,w_s:w_e,:]).astype(np.float64))
            mse_vid_list.append(mse)
            psnr_vid_list.append(psnr)
            ssim_vid_list.append(ssim)
            mae_vid_list.append(mae)
            
            # img_l=test_vid_fr[i,fr,h_s:h_e,w_s:w_e,0:3]
            # img_r=test_vid_fr[i,fr,h_s:h_e,w_s:w_e,3:6]
            # img_c=(img_l+img_r)/2
            # img_l=(img_l*norm_val).astype(np.uint8)
            # img_r=(img_r*norm_val).astype(np.uint8)
            # img_c=(img_c*norm_val).astype(np.uint8)

            if bit_depth == 8:
                temp_out_img=((predictions_vid[i,fr,h_s:h_e,w_s:w_e,:]*norm_val)+src_mean).astype(np.uint8)
                temp_gt_img=((gt_vid_fr[i,fr,h_s:h_e,w_s:w_e,:]*norm_val)+src_mean).astype(np.uint8)
            elif bit_depth == 16:
                temp_out_img=((predictions_vid[i,fr,h_s:h_e,w_s:w_e,:]*norm_val)+src_mean).astype(np.uint16)
                temp_gt_img=((gt_vid_fr[i,fr,h_s:h_e,w_s:w_e,:]*norm_val)+src_mean).astype(np.uint16)
            if resize_flag:
                temp_out_img=cv2.resize(temp_out_img,(size_set_vid[i*seq_fr+fr][0],size_set_vid[i*seq_fr+fr][1]))
                temp_gt_img=cv2.resize(temp_gt_img,(size_set_vid[i*seq_fr+fr][0],size_set_vid[i*seq_fr+fr][1]))
            
            img_name='seq_'+str(i).zfill(3)+'_'+str(fr).zfill(3)
            check_dir(path_to_save+'vid/deblurred/')
            check_dir(path_to_save+'vid/gt/')
            # check_dir(path_to_save+'vid/l/')
            # check_dir(path_to_save+'vid/r/')
            # check_dir(path_to_save+'vid/c/')

            cv2.imwrite(path_to_save+'vid/deblurred/'+str(img_name)+'_p.png',temp_out_img)
            cv2.imwrite(path_to_save+'vid/gt/'+str(img_name)+'_g.png',temp_gt_img)
            # cv2.imwrite(path_to_save+'vid/l/'+str(img_name)+'_l.png',img_l)
            # cv2.imwrite(path_to_save+'vid/r/'+str(img_name)+'_r.png',img_r)
            # cv2.imwrite(path_to_save+'vid/c/'+str(img_name)+'_c.png',img_c)
            
            print('Write clip: '+str(i)+'    frame: '+str(fr))