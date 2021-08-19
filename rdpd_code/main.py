"""
Codes for our ICCV 2021 paper: Learning to Reduce Defocus Blur by Realistically
Modeling Dual-Pixel Data.
GitHub: https://github.com/Abdullah-Abuolaim/recurrent-defocus-deblurring-synth-dual-pixel

This is the main module for linking different components of the input video
frames CNN-based model proposed for the task of video defocus deblurring based
on dual-pixel data. 

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

from model import *
from config import *
from data import *
            
check_dir(path_write)

if op_phase=='train':
    data_random_shuffling('train')
    data_random_shuffling('val')
    data_random_shuffling('test')
    in_data = Input(batch_shape=(None,None, patch_h, patch_w, nb_ch_all))
    
    m = Model(inputs=in_data, outputs=unet(in_data,False))
    
    if continue_checkpoint:
        m.load_weights(path_to_model)
        
    # m.summary()
    
    if ms_edge_loss:
        m.compile(optimizer = Adam(lr = lr_[0]), loss = [custom_mse,sobel_sh_loss_3,sobel_sh_loss_7,sobel_sh_loss_11])
        print('************************Network loss: MSE, MS-Sobel************************')
    else:
        m.compile(optimizer = Adam(lr = lr_[0]), loss = custom_mse) #'mean_squared_error'
        print('************************Network loss: MSE************************')
    
    # training callbacks
    model_checkpoint = ModelCheckpoint(path_to_model, monitor='loss',
                            verbose=1, save_best_only=True)
    l_r_scheduler_callback = LearningRateScheduler(schedule=schedule_learning_rate)
    
    history = m.fit_generator(generator('train'), nb_train, nb_epoch,
                        validation_data=generator('val'),
                        validation_steps=nb_val,callbacks=[model_checkpoint,
                        l_r_scheduler_callback])
    
    np.save(path_write+'train_loss_arr',history.history['loss'])
    np.save(path_write+'val_loss_arr',history.history['val_loss'])

elif op_phase=='test':
    vid_mini_b = 1
    img_mini_b = 1
    data_random_shuffling('test')
    """
    TODO: check how to use stateful(default False). If True, the last state
    for each sample at index i in a batch will be used as initial state for
    the sample of index i in the following batch.
    """
    if test_img:
        in_data = Input(batch_shape=(None,None, img_h_real, img_w_real, nb_ch_all))
        m = Model(inputs=in_data, outputs=unet(in_data,False))
        m.load_weights(path_to_model)
        
        test_imgaes, gt_images = test_generator_image(total_nb_test)
        predictions_img = m.predict(test_imgaes,img_mini_b,verbose=1)                   
        save_eval_predictions_image(path_write,test_imgaes,predictions_img,gt_images)
        
        np.save(path_write+img_camera+'_mse_img',np.asarray(mse_list))
        np.save(path_write+img_camera+'_psnr_img',np.asarray(psnr_list))
        np.save(path_write+img_camera+'_ssim_img',np.asarray(ssim_list))
        np.save(path_write+img_camera+'_mae_img',np.asarray(mae_list))
        np.save(path_write+img_camera+'_all_img',[np.mean(np.asarray(mse_list)),
                                              np.mean(np.asarray(psnr_list)),
                                              np.mean(np.asarray(ssim_list)),
                                              np.mean(np.asarray(mae_list))])
        test_imgaes, gt_images, predictions_img =[], [], []

    if test_vid:
        in_data = Input(batch_shape=(None,None, img_h_synth+8, img_w_synth, nb_ch_all))
        m = Model(inputs=in_data, outputs=unet(in_data,False))
        m.load_weights(path_to_model)
        
        test_vid_fr, gt_vid_fr = test_generator_video()#total_nb_test_vid
        predictions_vid = m.predict(test_vid_fr,vid_mini_b,verbose=1)                   
        save_eval_predictions_video(path_write,test_vid_fr,predictions_vid,gt_vid_fr)
        
        np.save(path_write+'synth_mse_vid',np.asarray(mse_vid_list))
        np.save(path_write+'synth_psnr_vid',np.asarray(psnr_vid_list))
        np.save(path_write+'synth_ssim_vid',np.asarray(ssim_vid_list))
        np.save(path_write+'synth_mae_vid',np.asarray(mae_vid_list))
        np.save(path_write+'synth_all_vid',[np.mean(np.asarray(mse_vid_list)),
                                              np.mean(np.asarray(psnr_vid_list)),
                                              np.mean(np.asarray(ssim_vid_list)),
                                              np.mean(np.asarray(mae_vid_list))])