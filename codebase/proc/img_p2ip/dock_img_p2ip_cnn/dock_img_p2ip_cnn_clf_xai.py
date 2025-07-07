import os
import sys
from pathlib import Path

path_root = Path(__file__).parents[3]  # upto 'codebase' folder
sys.path.insert(0, str(path_root))
# print(sys.path)

from utils import dl_reproducible_result_util
import glob
import joblib
import torch
import numpy as np
import pandas as pd
from captum.attr import IntegratedGradients

from proc.img_p2ip.dock_img_p2ip_datamodule import DockImgP2ipCustomDataModule
from proc.img_p2ip.img_p2ip_cnn.img_p2ip_cnn_clf_train import ImgP2ipCnn


def preproc_test_result_before_attr_calc(root_path='./', model_path='./', docking_version='5_5'):
    print('#### inside the preproc_test_result_before_attr_calc() method - Start')
    print('\n########## docking_version: ' + str(docking_version))
    # retrieve the docking_version specific prediction result
    test_tag = model_path.split('/')[-1]
    test_result_dir = os.path.join(root_path, f'dataset/proc_data_tl_feat_to_img/img_p2ip_cnn/test_dock_{docking_version}/{test_tag}')
    spec_pred_res_df = pd.read_csv(os.path.join(test_result_dir, 'pred_res.csv'))
    # retrieve prot-prot pair id list
    test_pair_list_path = os.path.join(root_path, 'dataset/preproc_data_docking_BM_' + str(docking_version), 'derived_feat/dock', 'dock_test.tsv')
    spec_test_pairs_df = pd.read_csv(test_pair_list_path, delimiter='\t', header=None, names=['prot1_id', 'prot2_id', 'label'])
    # concatenate spec_test_pairs_df and spec_pred_res_df column-wise 
    concat_pred_res_df = pd.concat([spec_test_pairs_df, spec_pred_res_df], axis=1, join='inner')
    # check whether the columns 'label' and 'actual_res' are identical
    concat_pred_res_df['label'] = concat_pred_res_df['label'].astype(int)  # convert float to int
    same = concat_pred_res_df['label'].equals(concat_pred_res_df['actual_res'])
    if(not same):
        print("!! ERROR !! The columns 'label' and 'actual_res' are not identical for " + docking_version)
    # create an index column
    index_lst = list(range(concat_pred_res_df.shape[0]))
    concat_pred_res_df.insert(loc = 0, column = 'idx', value = index_lst)
    # save the concat_pred_res_df
    xai_result_dir = os.path.join(root_path, f'dataset/proc_data_tl_feat_to_img/img_p2ip_cnn/xai_dock_{docking_version}/{test_tag}')
    try:
        # check if the xai_result_dir already exists and if not, then create it
        if not os.path.exists(xai_result_dir):
            print("The directory: " + str(xai_result_dir) + " does not exist.. Creating it...")
            os.makedirs(xai_result_dir)
    except OSError as ex:
        errorMessage = "Creation of the directory " + xai_result_dir + " failed. Exception is: " + str(ex)
        raise Exception(errorMessage)
    else:
        print("Successfully created the directory %s " % xai_result_dir)
    concat_pred_res_file_nm_with_loc = os.path.join(xai_result_dir, f'concat_pred_res_dock_{docking_version}.csv')
    concat_pred_res_df.to_csv(concat_pred_res_file_nm_with_loc, index=False)
    print('concat_pred_res_df is saved as : ' + str(concat_pred_res_file_nm_with_loc))
    print('#### inside the preproc_test_result_before_attr_calc() method - End')


def load_final_ckpt_model(root_path='./', model_path='./', partial_model_name = 'ImgP2ipCnn'):
    print('#### inside the load_final_ckpt_model() method - Start')
    # create the final checkpoint file name with path
    final_chkpt_path = os.path.join(model_path, partial_model_name + '*.ckpt' )
    final_ckpt_file_name = glob.glob(final_chkpt_path, recursive=False)[0]
    print('final_ckpt_file_name: ' + str(final_ckpt_file_name))
    # load the model
    model = ImgP2ipCnn.load_from_checkpoint(final_ckpt_file_name)
    # # automatically restores model, epoch, step, LR schedulers, apex, etc...
    # model = ImgP2ipCnn()command
    # trainer = pl.Trainer()
    # trainer.fit(model, ckpt_path=final_ckpt_file_name)
    # Please note that, all model hyper-params like batch_size, block_name, etc. can be retrieved 
    # using model.hparams.config
    print('#### inside the load_final_ckpt_model() method - End')
    return model


def prepare_test_data(root_path='./', model=None, docking_version='5_5'):
    print('#### inside the prepare_test_data() method - Start')
    print('\n########## docking_version: ' + str(docking_version))
    test_data_module = DockImgP2ipCustomDataModule(root_path=root_path, batch_size=model.hparams.config['batch_size']
                                               , workers=os.cpu_count() - 5  # model.hparams.config['num_workers']
                                               , img_resoln=model.hparams.config['img_resoln']
                                               , spec_type='human'
                                               , docking_version=docking_version
                                               , dbl_combi_flg=model.hparams.config['dbl_combi_flg'])
    test_data_module.setup(stage='test')
    custom_test_dataset = test_data_module.test_data
    print('#### inside the prepare_test_data() method - End')
    return custom_test_dataset


def calc_attribution(root_path='./', model_path='./', partial_model_name = 'ImgP2ipCnn', docking_version='5_5', device_type='cpu'):
    print('\n #############################\n inside the calc_attribution() method - Start\n')
    print('\n########## docking_version: ' + str(docking_version))
    # load the final checkpointed model
    print('loading the final checkpointed model')
    model = load_final_ckpt_model(root_path, model_path, partial_model_name)
    # prepare the test data
    print('\n preparing the test data')
    custom_test_dataset = prepare_test_data(root_path, model, docking_version)
    # fetch the prediction result
    test_tag = model_path.split('/')[-1]
    xai_result_dir = os.path.join(root_path, f'dataset/proc_data_tl_feat_to_img/img_p2ip_cnn/xai_dock_{docking_version}/{test_tag}')
    pred_res_file_nm_with_loc = os.path.join(xai_result_dir, f'concat_pred_res_dock_{docking_version}.csv')
    pred_res_df = pd.read_csv(pred_res_file_nm_with_loc)
    # # filter pred_res_df for the correct positive predictions
    # corct_pos_pred_res_df = pred_res_df.loc[(pred_res_df['label'] == 1) & (pred_res_df['pred_res'] == 1)]
    # corct_pos_pred_res_df = corct_pos_pred_res_df.reset_index(drop=True)

    # declare the consolidated list of attribution dictionaries 
    con_attr_dict_lst = []
    # set model to eval mode for interpretation purposes
    model.eval()
    # iterate over the pred_res_df and find attribution
    for index, row in pred_res_df.iterrows():
        if(index % 50 == 0): print('\n # starting ' + str(index + 1) + 'th attribution calculation for PPI out of ' + str(pred_res_df.shape[0]))
        # convert the 'row' into a dictionary
        attr_res_dict = row.to_dict()
        idx_col_val = row['idx']
        # use idx_col_val to fetch the respective test data sample
        prot_prot_3dTensor, norm_prot_prot_3dTensor, label_tensor = custom_test_dataset.get_item_with_orig_input(idx_col_val)
        # print('after fetching the test data sample with idx_col_val = ' + str(idx_col_val))
        # attribution method -start
        # print('#############  attribution method -start')
        # push model and input to gpu
        model = model.to(torch.device(device_type))
        norm_prot_prot_3dTensor = norm_prot_prot_3dTensor.to(torch.device(device_type))
        # add the batch dimension
        input = norm_prot_prot_3dTensor.unsqueeze(0)
        input.requires_grad = True
        # set model to eval mode for interpretation purposes
        #  ## model.eval()  # decalred before the loop

        # applies integrated gradients attribution algorithm on test image. 
        # Integrated Gradients computes the integral of the gradients of the output prediction for the class with 
        # respect to the input image pixels. 
        # More details about integrated gradients can be found in the original paper: https://arxiv.org/abs/1703.01365
        ig = IntegratedGradients(model)
        model.zero_grad()
        attr_ig, delta = ig.attribute(inputs=input, target=label_tensor, baselines=input * 0, return_convergence_delta=True)
        # bring attribution result to cpu and arrange it in channel-last format
        attr_ig_arr_cpu = np.transpose(attr_ig.squeeze().cpu().detach().numpy(), (1, 2, 0))
        # print('Approximation delta: ', abs(delta))
        # store attr_ig_arr_cpu in attr_res_dict
        attr_res_dict['attr_ig'] = attr_ig_arr_cpu
        # bring un-normalized input to cpu and arrange it in channel-last format
        input_arr_cpu = np.transpose(prot_prot_3dTensor.cpu().detach().numpy(), (1, 2, 0))
        # store input_arr_cpu in attr_res_dict
        attr_res_dict['orig_input'] = input_arr_cpu
        con_attr_dict_lst.append(attr_res_dict)
        # print('#############  attribution method -end')
        # attribution method -end
    # end of for loop: for index, row in pred_res_df.iterrows():

    # # save the con_attr_dict_lst as a pkl file
    # print('Saving the con_attr_dict_lst as a pkl file ...')
    # con_attr_dict_lst_file_nm_with_loc = os.path.join(xai_result_dir, f'con_attr_dict_lst_dock_{docking_version}.pkl')
    # joblib.dump(value=con_attr_dict_lst, filename=con_attr_dict_lst_file_nm_with_loc, compress=3)
    print('#### inside the calc_attribution() method - End')
    return con_attr_dict_lst


def postproc_attr_result(root_path='./', model_path='./', docking_version='5_5', con_attr_dict_lst=None):
    print('\n #############################\n inside the postproc_attr_result() method - Start\n')
    print('\n########## docking_version: ' + str(docking_version))
    # # load species specific con_attr_dict_lst
    # print('loading species specific con_attr_dict_lst ...')
    test_tag = model_path.split('/')[-1]
    xai_result_dir = os.path.join(root_path, f'dataset/proc_data_tl_feat_to_img/img_p2ip_cnn/xai_dock_{docking_version}/{test_tag}')
    # con_attr_dict_lst_file_nm_with_loc = os.path.join(xai_result_dir, f'con_attr_dict_lst_dock_{docking_version}.pkl')
    # con_attr_dict_lst = joblib.load(con_attr_dict_lst_file_nm_with_loc)
    # print('loaded con_attr_dict_lst ...')

    # create postproc_attr_result_dir (if not exists)
    postproc_attr_result_dir = os.path.join(xai_result_dir, 'attr_postproc')
    try:
        # check if the postproc_attr_result_dir already exists and if not, then create it
        if not os.path.exists(postproc_attr_result_dir):
            print("The directory: " + str(postproc_attr_result_dir) + " does not exist.. Creating it...")
            os.makedirs(postproc_attr_result_dir)
    except OSError as ex:
        errorMessage = "Creation of the directory " + postproc_attr_result_dir + " failed. Exception is: " + str(ex)
        raise Exception(errorMessage)
    else:
        print("Successfully created the directory %s " % postproc_attr_result_dir)

    # iterate over con_attr_dict_lst and post-process each entry in the list
    # postproc_attr_dict_lst = []
    con_attr_dict_lst_len = len(con_attr_dict_lst)
    start_itr = 0  # #################### This can be adjusted for resuming an execution after an abrupt stop
    for itr in range(start_itr, con_attr_dict_lst_len, 1):
        print('\n # docking_version: ' + docking_version + ' : starting ' + str(itr) + 'th attribution post-processing for PPI out of ' + str(con_attr_dict_lst_len -1))
        attr_res_dict = con_attr_dict_lst[itr]
        postproc_res_dict = {dict_key: attr_res_dict[dict_key] for dict_key in attr_res_dict if dict_key not in ['attr_ig', 'orig_input']}
        # retrieve original input image in (H, W, C) format
        input_3d_hwc_arr = attr_res_dict['orig_input']
        # check for the valid pixel where none of R,G,B values are zero.
        height = input_3d_hwc_arr.shape[0]; width = input_3d_hwc_arr.shape[1]
        # non_zero_pxl_indx_lst_lst is a list of lists where each inner list contains the non-zero pixel index (H,W) i.e. 2 integers w.r.t. input_3d_hwc_arr
        non_zero_pxl_indx_lst_lst = []
        for h in range(height):  # iterate height-wise
            for w in range(width):  # iterate width-wise
                # check the RGB values for this pixel
                # if all the three (R,G,B) values are non-zero, then add it in the non_zero_pxl_indx_lst_lst
                if((input_3d_hwc_arr[h,w,0] != 0.00)  # R-channel value
                   and (input_3d_hwc_arr[h,w,1] != 0.00)  # G-channel value 
                   and(input_3d_hwc_arr[h,w,2] != 0.00)):  # B-channel value
                    non_zero_pxl_indx_lst_lst.append([h,w])
                # else:
                    # print('itr: ' + str(itr) + ' : zero-pixel: (H,W): ' + str(h) + ',' + str(w) + ' : input_3d_hwc_arr[h,w]: ' + str(input_3d_hwc_arr[h,w]))
            # end of inner for loop: for w in range(width):
        # end of outer loop: for h in range(height):
        # add non_zero_pxl_indx_lst_lst to postproc_res_dict
        postproc_res_dict['non_zero_pxl_indx_lst_lst'] = non_zero_pxl_indx_lst_lst

        # next retrieve the attributions corresponding to the non-zero pixels
        attr_ig_3d_arr = attr_res_dict['attr_ig']
        # non_zero_pxl_attr_lst_lst is a list of lists where each inner list contains the non-zero pixel attributions i.e. 3 floats for 3 channels
        non_zero_pxl_attr_lst_lst = []
        # non_zero_pxl_tot_attr_lst contains the total attribution for all the 3 channels together for the non-zero pixel 
        non_zero_pxl_tot_attr_lst = []
        # iterate over non_zero_pxl_indx_lst_lst and populate non_zero_pxl_attr_lst_lst and non_zero_pxl_tot_attr_lst
        for h, w in non_zero_pxl_indx_lst_lst: 
            attr_list = [attr_ig_3d_arr[h,w,0], attr_ig_3d_arr[h,w,1], attr_ig_3d_arr[h,w,2]]
            non_zero_pxl_attr_lst_lst.append(attr_list)
            tot_attr = attr_ig_3d_arr[h,w,0] + attr_ig_3d_arr[h,w,1] + attr_ig_3d_arr[h,w,2]
            non_zero_pxl_tot_attr_lst.append(tot_attr)
        # end of for loop: for h, w in non_zero_pxl_indx_lst_lst:
        # add non_zero_pxl_attr_lst_lst and non_zero_pxl_tot_attr_lst to postproc_res_dict
        postproc_res_dict['non_zero_pxl_attr_lst_lst'] = non_zero_pxl_attr_lst_lst
        postproc_res_dict['non_zero_pxl_tot_attr_lst'] = non_zero_pxl_tot_attr_lst

        # save the postproc_res_dict as a pkl file
        print('Saving the postproc_res_dict as a pkl file ...')
        postproc_res_dict_file_nm_with_loc = os.path.join(postproc_attr_result_dir, 
                                                          'idx_' + str(postproc_res_dict['idx']) + '.pkl')
        joblib.dump(value=postproc_res_dict, filename=postproc_res_dict_file_nm_with_loc, compress=3)

        # # add postproc_res_dict to postproc_attr_dict_lst
        # postproc_attr_dict_lst.append(postproc_res_dict)
    # end of for loop: for itr in range(con_attr_dict_lst_len):

    # # save the postproc_attr_dict_lst as a pkl file
    # print('Saving the postproc_attr_dict_lst as a pkl file ...')
    # postproc_attr_dict_lst_file_nm_with_loc = os.path.join(xai_result_dir, spec_type + '_postproc_attr_dict_lst.pkl')
    # joblib.dump(value=postproc_attr_dict_lst, filename=postproc_attr_dict_lst_file_nm_with_loc, compress=3)
    print('#### inside the postproc_attr_result() method - End')


if __name__ == '__main__':
    root_path = os.path.join('/scratch/pralaycs/Shubh_Working_Remote/PPI_Wkspc/PPI_Code/mat_p2ip_prj_working/')
    model_path = os.path.join(root_path, 'dataset/proc_data_tl_feat_to_img/img_p2ip_cnn/train/ResNet_manTl_r400n18D')
    partial_model_name = 'ImgP2ipCnn'
    device_type = 'cuda'

    docking_version_lst = ['5_5']  # '5_5', '5_5'
    for docking_version in docking_version_lst:
        print('\n########## docking_version: ' + str(docking_version))
        ############## preprocessing before attribution calculation 
        preproc_test_result_before_attr_calc(root_path=root_path, model_path=model_path, docking_version=docking_version)

        # ############## calculate the attribution and post-process the result
        con_attr_dict_lst = calc_attribution(root_path=root_path, model_path=model_path, partial_model_name=partial_model_name, docking_version=docking_version, device_type=device_type)
        postproc_attr_result(root_path=root_path, model_path=model_path, docking_version=docking_version, con_attr_dict_lst = con_attr_dict_lst)
        # # below call is an alternative of above call when con_attr_dict_lst is saved as pkl file in calc_attribution() method
        # # postproc_attr_result(root_path=root_path, model_path=model_path, docking_version=docking_version, con_attr_dict_lst = None)
    # end of for loop: for docking_version in docking_version_lst:
