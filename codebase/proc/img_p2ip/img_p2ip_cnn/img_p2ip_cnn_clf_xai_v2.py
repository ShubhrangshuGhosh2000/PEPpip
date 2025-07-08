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

from proc.img_p2ip.img_p2ip_datamodule_struct_esmc_v2 import ImgP2ipCustomDataModule
from proc.img_p2ip.img_p2ip_cnn.img_p2ip_cnn_clf_train_struct_esmc_v2 import ImgP2ipCnn
from utils import PPIPUtils


def preproc_test_result_before_attr_calc(root_path='./', model_path='./', spec_type='ecoli'):
    print('#### inside the preproc_test_result_before_attr_calc() method - Start')
    print('\n########## spec_type: ' + str(spec_type))
    # retrieve the species specific prediction result
    test_tag = model_path.split('/')[-1]
    test_result_dir = os.path.join(root_path, 'dataset/proc_data_tl_feat_to_img/img_p2ip_cnn/test/dscript_other_spec_' + test_tag, spec_type)
    spec_pred_res_df = pd.read_csv(os.path.join(test_result_dir, spec_type + '_pred_res.csv'))
    # retrieve prot-prot pair id list
    test_pair_list_path = os.path.join(root_path, 'dataset/orig_data_DS/pairs', spec_type + '_test.tsv' )
    spec_test_pairs_df = pd.read_csv(test_pair_list_path, delimiter='\t', header=None, names=['prot1_id', 'prot2_id', 'label'])
    # concatenate spec_test_pairs_df and spec_pred_res_df column-wise 
    concat_pred_res_df = pd.concat([spec_test_pairs_df, spec_pred_res_df], axis=1, join='inner')
    # check whether the columns 'label' and 'actual_res' are identical
    concat_pred_res_df['label'] = concat_pred_res_df['label'].astype(int)  # convert float to int
    same = concat_pred_res_df['label'].equals(concat_pred_res_df['actual_res'])
    if(not same):
        print("!! ERROR !! The columns 'label' and 'actual_res' are not identical for " + spec_type)
    # create an index column
    index_lst = list(range(concat_pred_res_df.shape[0]))
    concat_pred_res_df.insert(loc = 0, column = 'idx', value = index_lst)
    # save the concat_pred_res_df
    xai_result_dir = os.path.join(root_path, 'dataset/proc_data_tl_feat_to_img/img_p2ip_cnn/xai/dscript_other_spec_' + test_tag, spec_type)
    PPIPUtils.createFolder(xai_result_dir)
    concat_pred_res_file_nm_with_loc = os.path.join(xai_result_dir, spec_type + '_concat_pred_res.csv')
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


def prepare_test_data(root_path='./', model=None, spec_type='ecoli'):
    print('#### inside the prepare_test_data() method - Start')
    print('\n########## spec_type: ' + str(spec_type))
    test_data_module = ImgP2ipCustomDataModule(root_path=root_path, batch_size=model.hparams.config['batch_size']
                                               , workers=os.cpu_count() - 5  # model.hparams.config['num_workers']
                                               , img_resoln=model.hparams.config['img_resoln'], spec_type=spec_type
                                               , dbl_combi_flg=model.hparams.config['dbl_combi_flg'])
    test_data_module.setup(stage='test')
    custom_test_dataset = test_data_module.test_data
    print('#### inside the prepare_test_data() method - End')
    return custom_test_dataset


def calc_attribution(root_path='./', model_path='./', partial_model_name = 'ImgP2ipCnn', spec_type='ecoli', device_type='cpu'):
    print('\n #############################\n inside the calc_attribution() method - Start\n')
    print('\n########## spec_type: ' + str(spec_type))
    # load the final checkpointed model
    print('loading the final checkpointed model')
    model = load_final_ckpt_model(root_path, model_path, partial_model_name)
    # prepare the test data
    print('\n preparing the test data')
    custom_test_dataset = prepare_test_data(root_path, model, spec_type)
    # fetch the prediction result
    test_tag = model_path.split('/')[-1]
    xai_result_dir = os.path.join(root_path, 'dataset/proc_data_tl_feat_to_img/img_p2ip_cnn/xai/dscript_other_spec_' + test_tag, spec_type)
    pred_res_file_nm_with_loc = os.path.join(xai_result_dir, spec_type + '_concat_pred_res.csv')
    pred_res_df = pd.read_csv(pred_res_file_nm_with_loc)
    # filter pred_res_df for the correct positive predictions
    corct_pos_pred_res_df = pred_res_df.loc[(pred_res_df['label'] == 1) & (pred_res_df['pred_res'] == 1)]
    corct_pos_pred_res_df = corct_pos_pred_res_df.reset_index(drop=True)

    # declare the consolidated list of attribution for the positive result dictionaries 
    con_attrForPositiveRes_dict_lst = []
    # set model to eval mode for interpretation purposes
    model.eval()
    # iterate over the corct_pos_pred_res_df and find attribution
    for index, row in corct_pos_pred_res_df.iterrows():
        if(index % 100 == 0): print('\n # starting ' + str(index + 1) + 'th attribution calculation for positive PPI out of ' + str(corct_pos_pred_res_df.shape[0]))
        # convert the 'row' into a dictionary
        attr_res_dict = row.to_dict()
        idx_col_val = row['idx']
        # use idx_col_val to fetch the respective test data sample
        prot_prot_3dTensor, norm_prot_prot_3dTensor, label_tensor, orig_ht_b4_padding, orig_width_b4_padding = custom_test_dataset.get_item_with_orig_input(idx_col_val)
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
        # Also store orig_ht_b4_padding and orig_width_b4_padding of input_arr_cpu
        attr_res_dict['orig_ht_b4_padding'] = orig_ht_b4_padding
        attr_res_dict['orig_width_b4_padding'] = orig_width_b4_padding
        con_attrForPositiveRes_dict_lst.append(attr_res_dict)
        # print('#############  attribution method -end')
        # attribution method -end
    # end of for loop: for index, row in corct_pos_pred_res_df.iterrows():

    # # save the con_attrForPositiveRes_dict_lst as a pkl file
    # print('Saving the con_attrForPositiveRes_dict_lst as a pkl file ...')
    # con_attrForPositiveRes_dict_lst_file_nm_with_loc = os.path.join(xai_result_dir, spec_type + '_con_attrForPositiveRes_dict_lst.pkl')
    # joblib.dump(value=con_attrForPositiveRes_dict_lst, filename=con_attrForPositiveRes_dict_lst_file_nm_with_loc, compress=3)
    print('#### inside the calc_attribution() method - End')
    return con_attrForPositiveRes_dict_lst


def postproc_attr_result(root_path='./', model_path='./', spec_type='ecoli', con_attrForPositiveRes_dict_lst=None):
    print('\n #############################\n inside the postproc_attr_result() method - Start\n')
    print('\n########## spec_type: ' + str(spec_type))
    # # load species specific con_attrForPositiveRes_dict_lst
    # print('loading species specific con_attrForPositiveRes_dict_lst ...')
    test_tag = model_path.split('/')[-1]
    xai_result_dir = os.path.join(root_path, 'dataset/proc_data_tl_feat_to_img/img_p2ip_cnn/xai/dscript_other_spec_' + test_tag, spec_type)
    # con_attrForPositiveRes_dict_lst_file_nm_with_loc = os.path.join(xai_result_dir, spec_type + '_con_attrForPositiveRes_dict_lst.pkl')
    # con_attrForPositiveRes_dict_lst = joblib.load(con_attrForPositiveRes_dict_lst_file_nm_with_loc)
    # print('loaded con_attrForPositiveRes_dict_lst ...')

    postproc_attr_result_dir = os.path.join(xai_result_dir, spec_type + '_attr_postproc')
    PPIPUtils.createFolder(postproc_attr_result_dir)
    # iterate over con_attrForPositiveRes_dict_lst and post-process each entry in the list
    # postproc_attrForPositiveRes_dict_lst = []
    con_attrForPositiveRes_dict_lst_len = len(con_attrForPositiveRes_dict_lst)
    start_itr = 0  # #################### This can be adjusted for resuming an execution after an abrupt stop
    for itr in range(start_itr, con_attrForPositiveRes_dict_lst_len, 1):
        print('\n # spec_type: ' + spec_type + ' : starting ' + str(itr) + 'th attribution post-processing for positive PPI out of ' + str(con_attrForPositiveRes_dict_lst_len -1))
        attr_res_dict = con_attrForPositiveRes_dict_lst[itr]
        postproc_attr_res_dict = {dict_key: attr_res_dict[dict_key] for dict_key in attr_res_dict if dict_key not in ['attr_ig', 'orig_input']}
        # retrieve original input image in (H, W, C) format
        input_3d_hwc_arr = attr_res_dict['orig_input']
        # check for the original pixels before padding using orig_ht_b4_padding and orig_width_b4_padding
        orig_ht_b4_padding = attr_res_dict['orig_ht_b4_padding']; orig_width_b4_padding = attr_res_dict['orig_width_b4_padding']
        # non_zero_pxl_indx_lst_lst is a list of lists where each inner list contains the non-zero pixel index (H,W) i.e. 2 integers w.r.t. input_3d_hwc_arr
        non_zero_pxl_indx_lst_lst = []
        for h in range(orig_ht_b4_padding):  # iterate height-wise
            for w in range(orig_width_b4_padding):  # iterate width-wise
                non_zero_pxl_indx_lst_lst.append([h,w])
            # end of inner for loop: for w in range(orig_width_b4_padding):
        # end of outer loop: for h in range(orig_ht_b4_padding):
        # add non_zero_pxl_indx_lst_lst to postproc_attr_res_dict
        postproc_attr_res_dict['non_zero_pxl_indx_lst_lst'] = non_zero_pxl_indx_lst_lst

        # next retrieve the attributions corresponding to the non-zero pixels
        attr_ig_3d_arr = attr_res_dict['attr_ig']
        # non_zero_pxl_attr_lst_lst is a list of lists where each inner list contains the non-zero pixel attributions i.e. 3 floats for 3 channels
        non_zero_pxl_attr_lst_lst = []
        # non_zero_pxl_tot_attr_lst contains the total attribution for all the 3 channels together for the non-zero pixel 
        non_zero_pxl_tot_attr_lst = []
        # iterate over non_zero_pxl_indx_lst_lst and populate non_zero_pxl_attr_lst_lst and non_zero_pxl_tot_attr_lst
        for h, w in non_zero_pxl_indx_lst_lst:
            attr_protTrans = attr_ig_3d_arr[h,w,0]; attr_prose = attr_ig_3d_arr[h,w,1]; attr_esmc = attr_ig_3d_arr[h,w,2]
            non_zero_pxl_attr_lst = [attr_protTrans, attr_prose, attr_esmc]
            non_zero_pxl_attr_lst_lst.append(non_zero_pxl_attr_lst)
            tot_attr = attr_protTrans + attr_prose + attr_esmc
            non_zero_pxl_tot_attr_lst.append(tot_attr)
        # end of for loop: for h, w in non_zero_pxl_indx_lst_lst:
        # add non_zero_pxl_attr_lst_lst and non_zero_pxl_tot_attr_lst to postproc_attr_res_dict
        postproc_attr_res_dict['non_zero_pxl_attr_lst_lst'] = non_zero_pxl_attr_lst_lst
        postproc_attr_res_dict['non_zero_pxl_tot_attr_lst'] = non_zero_pxl_tot_attr_lst

        # save the postproc_attr_res_dict as a pkl file
        print('Saving the postproc_attr_res_dict as a pkl file ...')
        postproc_attr_res_dict_file_nm_with_loc = os.path.join(postproc_attr_result_dir, 'idx_' + str(postproc_attr_res_dict['idx']) + '.pkl')
        joblib.dump(value=postproc_attr_res_dict, filename=postproc_attr_res_dict_file_nm_with_loc, compress=3)

        # # add postproc_attr_res_dict to postproc_attrForPositiveRes_dict_lst
        # postproc_attrForPositiveRes_dict_lst.append(postproc_attr_res_dict)
    # end of for loop: for itr in range(start_itr, con_attrForPositiveRes_dict_lst_len, 1):

    # # save the postproc_attrForPositiveRes_dict_lst as a pkl file
    # print('Saving the postproc_attrForPositiveRes_dict_lst as a pkl file ...')
    # postproc_attrForPositiveRes_dict_lst_file_nm_with_loc = os.path.join(xai_result_dir, spec_type + '_postproc_attr_dict_lst.pkl')
    # joblib.dump(value=postproc_attrForPositiveRes_dict_lst, filename=postproc_attrForPositiveRes_dict_lst_file_nm_with_loc, compress=3)
    print('#### inside the postproc_attr_result() method - End')



if __name__ == '__main__':
    root_path = os.path.join('/project/root/directory/path/here')
    model_path = os.path.join(root_path, 'dataset/proc_data_tl_feat_to_img/img_p2ip_cnn/train/ResNet_tlStructEsmc_r400n18DnoWS')
    partial_model_name = 'ImgP2ipCnn'
    device_type = 'cuda'

    # spec_type_lst = ['ecoli', 'fly', 'mouse', 'worm', 'yeast']  # human, ecoli, fly, mouse, worm, yeast
    spec_type_lst = ['yeast']  # human, ecoli, fly, mouse, worm, yeast
    for spec_type in spec_type_lst:
        print('\n########## spec_type: ' + str(spec_type))
        # ############## preprocessing before attribution calculation 
        preproc_test_result_before_attr_calc(root_path=root_path, model_path=model_path, spec_type=spec_type)

        # ############## calculate the attribution and post-process the result
        con_attrForPositiveRes_dict_lst = calc_attribution(root_path=root_path, model_path=model_path, partial_model_name=partial_model_name, spec_type=spec_type, device_type=device_type)
        postproc_attr_result(root_path=root_path, model_path=model_path, spec_type=spec_type, con_attrForPositiveRes_dict_lst = con_attrForPositiveRes_dict_lst)
        # below call is an alternative of above call when con_attrForPositiveRes_dict_lst is saved as pkl file in calc_attribution() method
        # postproc_attr_result(root_path=root_path, model_path=model_path, spec_type=spec_type, con_attrForPositiveRes_dict_lst = None)
    # end of for loop: for spec_type in spec_type_lst:
