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

from proc.img_p2ip.dock_img_p2ip_trx_datamodule_struct_esmc_v2 import DockImgP2ipCustomDataModule
from proc.img_p2ip.img_p2ip_trx.img_p2ip_trx_clf_train_struct_esmc_v2 import ImgP2ipTrx
from utils import PPIPUtils, ProteinContactMapUtils


def preproc_test_result_before_attn_calc(root_path='./', model_path='./', docking_version='4_0'):
    print('#### inside the preproc_test_result_before_attn_calc() method - Start')
    print('\n########## docking_version: ' + str(docking_version))
    # retrieve the docking_version specific prediction result
    test_tag = model_path.split('/')[-1]
    test_result_dir = os.path.join(root_path, f'dataset/proc_data_tl_feat_to_img/img_p2ip_trx/test_dock_{docking_version}/{test_tag}')
    spec_pred_res_df = pd.read_csv(os.path.join(test_result_dir, 'pred_res.csv'))
    # retrieve prot-prot pair id list
    test_pair_list_path = os.path.join(root_path, 'dataset/preproc_data_docking_BM_' + str(docking_version), 'derived_feat/dock', 'dock_test_lenLimit_400.tsv')
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
    xai_result_dir = os.path.join(root_path, f'dataset/proc_data_tl_feat_to_img/img_p2ip_trx/xai_dock_{docking_version}/{test_tag}')
    PPIPUtils.createFolder(xai_result_dir)
    concat_pred_res_file_nm_with_loc = os.path.join(xai_result_dir, f'concat_pred_res_dock_{docking_version}.csv')
    concat_pred_res_df.to_csv(concat_pred_res_file_nm_with_loc, index=False)
    print('concat_pred_res_df is saved as : ' + str(concat_pred_res_file_nm_with_loc))
    print('#### inside the preproc_test_result_before_attn_calc() method - End')


def load_final_ckpt_model(root_path='./', model_path='./', partial_model_name = 'ImgP2ipTrx'):
    print('#### inside the load_final_ckpt_model() method - Start')
    # create the final checkpoint file name with path
    final_chkpt_path = os.path.join(model_path, partial_model_name + '*.ckpt' )
    final_ckpt_file_name = glob.glob(final_chkpt_path, recursive=False)[0]
    print('final_ckpt_file_name: ' + str(final_ckpt_file_name))
    # load the model
    model = ImgP2ipTrx.load_from_checkpoint(final_ckpt_file_name)
    # # automatically restores model, epoch, step, LR schedulers, apex, etc...
    # model = ImgP2ipTrx()command
    # trainer = pl.Trainer()
    # trainer.fit(model, ckpt_path=final_ckpt_file_name)
      
    # Please note that, all model hyper-params like batch_size, block_name, etc. can be retrieved using model.hparams.config
    model.test_step_outputs.clear()  # reset the list

    print('#### inside the load_final_ckpt_model() method - End')
    return model


def prepare_test_data(root_path='./', model=None, docking_version='4_0'):
    print('#### inside the prepare_test_data() method - Start')
    print('\n########## docking_version: ' + str(docking_version))
    test_data_module = DockImgP2ipCustomDataModule(root_path=root_path, batch_size=model.hparams.config['batch_size']
                                               , workers= 2  # os.cpu_count() - 5  # model.hparams.config['num_workers']
                                               , img_resoln=model.hparams.config['img_resoln'], spec_type='human'
                                               , docking_version=docking_version
                                               , dbl_combi_flg=model.hparams.config['dbl_combi_flg'])
    test_data_module.setup(stage='test')
    custom_test_dataset = test_data_module.test_data
    print('#### inside the prepare_test_data() method - End')
    return custom_test_dataset


def calc_attention(root_path='./', model_path='./', partial_model_name = 'ImgP2ipTrx', docking_version='4_0', device_type='cpu'):
    print('\n #############################\n inside the calc_attention() method - Start\n')
    print('\n########## docking_version: ' + str(docking_version))
    # load the final checkpointed model
    print('loading the final checkpointed model')
    model = load_final_ckpt_model(root_path, model_path, partial_model_name)
    # prepare the test data
    print('\n preparing the test data')
    custom_test_dataset = prepare_test_data(root_path, model, docking_version)
    # fetch the prediction result
    test_tag = model_path.split('/')[-1]
    xai_result_dir = os.path.join(root_path, f'dataset/proc_data_tl_feat_to_img/img_p2ip_trx/xai_dock_{docking_version}/{test_tag}')
    pred_res_file_nm_with_loc = os.path.join(xai_result_dir, f'concat_pred_res_dock_{docking_version}.csv')
    pred_res_df = pd.read_csv(pred_res_file_nm_with_loc)
    # # filter pred_res_df for the correct positive predictions
    # corct_pos_pred_res_df = pred_res_df.loc[(pred_res_df['label'] == 1) & (pred_res_df['pred_res'] == 1)]
    # corct_pos_pred_res_df = corct_pos_pred_res_df.reset_index(drop=True)

    # declare the consolidated list of attention dictionaries 
    con_attn_dict_lst = []
    # set model to eval mode for interpretation purposes
    model.eval()  # self.train(False)  
    # iterate over the pred_res_df and find attention
    for index, row in pred_res_df.iterrows():
        if(index % 10 == 0): print('\n # starting ' + str(index + 1) + 'th attention calculation for PPI out of ' + str(pred_res_df.shape[0]))
        # convert the 'row' into a dictionary
        attn_res_dict = row.to_dict()
        idx_col_val = row['idx']
        # use idx_col_val to fetch the respective test data sample
        prot_prot_3dTensor, norm_prot_prot_3dTensor, label_tensor = custom_test_dataset.get_item_with_orig_input(idx_col_val)
        # print('after fetching the test data sample with idx_col_val = ' + str(idx_col_val))

        # Find patch-wise residue-residue interaction potential statistics
        patch_stat_2dArr = calc_patchwise_aa_interaction_pot_stat(root_path=root_path, model=model, docking_version=docking_version, attn_res_dict=attn_res_dict)
        patch_stat_2dTensor = torch.tensor(patch_stat_2dArr, dtype=torch.float32)
        # add the batch dimension
        patch_stat_2dTensor = patch_stat_2dTensor.unsqueeze(0)
        # Add patch_stat_2dTensor as an attribute to already loaded model.
        model.model.patch_stat_2dTensor = patch_stat_2dTensor  # The 1st model correponds to the loaded instance of ImgP2ipTrx class (as is evident from load_final_ckpt_model() method) and
        # the 2nd model (model.model) corresponds to the loaded instance of CDAMViT class as is evident from the definition of ImgP2ipTrx class.
        
        # attention method -start
        # print('#############  attention method -start')
        # push model and input to gpu
        model = model.to(torch.device(device_type))
        model.model.patch_stat_2dTensor = model.model.patch_stat_2dTensor.to(torch.device(device_type))
        norm_prot_prot_3dTensor = norm_prot_prot_3dTensor.to(torch.device(device_type))
        # add the batch dimension
        input = norm_prot_prot_3dTensor.unsqueeze(0)

        # Enable gradient tracking for CDAM generation during evaluation 
        input.requires_grad = True
        torch.set_grad_enabled(True)

        attn_map = None
        # Forward pass returns attention maps directly
        _, attn_map = model(input)  # return logits, attn_map
        # store attn_map in attn_res_dict
        attn_res_dict['attn'] = attn_map.squeeze().detach().cpu().numpy()
        # bring un-normalized input to cpu and arrange it in channel-last format
        input_arr_cpu = np.transpose(prot_prot_3dTensor.cpu().numpy(), (1, 2, 0))
        # store input_arr_cpu in attn_res_dict
        attn_res_dict['orig_input'] = input_arr_cpu
        con_attn_dict_lst.append(attn_res_dict)
        # print('#############  attention method -end')
        # attention method -end
    # end of for loop: for index, row in pred_res_df.iterrows():

    # # save the con_attn_dict_lst as a pkl file
    # print('Saving the con_attn_dict_lst as a pkl file ...')
    # con_attn_dict_lst_file_nm_with_loc = os.path.join(xai_result_dir, f'con_attn_dict_lst_dock_{docking_version}.pkl')
    # joblib.dump(value=con_attn_dict_lst, filename=con_attn_dict_lst_file_nm_with_loc, compress=3)
    print('#### inside the calc_attention() method - End')
    return con_attn_dict_lst


def postproc_attn_result(root_path='./', model_path='./', docking_version='4_0', con_attn_dict_lst=None):
    print('\n #############################\n inside the postproc_attn_result() method - Start\n')
    print('\n########## docking_version: ' + str(docking_version))
    test_tag = model_path.split('/')[-1]
    xai_result_dir = os.path.join(root_path, f'dataset/proc_data_tl_feat_to_img/img_p2ip_trx/xai_dock_{docking_version}/{test_tag}')
    gt_contact_map_dir = os.path.join(root_path, f"dataset/preproc_data_docking_BM_{docking_version}/prot_gt_contact_map")
    # con_attn_dict_lst_file_nm_with_loc = os.path.join(xai_result_dir, f'con_attn_dict_lst_dock_{docking_version}.pkl')
    # con_attn_dict_lst = joblib.load(con_attn_dict_lst_file_nm_with_loc)
    # print('loaded con_attn_dict_lst ...')

    # create postproc_attn_result_dir (if not exists)
    postproc_attn_result_dir = os.path.join(xai_result_dir, 'attn_postproc')
    PPIPUtils.createFolder(postproc_attn_result_dir)
    # iterate over con_attn_dict_lst and post-process each entry in the list
    # postproc_attn_dict_lst = []
    con_attn_dict_lst_len = len(con_attn_dict_lst)
    start_itr = 0  # #################### This can be adjusted for resuming an execution after an abrupt stop
    for itr in range(start_itr, con_attn_dict_lst_len, 1):
        print('\n # docking_version: ' + docking_version + ' : starting ' + str(itr) + 'th attention post-processing for PPI out of ' + str(con_attn_dict_lst_len -1))
        attn_res_dict = con_attn_dict_lst[itr]
        postproc_attn_res_dict = {dict_key: attn_res_dict[dict_key] for dict_key in attn_res_dict if dict_key not in ['orig_input']}
        # retrieve original input image in (H, W, C) format
        input_3d_hwc_arr = attn_res_dict['orig_input']
        # retrieve the gt_contact_map
        print('retrieving the gt_contact_map...')
        prot_1_id, prot_2_id = attn_res_dict['prot1_id'], attn_res_dict['prot2_id']
        # protein id has the format of [protein_name]_[chain_name]
        protein_name, chain_1_name = prot_1_id.split('_')
        protein_name, chain_2_name = prot_2_id.split('_')
        gt_contact_map_location = os.path.join(gt_contact_map_dir, f"contact_map_{protein_name}_{chain_1_name}_{chain_2_name}.pkl")
        gt_contact_map = joblib.load(gt_contact_map_location)
        # check for the valid pixel where none of R,G,B values are zero.
        height = gt_contact_map.shape[0]; width = gt_contact_map.shape[1]
        # non_zero_pxl_indx_lst_lst is a list of lists where each inner list contains the non-zero pixel index (H,W) i.e. 2 integers w.r.t. input_3d_hwc_arr
        non_zero_pxl_indx_lst_lst = []
        for h in range(height):  # iterate height-wise
            for w in range(width):  # iterate width-wise
                non_zero_pxl_indx_lst_lst.append([h,w])
            # end of inner for loop: for w in range(width):
        # end of outer loop: for h in range(height):
        # add non_zero_pxl_indx_lst_lst to postproc_attn_res_dict
        postproc_attn_res_dict['non_zero_pxl_indx_lst_lst'] = non_zero_pxl_indx_lst_lst

        # # next retrieve the attentions corresponding to the non-zero pixels
        # attn_3d_arr = attn_res_dict['attn']
        # # non_zero_pxl_attn_lst_lst is a list of lists where each inner list contains the non-zero pixel attentions i.e. 3 floats for 3 channels
        # non_zero_pxl_attn_lst_lst = []
        # # non_zero_pxl_tot_attn_lst contains the total attention for all the 3 channels together for the non-zero pixel 
        # non_zero_pxl_tot_attn_lst = []
        # # iterate over non_zero_pxl_indx_lst_lst and populate non_zero_pxl_attn_lst_lst and non_zero_pxl_tot_attn_lst
        # for h, w in non_zero_pxl_indx_lst_lst:
        #     attn_prot_trans = attn_3d_arr[h,w,0]; attn_prose = attn_3d_arr[h,w,1]; attn_esmc = attn_3d_arr[h,w,2]
        #     non_zero_pxl_attn_lst = [attn_prot_trans, attn_prose, attn_esmc]
        #     non_zero_pxl_attn_lst_lst.append(non_zero_pxl_attn_lst)
        #     tot_attn = attn_prot_trans + attn_prose + attn_esmc
        #     non_zero_pxl_tot_attn_lst.append(tot_attn)
        # # end of for loop: for h, w in non_zero_pxl_indx_lst_lst:
        # # add non_zero_pxl_attn_lst_lst and non_zero_pxl_tot_attn_lst to postproc_attn_res_dict
        # postproc_attn_res_dict['non_zero_pxl_attn_lst_lst'] = non_zero_pxl_attn_lst_lst
        # postproc_attn_res_dict['non_zero_pxl_tot_attn_lst'] = non_zero_pxl_tot_attn_lst

        # save the postproc_attn_res_dict as a pkl file
        print('Saving the postproc_attn_res_dict as a pkl file ...')
        postproc_attn_res_dict_file_nm_with_loc = os.path.join(postproc_attn_result_dir, 'idx_' + str(postproc_attn_res_dict['idx']) + '.pkl')
        joblib.dump(value=postproc_attn_res_dict, filename=postproc_attn_res_dict_file_nm_with_loc, compress=3)

        # # add postproc_attn_res_dict to postproc_attn_dict_lst
        # postproc_attn_dict_lst.append(postproc_attn_res_dict)
    # end of for loop: for itr in range(start_itr, con_attn_dict_lst_len, 1):

    # # save the postproc_attn_dict_lst as a pkl file
    # print('Saving the postproc_attn_dict_lst as a pkl file ...')
    # postproc_attn_dict_lst_file_nm_with_loc = os.path.join(xai_result_dir, spec_type + '_postproc_attn_dict_lst.pkl')
    # joblib.dump(value=postproc_attn_dict_lst, filename=postproc_attn_dict_lst_file_nm_with_loc, compress=3)
    print('#### inside the postproc_attn_result() method - End')


def calc_patchwise_aa_interaction_pot_stat(root_path='./', model=None, docking_version='4_0', attn_res_dict=None):
    # print('#### inside the calc_patchwise_aa_interaction_pot_stat() method - Start')
    patch_stat_2dArr = None
    # The directory for the saved PDB files
    pdb_file_location = os.path.join(root_path, f'dataset/preproc_data_docking_BM_{docking_version}/pdb_files')
    
    # Extract protein_name, chain_1_name and chain_2_name from attn_res_dict
    prot_1_id, prot_2_id = attn_res_dict['prot1_id'], attn_res_dict['prot2_id']
    # protein id has the format of [protein_name]_[chain_name]
    protein_name, chain_1_name = prot_1_id.split('_')
    protein_name, chain_2_name = prot_2_id.split('_')

    # Instantiate ProteinContactMapProcessor for the aaindex_potential calculation
    prot_contact_map_processor = ProteinContactMapUtils.ProteinContactMapProcessor(pdb_file_location=pdb_file_location, protein_name=protein_name
                                                                                    , chain_1_name=chain_1_name, chain_2_name=chain_2_name)
    aaindex_potential_orig_2darr = prot_contact_map_processor.aaindex_potential  # aaindex_potential is a 2D numpy array of contact potentials 
    # Make aaindex_potential zero-padded to match the resolution
    img_resoln=model.hparams.config['img_resoln']  # 400
    aaindex_potential_2darr = np.zeros((img_resoln, img_resoln))
    aaindex_potential_2darr[:aaindex_potential_orig_2darr.shape[0], :aaindex_potential_orig_2darr.shape[1]] = aaindex_potential_orig_2darr

    # ###### Apply sliding-window protocol on aaindex_potential_2darr to calculate meand and standard deviation of each 16 x 16 patch - Start ######
    # Calculate the number of patches
    patch_size = model.hparams.config['patch_size']  # 16
    num_patches_per_dim = img_resoln // patch_size  # 400 // 16 = 25
    total_patches = num_patches_per_dim * num_patches_per_dim  # 25 * 25 = 625

    # Initialize the output array
    patch_stat_2dArr = np.zeros((total_patches, 2))

    # Perform sliding window process
    stride = patch_size  # 16  (non-overlapping stride)
    patch_idx = 0
    for i in range(0, img_resoln, stride):
        for j in range(0, img_resoln, stride):
            # Extract the window
            window = aaindex_potential_2darr[i:i+patch_size, j:j+patch_size]
            # Calculate mean and standard deviation
            mean = np.mean(window)
            std = np.std(window)
            # Store the results
            patch_stat_2dArr[patch_idx, 0] = mean
            patch_stat_2dArr[patch_idx, 1] = std
            patch_idx += 1
        # End if inner for loop: for j in range(0, img_resoln, stride):
    # End of outer for loop: for i in range(0, img_resoln, stride):
    # ###### Apply sliding-window protocol to calculate meand and standard deviation of each 16 x 16 patch - End ######
    # print('#### inside the calc_patchwise_aa_interaction_pot_stat() method - End')
    return patch_stat_2dArr


if __name__ == '__main__':
    root_path = os.path.join('/scratch/pralaycs/Shubh_Working_Remote/PPI_Wkspc/PPI_Code/mat_p2ip_prj_working/')
    model_path = os.path.join(root_path, 'dataset/proc_data_tl_feat_to_img/img_p2ip_trx/train/CDAMViT_tlStructEsmc_r400p16')
    partial_model_name = 'ImgP2ipTrx'
    device_type = 'cuda'

    docking_version_lst = ['5_5']  # '4_0', '5_5'
    for docking_version in docking_version_lst:
        print('\n########## docking_version: ' + str(docking_version))
        ############## preprocessing before attention calculation 
        preproc_test_result_before_attn_calc(root_path=root_path, model_path=model_path, docking_version=docking_version)

        ############# calculate the attention and post-process the result
        con_attn_dict_lst = calc_attention(root_path=root_path, model_path=model_path, partial_model_name=partial_model_name, docking_version=docking_version, device_type=device_type)
        postproc_attn_result(root_path=root_path, model_path=model_path, docking_version=docking_version, con_attn_dict_lst = con_attn_dict_lst)
        # ## below call is an alternative of above call when con_attn_dict_lst is saved as pkl file in calc_attention() method
        # ## postproc_attn_result(root_path=root_path, model_path=model_path, docking_version=docking_version, con_attn_dict_lst = None)
    # end of for loop: for docking_version in docking_version_lst:
