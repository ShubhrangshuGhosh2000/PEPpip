import os
import sys
from pathlib import Path
path_root = Path(__file__).parents[3]  
sys.path.insert(0, str(path_root))
from utils import dl_reproducible_result_util
import glob
import joblib
import torch
import numpy as np
import pandas as pd
from codebase.proc.img_p2ip.dock_img_p2ip_trx_datamodule import DockImgP2ipCustomDataModule
from proc.img_p2ip.img_p2ip_trx.img_p2ip_trx_clf_train import ImgP2ipTrx
from utils import PPIPUtils, ProteinContactMapUtils


def preproc_test_result_before_attn_calc(root_path='./', model_path='./', docking_version='5_5'):
    test_tag = model_path.split('/')[-1]
    test_result_dir = os.path.join(root_path, f'dataset/proc_data_tl_feat_to_img/img_p2ip_trx/test_dock_{docking_version}/{test_tag}')
    spec_pred_res_df = pd.read_csv(os.path.join(test_result_dir, 'pred_res.csv'))
    test_pair_list_path = os.path.join(root_path, 'dataset/preproc_data_docking_BM_' + str(docking_version), 'derived_feat/dock', 'dock_test_lenLimit_400.tsv')
    spec_test_pairs_df = pd.read_csv(test_pair_list_path, delimiter='\t', header=None, names=['prot1_id', 'prot2_id', 'label'])
    concat_pred_res_df = pd.concat([spec_test_pairs_df, spec_pred_res_df], axis=1, join='inner')
    concat_pred_res_df['label'] = concat_pred_res_df['label'].astype(int)  
    same = concat_pred_res_df['label'].equals(concat_pred_res_df['actual_res'])
    if(not same):
        print("!! ERROR !! The columns 'label' and 'actual_res' are not identical for " + docking_version)
    # create an index column
    index_lst = list(range(concat_pred_res_df.shape[0]))
    concat_pred_res_df.insert(loc = 0, column = 'idx', value = index_lst)
    xai_result_dir = os.path.join(root_path, f'dataset/proc_data_tl_feat_to_img/img_p2ip_trx/xai_dock_{docking_version}/{test_tag}')
    PPIPUtils.createFolder(xai_result_dir)
    concat_pred_res_file_nm_with_loc = os.path.join(xai_result_dir, f'concat_pred_res_dock_{docking_version}.csv')
    concat_pred_res_df.to_csv(concat_pred_res_file_nm_with_loc, index=False)


def load_final_ckpt_model(root_path='./', model_path='./', partial_model_name = 'ImgP2ipTrx'):
    final_chkpt_path = os.path.join(model_path, partial_model_name + '*.ckpt' )
    final_ckpt_file_name = glob.glob(final_chkpt_path, recursive=False)[0]
    model = ImgP2ipTrx.load_from_checkpoint(final_ckpt_file_name)
    model.test_step_outputs.clear()  
    return model


def prepare_test_data(root_path='./', model=None, docking_version='5_5'):
    test_data_module = DockImgP2ipCustomDataModule(root_path=root_path, batch_size=model.hparams.config['batch_size']
                                               , workers= 2  
                                               , img_resoln=model.hparams.config['img_resoln'], spec_type='human'
                                               , docking_version=docking_version
                                               , dbl_combi_flg=model.hparams.config['dbl_combi_flg'])
    test_data_module.setup(stage='test')
    custom_test_dataset = test_data_module.test_data
    return custom_test_dataset


def calc_attention(root_path='./', model_path='./', partial_model_name = 'ImgP2ipTrx', docking_version='5_5', device_type='cpu'):
    model = load_final_ckpt_model(root_path, model_path, partial_model_name)
    custom_test_dataset = prepare_test_data(root_path, model, docking_version)
    test_tag = model_path.split('/')[-1]
    xai_result_dir = os.path.join(root_path, f'dataset/proc_data_tl_feat_to_img/img_p2ip_trx/xai_dock_{docking_version}/{test_tag}')
    pred_res_file_nm_with_loc = os.path.join(xai_result_dir, f'concat_pred_res_dock_{docking_version}.csv')
    pred_res_df = pd.read_csv(pred_res_file_nm_with_loc)
    con_attn_dict_lst = []
    model.eval()  
    for index, row in pred_res_df.iterrows():
        attn_res_dict = row.to_dict()
        idx_col_val = row['idx']
        prot_prot_3dTensor, norm_prot_prot_3dTensor, label_tensor = custom_test_dataset.get_item_with_orig_input(idx_col_val)
        patch_stat_2dArr = calc_patchwise_aa_interaction_pot_stat(root_path=root_path, model=model, docking_version=docking_version, attn_res_dict=attn_res_dict)
        patch_stat_2dTensor = torch.tensor(patch_stat_2dArr, dtype=torch.float32)
        patch_stat_2dTensor = patch_stat_2dTensor.unsqueeze(0)
        model.model.patch_stat_2dTensor = patch_stat_2dTensor  
        model = model.to(torch.device(device_type))
        model.model.patch_stat_2dTensor = model.model.patch_stat_2dTensor.to(torch.device(device_type))
        norm_prot_prot_3dTensor = norm_prot_prot_3dTensor.to(torch.device(device_type))
        input = norm_prot_prot_3dTensor.unsqueeze(0)
        input.requires_grad = True
        torch.set_grad_enabled(True)
        attn_map = None
        _, attn_map = model(input)  
        attn_res_dict['attn'] = attn_map.squeeze().detach().cpu().numpy()
        input_arr_cpu = np.transpose(prot_prot_3dTensor.cpu().numpy(), (1, 2, 0))
        attn_res_dict['orig_input'] = input_arr_cpu
        con_attn_dict_lst.append(attn_res_dict)
    return con_attn_dict_lst


def postproc_attn_result(root_path='./', model_path='./', docking_version='5_5', con_attn_dict_lst=None):
    test_tag = model_path.split('/')[-1]
    xai_result_dir = os.path.join(root_path, f'dataset/proc_data_tl_feat_to_img/img_p2ip_trx/xai_dock_{docking_version}/{test_tag}')
    gt_contact_map_dir = os.path.join(root_path, f"dataset/preproc_data_docking_BM_{docking_version}/prot_gt_contact_map")
    postproc_attn_result_dir = os.path.join(xai_result_dir, 'attn_postproc')
    PPIPUtils.createFolder(postproc_attn_result_dir)
    con_attn_dict_lst_len = len(con_attn_dict_lst)
    start_itr = 0  
    for itr in range(start_itr, con_attn_dict_lst_len, 1):
        attn_res_dict = con_attn_dict_lst[itr]
        postproc_attn_res_dict = {dict_key: attn_res_dict[dict_key] for dict_key in attn_res_dict if dict_key not in ['orig_input']}
        input_3d_hwc_arr = attn_res_dict['orig_input']
        prot_1_id, prot_2_id = attn_res_dict['prot1_id'], attn_res_dict['prot2_id']
        protein_name, chain_1_name = prot_1_id.split('_')
        protein_name, chain_2_name = prot_2_id.split('_')
        gt_contact_map_location = os.path.join(gt_contact_map_dir, f"contact_map_{protein_name}_{chain_1_name}_{chain_2_name}.pkl")
        gt_contact_map = joblib.load(gt_contact_map_location)
        height = gt_contact_map.shape[0]; width = gt_contact_map.shape[1]
        non_zero_pxl_indx_lst_lst = []
        for h in range(height):  
            for w in range(width):  
                non_zero_pxl_indx_lst_lst.append([h,w])
        postproc_attn_res_dict['non_zero_pxl_indx_lst_lst'] = non_zero_pxl_indx_lst_lst
        postproc_attn_res_dict_file_nm_with_loc = os.path.join(postproc_attn_result_dir, 'idx_' + str(postproc_attn_res_dict['idx']) + '.pkl')
        joblib.dump(value=postproc_attn_res_dict, filename=postproc_attn_res_dict_file_nm_with_loc, compress=3)


def calc_patchwise_aa_interaction_pot_stat(root_path='./', model=None, docking_version='5_5', attn_res_dict=None):
    patch_stat_2dArr = None
    pdb_file_location = os.path.join(root_path, f'dataset/preproc_data_docking_BM_{docking_version}/pdb_files')
    prot_1_id, prot_2_id = attn_res_dict['prot1_id'], attn_res_dict['prot2_id']
    protein_name, chain_1_name = prot_1_id.split('_')
    protein_name, chain_2_name = prot_2_id.split('_')
    prot_contact_map_processor = ProteinContactMapUtils.ProteinContactMapProcessor(pdb_file_location=pdb_file_location, protein_name=protein_name
                                                                                    , chain_1_name=chain_1_name, chain_2_name=chain_2_name)
    aaindex_potential_orig_2darr = prot_contact_map_processor.aaindex_potential  
    img_resoln=model.hparams.config['img_resoln']  
    aaindex_potential_2darr = np.zeros((img_resoln, img_resoln))
    aaindex_potential_2darr[:aaindex_potential_orig_2darr.shape[0], :aaindex_potential_orig_2darr.shape[1]] = aaindex_potential_orig_2darr
    patch_size = model.hparams.config['patch_size']  
    num_patches_per_dim = img_resoln // patch_size  
    total_patches = num_patches_per_dim * num_patches_per_dim  
    patch_stat_2dArr = np.zeros((total_patches, 2))
    stride = patch_size  
    patch_idx = 0
    for i in range(0, img_resoln, stride):
        for j in range(0, img_resoln, stride):
            window = aaindex_potential_2darr[i:i+patch_size, j:j+patch_size]
            mean = np.mean(window)
            std = np.std(window)
            patch_stat_2dArr[patch_idx, 0] = mean
            patch_stat_2dArr[patch_idx, 1] = std
            patch_idx += 1
    return patch_stat_2dArr


if __name__ == '__main__':
    root_path = os.path.join('/project/root/directory/path/here')
    model_path = os.path.join(root_path, 'dataset/proc_data_tl_feat_to_img/img_p2ip_trx/train/CDAMViT_tlStructEsmc_r400p16')
    partial_model_name = 'ImgP2ipTrx'
    device_type = 'cuda'
    docking_version_lst = ['5_5']  
    for docking_version in docking_version_lst:
        preproc_test_result_before_attn_calc(root_path=root_path, model_path=model_path, docking_version=docking_version)
        con_attn_dict_lst = calc_attention(root_path=root_path, model_path=model_path, partial_model_name=partial_model_name, docking_version=docking_version, device_type=device_type)
        postproc_attn_result(root_path=root_path, model_path=model_path, docking_version=docking_version, con_attn_dict_lst = con_attn_dict_lst)
