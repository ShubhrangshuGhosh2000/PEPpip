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
from captum.attr import IntegratedGradients
from codebase.proc.img_p2ip.dock_img_p2ip_datamodule import DockImgP2ipCustomDataModule
from proc.img_p2ip.img_p2ip_cnn.img_p2ip_cnn_clf_train import ImgP2ipCnn
from utils import PPIPUtils


def preproc_test_result_before_attr_calc(root_path='./', model_path='./', docking_version='5_5'):
    test_tag = model_path.split('/')[-1]
    test_result_dir = os.path.join(root_path, f'dataset/proc_data_tl_feat_to_img/img_p2ip_cnn/test_dock_{docking_version}/{test_tag}')
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
    xai_result_dir = os.path.join(root_path, f'dataset/proc_data_tl_feat_to_img/img_p2ip_cnn/xai_dock_{docking_version}/{test_tag}')
    PPIPUtils.createFolder(xai_result_dir)
    concat_pred_res_file_nm_with_loc = os.path.join(xai_result_dir, f'concat_pred_res_dock_{docking_version}.csv')
    concat_pred_res_df.to_csv(concat_pred_res_file_nm_with_loc, index=False)


def load_final_ckpt_model(root_path='./', model_path='./', partial_model_name = 'ImgP2ipCnn'):
    final_chkpt_path = os.path.join(model_path, partial_model_name + '*.ckpt' )
    final_ckpt_file_name = glob.glob(final_chkpt_path, recursive=False)[0]
    model = ImgP2ipCnn.load_from_checkpoint(final_ckpt_file_name)
    return model


def prepare_test_data(root_path='./', model=None, docking_version='5_5'):
    test_data_module = DockImgP2ipCustomDataModule(root_path=root_path, batch_size=model.hparams.config['batch_size']
                                               , workers=os.cpu_count() - 5  
                                               , img_resoln=model.hparams.config['img_resoln'], spec_type='human'
                                               , docking_version=docking_version
                                               , dbl_combi_flg=model.hparams.config['dbl_combi_flg'])
    test_data_module.setup(stage='test')
    custom_test_dataset = test_data_module.test_data
    return custom_test_dataset


def calc_attribution(root_path='./', model_path='./', partial_model_name = 'ImgP2ipCnn', docking_version='5_5', device_type='cpu'):
    model = load_final_ckpt_model(root_path, model_path, partial_model_name)
    custom_test_dataset = prepare_test_data(root_path, model, docking_version)
    test_tag = model_path.split('/')[-1]
    xai_result_dir = os.path.join(root_path, f'dataset/proc_data_tl_feat_to_img/img_p2ip_cnn/xai_dock_{docking_version}/{test_tag}')
    pred_res_file_nm_with_loc = os.path.join(xai_result_dir, f'concat_pred_res_dock_{docking_version}.csv')
    pred_res_df = pd.read_csv(pred_res_file_nm_with_loc)
    con_attr_dict_lst = []
    model.eval()
    for index, row in pred_res_df.iterrows():
        attr_res_dict = row.to_dict()
        idx_col_val = row['idx']
        prot_prot_3dTensor, norm_prot_prot_3dTensor, label_tensor = custom_test_dataset.get_item_with_orig_input(idx_col_val)
        model = model.to(torch.device(device_type))
        norm_prot_prot_3dTensor = norm_prot_prot_3dTensor.to(torch.device(device_type))
        input = norm_prot_prot_3dTensor.unsqueeze(0)
        input.requires_grad = True
        ig = IntegratedGradients(model)
        model.zero_grad()
        attr_ig, delta = ig.attribute(inputs=input, target=label_tensor, baselines=input * 0, return_convergence_delta=True)
        attr_ig_arr_cpu = np.transpose(attr_ig.squeeze().detach().cpu().numpy(), (1, 2, 0))
        attr_res_dict['attr_ig'] = attr_ig_arr_cpu
        input_arr_cpu = np.transpose(prot_prot_3dTensor.detach().cpu().numpy(), (1, 2, 0))
        attr_res_dict['orig_input'] = input_arr_cpu
        con_attr_dict_lst.append(attr_res_dict)
    return con_attr_dict_lst


def postproc_attr_result(root_path='./', model_path='./', docking_version='5_5', con_attr_dict_lst=None):
    test_tag = model_path.split('/')[-1]
    xai_result_dir = os.path.join(root_path, f'dataset/proc_data_tl_feat_to_img/img_p2ip_cnn/xai_dock_{docking_version}/{test_tag}')
    gt_contact_map_dir = os.path.join(root_path, f"dataset/preproc_data_docking_BM_{docking_version}/prot_gt_contact_map")
    postproc_attr_result_dir = os.path.join(xai_result_dir, 'attr_postproc')
    PPIPUtils.createFolder(postproc_attr_result_dir)
    con_attr_dict_lst_len = len(con_attr_dict_lst)
    start_itr = 0  
    for itr in range(start_itr, con_attr_dict_lst_len, 1):
        attr_res_dict = con_attr_dict_lst[itr]
        postproc_attr_res_dict = {dict_key: attr_res_dict[dict_key] for dict_key in attr_res_dict if dict_key not in ['attr_ig', 'orig_input']}
        input_3d_hwc_arr = attr_res_dict['orig_input']
        prot_1_id, prot_2_id = attr_res_dict['prot1_id'], attr_res_dict['prot2_id']
        protein_name, chain_1_name = prot_1_id.split('_')
        protein_name, chain_2_name = prot_2_id.split('_')
        gt_contact_map_location = os.path.join(gt_contact_map_dir, f"contact_map_{protein_name}_{chain_1_name}_{chain_2_name}.pkl")
        gt_contact_map = joblib.load(gt_contact_map_location)
        height = gt_contact_map.shape[0]; width = gt_contact_map.shape[1]
        non_zero_pxl_indx_lst_lst = []
        for h in range(height):  
            for w in range(width):  
                non_zero_pxl_indx_lst_lst.append([h,w])
        postproc_attr_res_dict['non_zero_pxl_indx_lst_lst'] = non_zero_pxl_indx_lst_lst
        attr_ig_3d_arr = attr_res_dict['attr_ig']
        non_zero_pxl_attr_lst_lst = []
        non_zero_pxl_tot_attr_lst = []
        for h, w in non_zero_pxl_indx_lst_lst:
            attr_prot_trans = attr_ig_3d_arr[h,w,0]; attr_prose = attr_ig_3d_arr[h,w,1]; attr_esmc = attr_ig_3d_arr[h,w,2]
            non_zero_pxl_attr_lst = [attr_prot_trans, attr_prose, attr_esmc]
            non_zero_pxl_attr_lst_lst.append(non_zero_pxl_attr_lst)
            tot_attr = attr_prot_trans + attr_prose + attr_esmc
            non_zero_pxl_tot_attr_lst.append(tot_attr)
        postproc_attr_res_dict['non_zero_pxl_attr_lst_lst'] = non_zero_pxl_attr_lst_lst
        postproc_attr_res_dict['non_zero_pxl_tot_attr_lst'] = non_zero_pxl_tot_attr_lst
        postproc_attr_res_dict_file_nm_with_loc = os.path.join(postproc_attr_result_dir, 'idx_' + str(postproc_attr_res_dict['idx']) + '.pkl')
        joblib.dump(value=postproc_attr_res_dict, filename=postproc_attr_res_dict_file_nm_with_loc, compress=3)


if __name__ == '__main__':
    root_path = os.path.join('/project/root/directory/path/here')
    model_path = os.path.join(root_path, 'dataset/proc_data_tl_feat_to_img/img_p2ip_cnn/train/ResNet_tlStructEsmc_r400n18DnoWS')
    partial_model_name = 'ImgP2ipCnn'
    device_type = 'cuda'
    docking_version_lst = ['5_5']  
    for docking_version in docking_version_lst:
        preproc_test_result_before_attr_calc(root_path=root_path, model_path=model_path, docking_version=docking_version)
        con_attr_dict_lst = calc_attribution(root_path=root_path, model_path=model_path, partial_model_name=partial_model_name, docking_version=docking_version, device_type=device_type)
        postproc_attr_result(root_path=root_path, model_path=model_path, docking_version=docking_version, con_attr_dict_lst = con_attr_dict_lst)
