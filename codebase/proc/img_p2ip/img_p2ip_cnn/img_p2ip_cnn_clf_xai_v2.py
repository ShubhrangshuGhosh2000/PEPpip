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
from proc.img_p2ip.img_p2ip_datamodule_struct_esmc_v2 import ImgP2ipCustomDataModule
from proc.img_p2ip.img_p2ip_cnn.img_p2ip_cnn_clf_train_struct_esmc_v2 import ImgP2ipCnn
from utils import PPIPUtils


def preproc_test_result_before_attr_calc(root_path='./', model_path='./', spec_type='ecoli'):
    test_tag = model_path.split('/')[-1]
    test_result_dir = os.path.join(root_path, 'dataset/proc_data_tl_feat_to_img/img_p2ip_cnn/test/dscript_other_spec_' + test_tag, spec_type)
    spec_pred_res_df = pd.read_csv(os.path.join(test_result_dir, spec_type + '_pred_res.csv'))
    test_pair_list_path = os.path.join(root_path, 'dataset/orig_data_DS/pairs', spec_type + '_test.tsv' )
    spec_test_pairs_df = pd.read_csv(test_pair_list_path, delimiter='\t', header=None, names=['prot1_id', 'prot2_id', 'label'])
    concat_pred_res_df = pd.concat([spec_test_pairs_df, spec_pred_res_df], axis=1, join='inner')
    concat_pred_res_df['label'] = concat_pred_res_df['label'].astype(int)  
    same = concat_pred_res_df['label'].equals(concat_pred_res_df['actual_res'])
    if(not same):
        print("!! ERROR !! The columns 'label' and 'actual_res' are not identical for " + spec_type)
    # create an index column
    index_lst = list(range(concat_pred_res_df.shape[0]))
    concat_pred_res_df.insert(loc = 0, column = 'idx', value = index_lst)
    xai_result_dir = os.path.join(root_path, 'dataset/proc_data_tl_feat_to_img/img_p2ip_cnn/xai/dscript_other_spec_' + test_tag, spec_type)
    PPIPUtils.createFolder(xai_result_dir)
    concat_pred_res_file_nm_with_loc = os.path.join(xai_result_dir, spec_type + '_concat_pred_res.csv')
    concat_pred_res_df.to_csv(concat_pred_res_file_nm_with_loc, index=False)


def load_final_ckpt_model(root_path='./', model_path='./', partial_model_name = 'ImgP2ipCnn'):
    final_chkpt_path = os.path.join(model_path, partial_model_name + '*.ckpt' )
    final_ckpt_file_name = glob.glob(final_chkpt_path, recursive=False)[0]
    model = ImgP2ipCnn.load_from_checkpoint(final_ckpt_file_name)
    return model


def prepare_test_data(root_path='./', model=None, spec_type='ecoli'):
    test_data_module = ImgP2ipCustomDataModule(root_path=root_path, batch_size=model.hparams.config['batch_size']
                                               , workers=os.cpu_count() - 5  
                                               , img_resoln=model.hparams.config['img_resoln'], spec_type=spec_type
                                               , dbl_combi_flg=model.hparams.config['dbl_combi_flg'])
    test_data_module.setup(stage='test')
    custom_test_dataset = test_data_module.test_data
    return custom_test_dataset


def calc_attribution(root_path='./', model_path='./', partial_model_name = 'ImgP2ipCnn', spec_type='ecoli', device_type='cpu'):
    model = load_final_ckpt_model(root_path, model_path, partial_model_name)
    custom_test_dataset = prepare_test_data(root_path, model, spec_type)
    test_tag = model_path.split('/')[-1]
    xai_result_dir = os.path.join(root_path, 'dataset/proc_data_tl_feat_to_img/img_p2ip_cnn/xai/dscript_other_spec_' + test_tag, spec_type)
    pred_res_file_nm_with_loc = os.path.join(xai_result_dir, spec_type + '_concat_pred_res.csv')
    pred_res_df = pd.read_csv(pred_res_file_nm_with_loc)
    corct_pos_pred_res_df = pred_res_df.loc[(pred_res_df['label'] == 1) & (pred_res_df['pred_res'] == 1)]
    corct_pos_pred_res_df = corct_pos_pred_res_df.reset_index(drop=True)
    con_attrForPositiveRes_dict_lst = []
    model.eval()
    for index, row in corct_pos_pred_res_df.iterrows():
        attr_res_dict = row.to_dict()
        idx_col_val = row['idx']
        prot_prot_3dTensor, norm_prot_prot_3dTensor, label_tensor, orig_ht_b4_padding, orig_width_b4_padding = custom_test_dataset.get_item_with_orig_input(idx_col_val)
        model = model.to(torch.device(device_type))
        norm_prot_prot_3dTensor = norm_prot_prot_3dTensor.to(torch.device(device_type))
        input = norm_prot_prot_3dTensor.unsqueeze(0)
        input.requires_grad = True
        ig = IntegratedGradients(model)
        model.zero_grad()
        attr_ig, delta = ig.attribute(inputs=input, target=label_tensor, baselines=input * 0, return_convergence_delta=True)
        attr_ig_arr_cpu = np.transpose(attr_ig.squeeze().cpu().detach().numpy(), (1, 2, 0))
        attr_res_dict['attr_ig'] = attr_ig_arr_cpu
        input_arr_cpu = np.transpose(prot_prot_3dTensor.cpu().detach().numpy(), (1, 2, 0))
        attr_res_dict['orig_input'] = input_arr_cpu
        attr_res_dict['orig_ht_b4_padding'] = orig_ht_b4_padding
        attr_res_dict['orig_width_b4_padding'] = orig_width_b4_padding
        con_attrForPositiveRes_dict_lst.append(attr_res_dict)
    return con_attrForPositiveRes_dict_lst


def postproc_attr_result(root_path='./', model_path='./', spec_type='ecoli', con_attrForPositiveRes_dict_lst=None):
    test_tag = model_path.split('/')[-1]
    xai_result_dir = os.path.join(root_path, 'dataset/proc_data_tl_feat_to_img/img_p2ip_cnn/xai/dscript_other_spec_' + test_tag, spec_type)
    postproc_attr_result_dir = os.path.join(xai_result_dir, spec_type + '_attr_postproc')
    PPIPUtils.createFolder(postproc_attr_result_dir)
    con_attrForPositiveRes_dict_lst_len = len(con_attrForPositiveRes_dict_lst)
    start_itr = 0  
    for itr in range(start_itr, con_attrForPositiveRes_dict_lst_len, 1):
        attr_res_dict = con_attrForPositiveRes_dict_lst[itr]
        postproc_attr_res_dict = {dict_key: attr_res_dict[dict_key] for dict_key in attr_res_dict if dict_key not in ['attr_ig', 'orig_input']}
        input_3d_hwc_arr = attr_res_dict['orig_input']
        orig_ht_b4_padding = attr_res_dict['orig_ht_b4_padding']; orig_width_b4_padding = attr_res_dict['orig_width_b4_padding']
        non_zero_pxl_indx_lst_lst = []
        for h in range(orig_ht_b4_padding):  
            for w in range(orig_width_b4_padding):  
                non_zero_pxl_indx_lst_lst.append([h,w])
        postproc_attr_res_dict['non_zero_pxl_indx_lst_lst'] = non_zero_pxl_indx_lst_lst
        attr_ig_3d_arr = attr_res_dict['attr_ig']
        non_zero_pxl_attr_lst_lst = []
        non_zero_pxl_tot_attr_lst = []
        for h, w in non_zero_pxl_indx_lst_lst:
            attr_protTrans = attr_ig_3d_arr[h,w,0]; attr_prose = attr_ig_3d_arr[h,w,1]; attr_esmc = attr_ig_3d_arr[h,w,2]
            non_zero_pxl_attr_lst = [attr_protTrans, attr_prose, attr_esmc]
            non_zero_pxl_attr_lst_lst.append(non_zero_pxl_attr_lst)
            tot_attr = attr_protTrans + attr_prose + attr_esmc
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
    spec_type_lst = ['human', 'ecoli', 'fly', 'mouse', 'worm', 'yeast']  
    for spec_type in spec_type_lst:
        preproc_test_result_before_attr_calc(root_path=root_path, model_path=model_path, spec_type=spec_type)
        con_attrForPositiveRes_dict_lst = calc_attribution(root_path=root_path, model_path=model_path, partial_model_name=partial_model_name, spec_type=spec_type, device_type=device_type)
        postproc_attr_result(root_path=root_path, model_path=model_path, spec_type=spec_type, con_attrForPositiveRes_dict_lst = con_attrForPositiveRes_dict_lst)
