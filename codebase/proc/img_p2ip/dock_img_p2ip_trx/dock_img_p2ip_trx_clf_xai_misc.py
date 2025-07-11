import os
import sys
from pathlib import Path
path_root = Path(__file__).parents[3]  
sys.path.insert(0, str(path_root))
from utils import dl_reproducible_result_util
import glob
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from utils import DockUtils, PPIPUtils, ProteinContactMapUtils


def create_predicted_contact_map(root_path='./', model_path='./', docking_version='5_5', min_max_mode='R', attn_mode='total', pcmp_mode='SCGB'):
    pdb_file_location = os.path.join(root_path, f'dataset/preproc_data_docking_BM_{docking_version}/pdb_files')
    test_tag = model_path.split('/')[-1]
    xai_result_dir = os.path.join(root_path, f'dataset/proc_data_tl_feat_to_img/img_p2ip_trx/xai_dock_{docking_version}/{test_tag}')
    concat_pred_res_file_nm_with_loc = os.path.join(xai_result_dir, f'concat_pred_res_dock_{docking_version}.csv')
    concat_pred_res_df = pd.read_csv(concat_pred_res_file_nm_with_loc)
    tp_df = concat_pred_res_df[ concat_pred_res_df['actual_res'] == concat_pred_res_df['pred_res']]
    tp_df = tp_df.reset_index(drop=True)
    fn_df = concat_pred_res_df[ concat_pred_res_df['actual_res'] != concat_pred_res_df['pred_res']]
    fn_df = fn_df.reset_index(drop=True)
    tp_pred_contact_map_loc = os.path.join(xai_result_dir, 'tp_pred_contact_map', f'attnMode_{attn_mode}', f'minMaxMode_{min_max_mode}')
    PPIPUtils.createFolder(tp_pred_contact_map_loc)
    tp_pred_contact_map_proc_loc = os.path.join(xai_result_dir, 'tp_pred_contact_map_proc', f'attnMode_{attn_mode}', f'minMaxMode_{min_max_mode}', f'pcmp_mode_{pcmp_mode}')
    PPIPUtils.createFolder(tp_pred_contact_map_proc_loc)
    fn_pred_contact_map_loc = os.path.join(xai_result_dir, 'fn_pred_contact_map', f'attnMode_{attn_mode}', f'minMaxMode_{min_max_mode}')
    PPIPUtils.createFolder(fn_pred_contact_map_loc)
    fn_pred_contact_map_proc_loc = os.path.join(xai_result_dir, 'fn_pred_contact_map_proc', f'attnMode_{attn_mode}', f'minMaxMode_{min_max_mode}', f'pcmp_mode_{pcmp_mode}')
    PPIPUtils.createFolder(fn_pred_contact_map_proc_loc)
    postproc_attn_result_dir = os.path.join(xai_result_dir, 'attn_postproc')
    for itr, row in tp_df.iterrows():
        idx, prot1_id, prot2_id = row['idx'], row['prot1_id'], row['prot2_id']
        postproc_res_dict_file_nm_with_loc = os.path.join(postproc_attn_result_dir, 'idx_' + str(idx) + '.pkl')
        postproc_res_dict = joblib.load(postproc_res_dict_file_nm_with_loc)
        non_zero_pxl_indx_lst_lst = postproc_res_dict['non_zero_pxl_indx_lst_lst']
        pxl_attn_2d_arr = postproc_res_dict['attn']
        pred_attn_map_2d = create_pred_attn_map_2d(non_zero_pxl_indx_lst_lst, pxl_attn_2d_arr, attn_mode)
        pred_contact_map = None
        min_val, max_val = pred_attn_map_2d.min(), pred_attn_map_2d.max()
        if(min_max_mode == 'N'):  
            pred_contact_map = (pred_attn_map_2d - min_val) / (max_val - min_val)
        elif(min_max_mode == 'R'):  
            pred_contact_map = (max_val - pred_attn_map_2d) / (max_val - min_val)
        protein_name, chain_1_name = prot1_id.split('_')
        protein_name, chain_2_name = prot2_id.split('_')
        pred_contact_map_location = os.path.join(tp_pred_contact_map_loc, f"pred_contact_map_{protein_name}_{chain_1_name}_{chain_2_name}_attnMode_{attn_mode}_minMaxMode_{min_max_mode}.pkl")
        joblib.dump(pred_contact_map, pred_contact_map_location)
        lambda_weight = 0.1  
        prot_contact_map_processor = ProteinContactMapUtils.ProteinContactMapProcessor(pdb_file_location=pdb_file_location, protein_name=protein_name
                                                                                       , chain_1_name=chain_1_name, chain_2_name=chain_2_name
                                                                                       , pred_contact_map=pred_contact_map, seq_order=pcmp_mode, lambda_weight=lambda_weight)
        pred_contact_map_proc = prot_contact_map_processor.process()
        pred_contact_map_proc_location = os.path.join(tp_pred_contact_map_proc_loc, f"pred_contact_map_{protein_name}_{chain_1_name}_{chain_2_name}_attnMode_{attn_mode}_minMaxMode_{min_max_mode}_pcmp_mode_{pcmp_mode}.pkl")
        joblib.dump(pred_contact_map_proc, pred_contact_map_proc_location)
    for itr, row in fn_df.iterrows():
        idx, prot1_id, prot2_id = row['idx'], row['prot1_id'], row['prot2_id']
        postproc_res_dict_file_nm_with_loc = os.path.join(postproc_attn_result_dir, 'idx_' + str(idx) + '.pkl')
        postproc_res_dict = joblib.load(postproc_res_dict_file_nm_with_loc)
        non_zero_pxl_indx_lst_lst = postproc_res_dict['non_zero_pxl_indx_lst_lst']
        pxl_attn_2d_arr = postproc_res_dict['attn']
        pred_attn_map_2d = create_pred_attn_map_2d(non_zero_pxl_indx_lst_lst, pxl_attn_2d_arr, attn_mode)
        pred_contact_map = None
        min_val, max_val = pred_attn_map_2d.min(), pred_attn_map_2d.max()
        if(min_max_mode == 'N'):  
            pred_contact_map = (max_val - pred_attn_map_2d) / (max_val - min_val)
        elif(min_max_mode == 'R'):  
            pred_contact_map = (pred_attn_map_2d - min_val) / (max_val - min_val)
        protein_name, chain_1_name = prot1_id.split('_')
        protein_name, chain_2_name = prot2_id.split('_')
        pred_contact_map_location = os.path.join(fn_pred_contact_map_loc, f"pred_contact_map_{protein_name}_{chain_1_name}_{chain_2_name}_attnMode_{attn_mode}_minMaxMode_{min_max_mode}.pkl")
        joblib.dump(pred_contact_map, pred_contact_map_location)
        lambda_weight = 0.3  
        prot_contact_map_processor = ProteinContactMapUtils.ProteinContactMapProcessor(pdb_file_location=pdb_file_location, protein_name=protein_name
                                                                                       , chain_1_name=chain_1_name, chain_2_name=chain_2_name
                                                                                       , pred_contact_map=pred_contact_map, seq_order=pcmp_mode, lambda_weight=lambda_weight)
        pred_contact_map_proc = prot_contact_map_processor.process()
        pred_contact_map_proc_location = os.path.join(fn_pred_contact_map_proc_loc, f"pred_contact_map_{protein_name}_{chain_1_name}_{chain_2_name}_attnMode_{attn_mode}_minMaxMode_{min_max_mode}_pcmp_mode_{pcmp_mode}.pkl")
        joblib.dump(pred_contact_map_proc, pred_contact_map_proc_location)


def create_pred_attn_map_2d(non_zero_pxl_indx_lst_lst, pxl_attn_2d_arr, attn_mode):
    h_max = max(non_zero_pxl_indx_lst_lst, key=lambda x: x[0])[0]
    w_max = max(non_zero_pxl_indx_lst_lst, key=lambda x: x[1])[1]
    pred_attn_map_2d = np.zeros((h_max+1, w_max+1))
    for i in range(len(non_zero_pxl_indx_lst_lst)):
        x, y = non_zero_pxl_indx_lst_lst[i]
        if(attn_mode == 'total'):
            pred_attn_map_2d[x, y] = pxl_attn_2d_arr[x, y]
    return pred_attn_map_2d


def calculate_emd_betwn_gt_and_pred_contact_maps(root_path='./', model_path='./', docking_version='5_5'
                                                 , consider_full=False, min_max_mode='R', attn_mode='total'
                                                 , pct=99.0, pcmp_mode='SCGB'):
    gt_contact_map_dir = os.path.join(root_path, f"dataset/preproc_data_docking_BM_{docking_version}/prot_gt_contact_map")
    test_tag = model_path.split('/')[-1]
    xai_result_dir = os.path.join(root_path, f'dataset/proc_data_tl_feat_to_img/img_p2ip_trx/xai_dock_{docking_version}/{test_tag}')
    tp_pred_contact_map_proc_loc = os.path.join(xai_result_dir, 'tp_pred_contact_map_proc', f'attnMode_{attn_mode}', f'minMaxMode_{min_max_mode}', f'pcmp_mode_{pcmp_mode}')
    fn_pred_contact_map_proc_loc = os.path.join(xai_result_dir, 'fn_pred_contact_map_proc', f'attnMode_{attn_mode}', f'minMaxMode_{min_max_mode}', f'pcmp_mode_{pcmp_mode}')
    emd_result_dir = os.path.join(xai_result_dir, 'emd_result')
    PPIPUtils.createFolder(emd_result_dir)
    contact_heatmap_plt_dir = None
    if(consider_full):
        contact_heatmap_plt_dir = os.path.join(emd_result_dir, 'contact_heatmap_plt_full')
    else:
        contact_heatmap_plt_dir = os.path.join(emd_result_dir, 'contact_heatmap_plt_9Xp')
    PPIPUtils.createFolder(contact_heatmap_plt_dir)
    doc_test_tsv_fl_nm_loc = os.path.join(root_path, f'dataset/preproc_data_docking_BM_{docking_version}/derived_feat/dock', 'dock_test_lenLimit_400.tsv')
    dock_test_df = pd.read_csv(doc_test_tsv_fl_nm_loc, sep='\t', header=None, names=['prot_1_id', 'prot_2_id', 'test_labels'])
    dock_test_row_idx_lst, prot_1_id_lst, prot_2_id_lst, attn_mode_lst, min_max_mode_lst, tp_fn_lst, emd_lst, pcmp_mode_lst = [], [], [], [], [], [], [], []  
    top_Lby5_prec_lst, top_Lby10_prec_lst, top_50_prec_lst = [], [], []  
    for index, row in dock_test_df.iterrows():
        prot_1_id, prot_2_id = row['prot_1_id'], row['prot_2_id']
        protein_name, chain_1_name = prot_1_id.split('_')
        protein_name, chain_2_name = prot_2_id.split('_')
        tp_pred_contact_map_proc_path = os.path.join(tp_pred_contact_map_proc_loc, f"pred_contact_map_{protein_name}_{chain_1_name}_{chain_2_name}_attnMode_{attn_mode}_minMaxMode_{min_max_mode}_pcmp_mode_{pcmp_mode}.pkl")
        pred_contact_map_lst = glob.glob(tp_pred_contact_map_proc_path, recursive=False)
        if(len(pred_contact_map_lst) == 0):
            fn_pred_contact_map_proc_path = os.path.join(fn_pred_contact_map_proc_loc, f"pred_contact_map_{protein_name}_{chain_1_name}_{chain_2_name}_attnMode_{attn_mode}_minMaxMode_{min_max_mode}_pcmp_mode_{pcmp_mode}.pkl")
            pred_contact_map_lst = glob.glob(fn_pred_contact_map_proc_path, recursive=False)
            if(len(pred_contact_map_lst) == 0):
                raise Exception(f"No pred_contact_map found in tp_pred_contact_map_proc and fn_pred_contact_map_proc folders for \
                                protein_name: {protein_name} :: chain_1_name: {chain_1_name} :: chain_2_name: {chain_2_name}")
            else:  
                tp_fn_lst.append('fn')
        else:
            tp_fn_lst.append('tp')
        pred_contact_map_name = pred_contact_map_lst[0].split('/')[-1]
        pcmp_mode = pred_contact_map_name.split('.')[0].split('_')[-1]  
        pred_contact_map = joblib.load(pred_contact_map_lst[0])
        gt_contact_map_location = os.path.join(gt_contact_map_dir, f"contact_map_{protein_name}_{chain_1_name}_{chain_2_name}.pkl")
        gt_contact_map = joblib.load(gt_contact_map_location)
        if(pred_contact_map.shape == gt_contact_map.shape):
           print('dimensionally both the interaction maps (pred_contact_map and gt_contact_map) are same')
        else:
            raise Exception(f"Dimensionally both the interaction maps (pred_contact_map and gt_contact_map) are not the same for \
                                protein_name: {protein_name} :: chain_1_name: {chain_1_name} :: chain_2_name: {chain_2_name} \
                                pred_contact_map.shape: {pred_contact_map.shape} :: gt_contact_map.shape: {gt_contact_map.shape}")
        if(not consider_full):
            percentile = np.percentile(pred_contact_map, pct)
            pred_contact_map[pred_contact_map <= percentile] = 0
            pred_contact_map[pred_contact_map > percentile] = 1
        emd = DockUtils.calculateEMD(pred_contact_map, gt_contact_map)
        perf_metric_dict = DockUtils.evaluate_contact_maps(pred_contact_map, gt_contact_map)
        dock_test_row_idx_lst.append(index); prot_1_id_lst.append(prot_1_id); prot_2_id_lst.append(prot_2_id)
        attn_mode_lst.append(attn_mode); min_max_mode_lst.append(min_max_mode); pcmp_mode_lst.append(pcmp_mode); emd_lst.append(emd)
        top_Lby5_prec_lst.append(perf_metric_dict['top_Lby5_prec']); top_Lby10_prec_lst.append(perf_metric_dict['top_Lby10_prec']); top_50_prec_lst.append(perf_metric_dict['top_50_prec'])
    emd_res_df = pd.DataFrame({'dock_test_row_idx': dock_test_row_idx_lst, 'prot_1_id': prot_1_id_lst, 'prot_2_id': prot_2_id_lst, 'min_max_mode': min_max_mode_lst
                                , 'attn_mode': attn_mode_lst, 'tp_fn': tp_fn_lst, 'pcmp_mode': pcmp_mode, 'emd': emd_lst
                                , 'top_Lby5_prec': top_Lby5_prec_lst, 'top_Lby10_prec': top_Lby10_prec_lst, 'top_50_prec': top_50_prec_lst
                            })
    return emd_res_df


def create_consolidated_emd_res_csv(emd_res_df_lst=None, root_path='./', model_path='./', docking_version='5_5'
                                                 , consider_full=False):
    test_tag = model_path.split('/')[-1]
    xai_result_dir = os.path.join(root_path, f'dataset/proc_data_tl_feat_to_img/img_p2ip_trx/xai_dock_{docking_version}/{test_tag}')
    emd_result_dir = os.path.join(xai_result_dir, 'emd_result')
    PPIPUtils.createFolder(emd_result_dir)
    consolidated_emd_res_df = pd.concat(emd_res_df_lst)
    if(consider_full):
        consolidated_emd_res_df.to_csv(os.path.join(emd_result_dir, f'gt_vs_pred_emd_full.csv'), index=False)
    else:
        consolidated_emd_res_df.to_csv(os.path.join(emd_result_dir, f'gt_vs_pred_emd_9Xp.csv'), index=False)
    consolidated_emd_res_df = consolidated_emd_res_df.drop(columns=['dock_test_row_idx', 'prot_1_id', 'prot_2_id'])
    grouped_df = consolidated_emd_res_df.groupby(['tp_fn', 'attn_mode', 'min_max_mode', 'pcmp_mode']).agg(
                                top_Lby5_prec_avg=('top_Lby5_prec', 'mean'), top_Lby10_prec_avg=('top_Lby10_prec', 'mean'), top_50_prec_avg=('top_50_prec', 'mean'),
                            ).reset_index()
    grouped_df['top_Lby5_prec_avg'] = grouped_df['top_Lby5_prec_avg'].round(3); grouped_df['top_Lby10_prec_avg'] = grouped_df['top_Lby10_prec_avg'].round(3); grouped_df['top_50_prec_avg'] = grouped_df['top_50_prec_avg'].round(3)
    if(consider_full):
        grouped_df.to_csv(os.path.join(emd_result_dir, f'overall_eval_res_full.csv'), index=False)
    else:
        grouped_df.to_csv(os.path.join(emd_result_dir, f'overall_eval_res_9Xp.csv'), index=False)


if __name__ == '__main__':
    root_path = os.path.join('/project/root/directory/path/here')
    model_path = os.path.join(root_path, 'dataset/proc_data_tl_feat_to_img/img_p2ip_trx/train/CDAMViT_tlStructEsmc_r400p16')
    pct = 99.0  
    docking_version_lst = ['5_5']  
    consider_full_lst = [True]  
    attn_mode_lst = ['total']  
    min_max_mode_lst = ['N', 'R']  
    pcmp_mode_lst = ['SCG', 'CSG']  
    for docking_version in docking_version_lst:
        for consider_full in consider_full_lst:
            emd_res_df_lst = []  
            for attn_mode in attn_mode_lst:
                for min_max_mode in min_max_mode_lst:
                    for pcmp_mode in pcmp_mode_lst:
                        create_predicted_contact_map(root_path=root_path, model_path=model_path, docking_version=docking_version, min_max_mode=min_max_mode, attn_mode=attn_mode, pcmp_mode=pcmp_mode)
                        emd_res_df = calculate_emd_betwn_gt_and_pred_contact_maps(root_path=root_path, model_path=model_path, docking_version=docking_version, consider_full=consider_full
                                                                                , min_max_mode=min_max_mode, attn_mode=attn_mode, pct=pct, pcmp_mode=pcmp_mode)
                        emd_res_df_lst.append(emd_res_df)
            create_consolidated_emd_res_csv(emd_res_df_lst=emd_res_df_lst, root_path=root_path, model_path=model_path, docking_version=docking_version, consider_full=consider_full)

