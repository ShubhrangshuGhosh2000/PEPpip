import os
import sys
from pathlib import Path
path_root = Path(__file__).parents[2]  
sys.path.insert(0, str(path_root))
from utils import dl_reproducible_result_util
import glob
import numpy as np
import pandas as pd
import joblib
from utils import DockUtils, PPIPUtils


def create_hyb_predicted_contact_map(root_path='./', cnn_model_path='./', trx_model_path='./', docking_version='5_5'
                               , trx_contrib_wt=0.5, consider_full=False, min_max_mode='R', attr_mode='total',  pcmp_mode='SCGB'
                               ):
    img_pip_xai_dock_hyb_data_path = os.path.join(root_path, f'dataset/proc_data_tl_feat_to_img/img_pip_hybrid/img_pip_xai_dock_{docking_version}_hybrid')
    tp_pred_hyb_contact_map_loc = os.path.join(img_pip_xai_dock_hyb_data_path, 'tp_pred_hyb_contact_map', f'attrMode_{attr_mode}', f'minMaxMode_{min_max_mode}', f'pcmp_mode_{pcmp_mode}')
    PPIPUtils.createFolder(tp_pred_hyb_contact_map_loc)
    fn_pred_hyb_contact_map_loc = os.path.join(img_pip_xai_dock_hyb_data_path, 'fn_pred_hyb_contact_map', f'attrMode_{attr_mode}', f'minMaxMode_{min_max_mode}', f'pcmp_mode_{pcmp_mode}')
    PPIPUtils.createFolder(fn_pred_hyb_contact_map_loc)
    cnn_test_tag = cnn_model_path.split('/')[-1]
    cnn_xai_result_dir = os.path.join(root_path, f'dataset/proc_data_tl_feat_to_img/img_p2ip_cnn/xai_dock_{docking_version}/{cnn_test_tag}')
    cnn_tp_pred_contact_map_proc_loc = os.path.join(cnn_xai_result_dir, 'tp_pred_contact_map_proc', f'attrMode_{attr_mode}', f'minMaxMode_{min_max_mode}', f'pcmp_mode_{pcmp_mode}')
    cnn_fn_pred_contact_map_proc_loc = os.path.join(cnn_xai_result_dir, 'fn_pred_contact_map_proc', f'attrMode_{attr_mode}', f'minMaxMode_{min_max_mode}', f'pcmp_mode_{pcmp_mode}')
    trx_test_tag = trx_model_path.split('/')[-1]
    trx_xai_result_dir = os.path.join(root_path, f'dataset/proc_data_tl_feat_to_img/img_p2ip_trx/xai_dock_{docking_version}_pp/{trx_test_tag}')
    trx_tp_pred_contact_map_proc_loc = os.path.join(trx_xai_result_dir, 'tp_pred_contact_map_proc', f'attnMode_{attr_mode}', f'minMaxMode_{min_max_mode}', f'pcmp_mode_{pcmp_mode}')
    trx_fn_pred_contact_map_proc_loc = os.path.join(trx_xai_result_dir, 'fn_pred_contact_map_proc', f'attnMode_{attr_mode}', f'minMaxMode_{min_max_mode}', f'pcmp_mode_{pcmp_mode}')
    doc_test_tsv_fl_nm_loc = os.path.join(root_path, f'dataset/preproc_data_docking_BM_{docking_version}/derived_feat/dock', 'dock_test_lenLimit_400.tsv')
    dock_test_df = pd.read_csv(doc_test_tsv_fl_nm_loc, sep='\t', header=None, names=['prot_1_id', 'prot_2_id', 'test_labels'])
    cnn_tp_fn_lst, trx_tp_fn_lst = [], []
    for index, row in dock_test_df.iterrows():
        prot_1_id, prot_2_id = row['prot_1_id'], row['prot_2_id']
        protein_name, chain_1_name = prot_1_id.split('_')
        protein_name, chain_2_name = prot_2_id.split('_')
        cnn_tp_pred_contact_map_proc_path = os.path.join(cnn_tp_pred_contact_map_proc_loc, f"pred_contact_map_{protein_name}_{chain_1_name}_{chain_2_name}_attrMode_{attr_mode}_minMaxMode_{min_max_mode}_pcmp_mode_{pcmp_mode}.pkl")
        cnn_pred_contact_map_lst = glob.glob(cnn_tp_pred_contact_map_proc_path, recursive=False)
        if(len(cnn_pred_contact_map_lst) == 0):
            cnn_fn_pred_contact_map_proc_path = os.path.join(cnn_fn_pred_contact_map_proc_loc, f"pred_contact_map_{protein_name}_{chain_1_name}_{chain_2_name}_attrMode_{attr_mode}_minMaxMode_{min_max_mode}_pcmp_mode_{pcmp_mode}.pkl")
            cnn_pred_contact_map_lst = glob.glob(cnn_fn_pred_contact_map_proc_path, recursive=False)
            if(len(cnn_pred_contact_map_lst) == 0):
                raise Exception(f"No cnn_pred_contact_map found in cnn_tp_pred_contact_map_proc and cnn_fn_pred_contact_map_proc folders for \
                                protein_name: {protein_name} :: chain_1_name: {chain_1_name} :: chain_2_name: {chain_2_name}")
            else:  
                cnn_tp_fn_lst.append('fn')
        else:
            cnn_tp_fn_lst.append('tp')
        cnn_pred_contact_map_name = cnn_pred_contact_map_lst[0].split('/')[-1]
        cnn_pcmp_mode = cnn_pred_contact_map_name.split('.')[0].split('_')[-1]  
        cnn_pred_contact_map = joblib.load(cnn_pred_contact_map_lst[0])
        trx_tp_pred_contact_map_proc_path = os.path.join(trx_tp_pred_contact_map_proc_loc, f"pred_contact_map_{protein_name}_{chain_1_name}_{chain_2_name}_attnMode_{attr_mode}_minMaxMode_{min_max_mode}_pcmp_mode_{pcmp_mode}.pkl")
        trx_pred_contact_map_lst = glob.glob(trx_tp_pred_contact_map_proc_path, recursive=False)
        if(len(trx_pred_contact_map_lst) == 0):
            trx_fn_pred_contact_map_proc_path = os.path.join(trx_fn_pred_contact_map_proc_loc, f"pred_contact_map_{protein_name}_{chain_1_name}_{chain_2_name}_attnMode_{attr_mode}_minMaxMode_{min_max_mode}_pcmp_mode_{pcmp_mode}.pkl")
            trx_pred_contact_map_lst = glob.glob(trx_fn_pred_contact_map_proc_path, recursive=False)
            if(len(trx_pred_contact_map_lst) == 0):
                raise Exception(f"No trx_pred_contact_map found in trx_tp_pred_contact_map_proc and trx_fn_pred_contact_map_proc folders for \
                                protein_name: {protein_name} :: chain_1_name: {chain_1_name} :: chain_2_name: {chain_2_name}")
            else:  
                trx_tp_fn_lst.append('fn')
        else:
            trx_tp_fn_lst.append('tp')
        trx_pred_contact_map_name = trx_pred_contact_map_lst[0].split('/')[-1]
        trx_pcmp_mode = trx_pred_contact_map_name.split('.')[0].split('_')[-1]  
        trx_pred_contact_map = joblib.load(trx_pred_contact_map_lst[0])
        pred_hyb_contact_map = DockUtils.weighted_addition_normalized(cnn_pred_contact_map, trx_pred_contact_map, trx_contrib_wt=trx_contrib_wt)
        pred_hyb_contact_map_location = None
        if(trx_tp_fn_lst[-1] == 'tp'):  
            pred_hyb_contact_map_location = os.path.join(tp_pred_hyb_contact_map_loc, f"pred_hyb_contact_map_{protein_name}_{chain_1_name}_{chain_2_name}_attrMode_{attr_mode}_minMaxMode_{min_max_mode}_pcmp_mode_{pcmp_mode}.pkl")
        else:
            pred_hyb_contact_map_location = os.path.join(fn_pred_hyb_contact_map_loc, f"pred_hyb_contact_map_{protein_name}_{chain_1_name}_{chain_2_name}_attrMode_{attr_mode}_minMaxMode_{min_max_mode}_pcmp_mode_{pcmp_mode}.pkl")
        joblib.dump(pred_hyb_contact_map, pred_hyb_contact_map_location)


def calculate_emd_betwn_gt_and_pred_hyb_contact_maps(root_path='./', docking_version='5_5'
                                                 , consider_full=False, min_max_mode='R', attr_mode='total'
                                                 , pct=99.0, pcmp_mode='SCGB'):
    gt_contact_map_dir = os.path.join(root_path, f"dataset/preproc_data_docking_BM_{docking_version}/prot_gt_contact_map")
    img_pip_xai_dock_hyb_data_path = os.path.join(root_path, f'dataset/proc_data_tl_feat_to_img/img_pip_hybrid/img_pip_xai_dock_{docking_version}_hybrid')
    tp_pred_hyb_contact_map_loc = os.path.join(img_pip_xai_dock_hyb_data_path, 'tp_pred_hyb_contact_map', f'attrMode_{attr_mode}', f'minMaxMode_{min_max_mode}', f'pcmp_mode_{pcmp_mode}')
    fn_pred_hyb_contact_map_loc = os.path.join(img_pip_xai_dock_hyb_data_path, 'fn_pred_hyb_contact_map', f'attrMode_{attr_mode}', f'minMaxMode_{min_max_mode}', f'pcmp_mode_{pcmp_mode}')
    emd_result_dir = os.path.join(img_pip_xai_dock_hyb_data_path, 'emd_result')
    PPIPUtils.createFolder(emd_result_dir)
    contact_heatmap_plt_dir = None
    if(consider_full):
        contact_heatmap_plt_dir = os.path.join(emd_result_dir, 'contact_heatmap_plt_full')
    else:
        contact_heatmap_plt_dir = os.path.join(emd_result_dir, 'contact_heatmap_plt_9Xp')
    PPIPUtils.createFolder(contact_heatmap_plt_dir)
    doc_test_tsv_fl_nm_loc = os.path.join(root_path, f'dataset/preproc_data_docking_BM_{docking_version}/derived_feat/dock', 'dock_test_lenLimit_400.tsv')
    dock_test_df = pd.read_csv(doc_test_tsv_fl_nm_loc, sep='\t', header=None, names=['prot_1_id', 'prot_2_id', 'test_labels'])
    dock_test_row_idx_lst, prot_1_id_lst, prot_2_id_lst, attr_mode_lst, min_max_mode_lst, tp_fn_lst, emd_lst, pcmp_mode_lst = [], [], [], [], [], [], [], []  
    aupr_lst, auroc_lst, specificity_lst, precision_lst, recall_lst, f1_score_lst = [], [], [], [], [], []  
    tp_cont_lst, fp_cont_lst, fn_cont_lst, tn_cont_lst = [], [], [], []  
    for index, row in dock_test_df.iterrows():
        prot_1_id, prot_2_id = row['prot_1_id'], row['prot_2_id']
        protein_name, chain_1_name = prot_1_id.split('_')
        protein_name, chain_2_name = prot_2_id.split('_')
        tp_pred_hyb_contact_map_proc_path = os.path.join(tp_pred_hyb_contact_map_loc, f"pred_hyb_contact_map_{protein_name}_{chain_1_name}_{chain_2_name}_attrMode_{attr_mode}_minMaxMode_{min_max_mode}_pcmp_mode_{pcmp_mode}.pkl")
        pred_hyb_contact_map_lst = glob.glob(tp_pred_hyb_contact_map_proc_path, recursive=False)
        if(len(pred_hyb_contact_map_lst) == 0):
            fn_pred_hyb_contact_map_proc_path = os.path.join(fn_pred_hyb_contact_map_loc, f"pred_hyb_contact_map_{protein_name}_{chain_1_name}_{chain_2_name}_attrMode_{attr_mode}_minMaxMode_{min_max_mode}_pcmp_mode_{pcmp_mode}.pkl")
            pred_hyb_contact_map_lst = glob.glob(fn_pred_hyb_contact_map_proc_path, recursive=False)
            if(len(pred_hyb_contact_map_lst) == 0):
                raise Exception(f"No pred_hyb_contact_map found in tp_pred_hyb_contact_map_proc and fn_pred_hyb_contact_map_proc folders for \
                                protein_name: {protein_name} :: chain_1_name: {chain_1_name} :: chain_2_name: {chain_2_name}")
            else:  
                tp_fn_lst.append('fn')
        else:
            tp_fn_lst.append('tp')
        pred_hyb_contact_map_name = pred_hyb_contact_map_lst[0].split('/')[-1]
        pcmp_mode = pred_hyb_contact_map_name.split('.')[0].split('_')[-1]  
        pred_hyb_contact_map = joblib.load(pred_hyb_contact_map_lst[0])
        gt_contact_map_location = os.path.join(gt_contact_map_dir, f"contact_map_{protein_name}_{chain_1_name}_{chain_2_name}.pkl")
        gt_contact_map = joblib.load(gt_contact_map_location)
        if(pred_hyb_contact_map.shape == gt_contact_map.shape):
           print('dimensionally both the interaction maps (pred_hyb_contact_map and gt_contact_map) are same')
        else:
            raise Exception(f"Dimensionally both the interaction maps (pred_hyb_contact_map and gt_contact_map) are not the same for \
                                protein_name: {protein_name} :: chain_1_name: {chain_1_name} :: chain_2_name: {chain_2_name} \
                                pred_hyb_contact_map.shape: {pred_hyb_contact_map.shape} :: gt_contact_map.shape: {gt_contact_map.shape}")
        if(not consider_full):
            percentile = np.percentile(pred_hyb_contact_map, pct)
            pred_hyb_contact_map[pred_hyb_contact_map <= percentile] = 0
            pred_hyb_contact_map[pred_hyb_contact_map > percentile] = 1
        emd = DockUtils.calculateEMD(pred_hyb_contact_map, gt_contact_map)
        perf_metric_dict = DockUtils.evaluate_contact_maps(pred_hyb_contact_map, gt_contact_map, distance_metric='cityblock')
        dock_test_row_idx_lst.append(index); prot_1_id_lst.append(prot_1_id); prot_2_id_lst.append(prot_2_id)
        attr_mode_lst.append(attr_mode); min_max_mode_lst.append(min_max_mode); pcmp_mode_lst.append(pcmp_mode); emd_lst.append(emd)
        aupr_lst.append(perf_metric_dict['aupr']); auroc_lst.append(perf_metric_dict['auroc']); specificity_lst.append(perf_metric_dict['specificity'])
        precision_lst.append(perf_metric_dict['precision']); recall_lst.append(perf_metric_dict['recall']); f1_score_lst.append(perf_metric_dict['f1_score'])
        tp_cont_lst.append(perf_metric_dict['conf_matrix_dict']['tp']); fp_cont_lst.append(perf_metric_dict['conf_matrix_dict']['fp'])
        fn_cont_lst.append(perf_metric_dict['conf_matrix_dict']['fn']); tn_cont_lst.append(perf_metric_dict['conf_matrix_dict']['tn'])
    emd_res_df = pd.DataFrame({'dock_test_row_idx': dock_test_row_idx_lst, 'prot_1_id': prot_1_id_lst, 'prot_2_id': prot_2_id_lst, 'min_max_mode': min_max_mode_lst
                           , 'attr_mode': attr_mode_lst, 'tp_fn': tp_fn_lst, 'pcmp_mode': pcmp_mode, 'emd': emd_lst
                           , 'aupr': aupr_lst, 'auroc': auroc_lst, 'specificity': specificity_lst
                           , 'precision': precision_lst, 'recall': recall_lst, 'f1_score': f1_score_lst
                           , 'tp_cont': tp_cont_lst, 'fp_cont': fp_cont_lst, 'fn_cont': fn_cont_lst, 'tn_cont': tn_cont_lst})
    return emd_res_df


def create_consolidated_emd_res_csv(emd_res_df_lst=None, root_path='./', docking_version='5_5'
                                                 , consider_full=False):
    img_pip_xai_dock_hyb_data_path = os.path.join(root_path, f'dataset/proc_data_tl_feat_to_img/img_pip_hybrid/img_pip_xai_dock_{docking_version}_hybrid')
    emd_result_dir = os.path.join(img_pip_xai_dock_hyb_data_path, 'emd_result')
    PPIPUtils.createFolder(emd_result_dir)
    consolidated_emd_res_df = pd.concat(emd_res_df_lst)
    if(consider_full):
        consolidated_emd_res_df.to_csv(os.path.join(emd_result_dir, f'gt_vs_pred_emd_full_hybrid.csv'), index=False)
    else:
        consolidated_emd_res_df.to_csv(os.path.join(emd_result_dir, f'gt_vs_pred_emd_9Xp_hybrid.csv'), index=False)
    consolidated_emd_res_df = consolidated_emd_res_df.drop(columns=['dock_test_row_idx', 'prot_1_id', 'prot_2_id'])
    grouped_df = consolidated_emd_res_df.groupby(['tp_fn', 'attr_mode', 'min_max_mode', 'pcmp_mode']).agg(
                                aupr_avg=('aupr', 'mean'), auroc_avg=('auroc', 'mean'), specif_avg=('specificity', 'mean'),
                                prec_avg=('precision', 'mean'), recall_avg=('recall', 'mean'), f1_avg=('f1_score', 'mean'),
                                tp_cont_sum=('tp_cont', 'sum'), fp_cont_sum=('fp_cont', 'sum'), fn_cont_sum=('fn_cont', 'sum'), tn_cont_sum=('tn_cont', 'sum')
                            ).reset_index()
    grouped_df['aupr_avg'] = grouped_df['aupr_avg'].round(3); grouped_df['auroc_avg'] = grouped_df['auroc_avg'].round(3); grouped_df['specif_avg'] = grouped_df['specif_avg'].round(3)
    grouped_df['prec_avg'] = grouped_df['prec_avg'].round(3); grouped_df['recall_avg'] = grouped_df['recall_avg'].round(3); grouped_df['f1_avg'] = grouped_df['f1_avg'].round(3)
    if(consider_full):
        grouped_df.to_csv(os.path.join(emd_result_dir, f'overall_eval_res_full_hybrid.csv'), index=False)
    else:
        grouped_df.to_csv(os.path.join(emd_result_dir, f'overall_eval_res_9Xp_hybrid.csv'), index=False)


if __name__ == '__main__':
    root_path = os.path.join('/project/root/directory/path/here')
    cnn_model_path = os.path.join(root_path, 'dataset/proc_data_tl_feat_to_img/img_p2ip_cnn/train/ResNet_tlStructEsmc_r400n18DnoWS')
    trx_model_path = os.path.join(root_path, 'dataset/proc_data_tl_feat_to_img/img_p2ip_trx/train/CDAMViT_tlStructEsmc_r400p16')
    trx_contrib_wt = 0.5  
    pct = 99.0  
    docking_version_lst = ['5_5']  
    consider_full_lst = [True]  
    attr_mode_lst = ['total']  
    min_max_mode_lst = ['N', 'R']  
    pcmp_mode_lst = ['SCGB', 'CSGB']  
    for docking_version in docking_version_lst:
        for consider_full in consider_full_lst:
            emd_res_df_lst = []  
            for attr_mode in attr_mode_lst:
                for min_max_mode in min_max_mode_lst:
                    for pcmp_mode in pcmp_mode_lst:
                        create_hyb_predicted_contact_map(root_path=root_path, cnn_model_path=cnn_model_path, trx_model_path=trx_model_path, docking_version=docking_version
                                                   , trx_contrib_wt=trx_contrib_wt, consider_full=consider_full, min_max_mode=min_max_mode, attr_mode=attr_mode, pcmp_mode=pcmp_mode)
                        emd_res_df = calculate_emd_betwn_gt_and_pred_hyb_contact_maps(root_path=root_path, docking_version=docking_version, consider_full=consider_full
                                                                                , min_max_mode=min_max_mode, attr_mode=attr_mode, pct=pct, pcmp_mode=pcmp_mode)
                        emd_res_df_lst.append(emd_res_df)
            create_consolidated_emd_res_csv(emd_res_df_lst=emd_res_df_lst, root_path=root_path, docking_version=docking_version, consider_full=consider_full)
