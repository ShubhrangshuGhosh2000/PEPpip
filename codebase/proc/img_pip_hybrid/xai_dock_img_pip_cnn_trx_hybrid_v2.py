import os
import sys
from pathlib import Path

path_root = Path(__file__).parents[2]  # upto 'codebase' folder
sys.path.insert(0, str(path_root))
# print(sys.path)

from utils import dl_reproducible_result_util
import glob
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
# from matplotlib.colors import Normalize

from utils import DockUtils, PPIPUtils, ProteinContactMapUtils


def create_hyb_predicted_contact_map(root_path='./', cnn_model_path='./', trx_model_path='./', docking_version='5_5'
                               , trx_contrib_wt=0.5, consider_full=False, min_max_mode='R', attr_mode='total',  pcmp_mode='SCGB'
                               ):
    print('\n #############################\n inside the create_hybrid_contact_maps() method - Start\n')
    print('\n########## docking_version: ' + str(docking_version))

    # Create folders for the hybrid interaction maps for the tp and fn result. tp and fn result is w.r.t. the transformer-based model prediction.
    print('Creating folders for the hybrid interaction maps for the tp and fn result')
    img_pip_xai_dock_hyb_data_path = os.path.join(root_path, f'dataset/proc_data_tl_feat_to_img/img_pip_hybrid/img_pip_xai_dock_{docking_version}_hybrid')
    tp_pred_hyb_contact_map_loc = os.path.join(img_pip_xai_dock_hyb_data_path, 'tp_pred_hyb_contact_map', f'attrMode_{attr_mode}', f'minMaxMode_{min_max_mode}', f'pcmp_mode_{pcmp_mode}')
    PPIPUtils.createFolder(tp_pred_hyb_contact_map_loc)
    fn_pred_hyb_contact_map_loc = os.path.join(img_pip_xai_dock_hyb_data_path, 'fn_pred_hyb_contact_map', f'attrMode_{attr_mode}', f'minMaxMode_{min_max_mode}', f'pcmp_mode_{pcmp_mode}')
    PPIPUtils.createFolder(fn_pred_hyb_contact_map_loc)

    # Set CNN estimated interaction map location -Start
    cnn_test_tag = cnn_model_path.split('/')[-1]
    cnn_xai_result_dir = os.path.join(root_path, f'dataset/proc_data_tl_feat_to_img/img_p2ip_cnn/xai_dock_{docking_version}/{cnn_test_tag}')
    cnn_tp_pred_contact_map_proc_loc = os.path.join(cnn_xai_result_dir, 'tp_pred_contact_map_proc', f'attrMode_{attr_mode}', f'minMaxMode_{min_max_mode}', f'pcmp_mode_{pcmp_mode}')
    cnn_fn_pred_contact_map_proc_loc = os.path.join(cnn_xai_result_dir, 'fn_pred_contact_map_proc', f'attrMode_{attr_mode}', f'minMaxMode_{min_max_mode}', f'pcmp_mode_{pcmp_mode}')
    # Set CNN estimated interaction map location -End

    # Set transformer (trx) estimated interaction map location -Start
    trx_test_tag = trx_model_path.split('/')[-1]
    trx_xai_result_dir = os.path.join(root_path, f'dataset/proc_data_tl_feat_to_img/img_p2ip_trx/xai_dock_{docking_version}_pp/{trx_test_tag}')
    trx_tp_pred_contact_map_proc_loc = os.path.join(trx_xai_result_dir, 'tp_pred_contact_map_proc', f'attnMode_{attr_mode}', f'minMaxMode_{min_max_mode}', f'pcmp_mode_{pcmp_mode}')
    trx_fn_pred_contact_map_proc_loc = os.path.join(trx_xai_result_dir, 'fn_pred_contact_map_proc', f'attnMode_{attr_mode}', f'minMaxMode_{min_max_mode}', f'pcmp_mode_{pcmp_mode}')
    # Set transformer (trx) estimated interaction map location -End

    # read the dock_test.tsv file in a pandas dataframe
    # doc_test_tsv_fl_nm_loc = os.path.join(root_path, f'dataset/preproc_data_docking_BM_{docking_version}/derived_feat/dock', 'dock_test.tsv')
    doc_test_tsv_fl_nm_loc = os.path.join(root_path, f'dataset/preproc_data_docking_BM_{docking_version}/derived_feat/dock', 'dock_test_lenLimit_400.tsv')
    dock_test_df = pd.read_csv(doc_test_tsv_fl_nm_loc, sep='\t', header=None, names=['prot_1_id', 'prot_2_id', 'test_labels'])
    
    cnn_tp_fn_lst, trx_tp_fn_lst = [], []
    
    # iterate over the dock_test_df
    for index, row in dock_test_df.iterrows():
        print(f"\n ################# starting {index}-th row out of {dock_test_df.shape[0]-1}\n")
        prot_1_id, prot_2_id = row['prot_1_id'], row['prot_2_id']
        # protein id has the format of [protein_name]_[chain_name]
        protein_name, chain_1_name = prot_1_id.split('_')
        protein_name, chain_2_name = prot_2_id.split('_')
        print(f"protein_name: {protein_name} :: chain_1_name: {chain_1_name} :: chain_2_name: {chain_2_name} :: attrMode: {attr_mode} :: minMaxMode: {min_max_mode} :: pcmp_mode: {pcmp_mode}")
        
        # ################### Retrieve the cnn_pred_contact_map -Start
        print('\n################### Retrieve the cnn_pred_contact_map -Start')
        # first check whether it belongs to cnn_tp_pred_contact_map_proc_loc
        cnn_tp_pred_contact_map_proc_path = os.path.join(cnn_tp_pred_contact_map_proc_loc, f"pred_contact_map_{protein_name}_{chain_1_name}_{chain_2_name}_attrMode_{attr_mode}_minMaxMode_{min_max_mode}_pcmp_mode_{pcmp_mode}.pkl")
        cnn_pred_contact_map_lst = glob.glob(cnn_tp_pred_contact_map_proc_path, recursive=False)
        if(len(cnn_pred_contact_map_lst) == 0):
            print('No cnn_pred_contact_map found in cnn_tp_pred_contact_map_proc folder and next searching in cnn_fn_pred_contact_map_proc folder...')
            # if the desired pred_contact_map does not exist inside tp_pred_contact_map_proc_path, then
            # search inside fn_pred_contact_map_proc_path for the same
            cnn_fn_pred_contact_map_proc_path = os.path.join(cnn_fn_pred_contact_map_proc_loc, f"pred_contact_map_{protein_name}_{chain_1_name}_{chain_2_name}_attrMode_{attr_mode}_minMaxMode_{min_max_mode}_pcmp_mode_{pcmp_mode}.pkl")
            cnn_pred_contact_map_lst = glob.glob(cnn_fn_pred_contact_map_proc_path, recursive=False)
            if(len(cnn_pred_contact_map_lst) == 0):
                raise Exception(f"No cnn_pred_contact_map found in cnn_tp_pred_contact_map_proc and cnn_fn_pred_contact_map_proc folders for \
                                protein_name: {protein_name} :: chain_1_name: {chain_1_name} :: chain_2_name: {chain_2_name}")
            else:  
                # desired cnn_pred_contact_map exists inside cnn_fn_pred_contact_map_proc folder
                print('desired cnn_pred_contact_map exists inside cnn_fn_pred_contact_map_proc folder')
                cnn_tp_fn_lst.append('fn')
            # end of else block
        else:
            print('desired pred_contact_map exists inside tp_pred_contact_map_proc folder')
            cnn_tp_fn_lst.append('tp')
        # end of if-else block: if(len(cnn_pred_contact_map_lst) == 0):
        cnn_pred_contact_map_name = cnn_pred_contact_map_lst[0].split('/')[-1]
        cnn_pcmp_mode = cnn_pred_contact_map_name.split('.')[0].split('_')[-1]  # f"pred_contact_map_{protein_name}_{chain_1_name}_{chain_2_name}_attrMode_{attr_mode}_minMaxMode_{min_max_mode}_pcmp_mode_{pcmp_mode}.pkl"
        print(f'cnn_pcmp_mode: {cnn_pcmp_mode}')
        cnn_pred_contact_map = joblib.load(cnn_pred_contact_map_lst[0])
        # ################### Retrieve the cnn_pred_contact_map -End
        print('################### Retrieve the cnn_pred_contact_map -End \n')
        
        # ################### Retrieve the trx_pred_contact_map -Start
        print('\n################### Retrieve the trx_pred_contact_map -Start')
        # first check whether it belongs to trx_tp_pred_contact_map_proc_loc
        trx_tp_pred_contact_map_proc_path = os.path.join(trx_tp_pred_contact_map_proc_loc, f"pred_contact_map_{protein_name}_{chain_1_name}_{chain_2_name}_attnMode_{attr_mode}_minMaxMode_{min_max_mode}_pcmp_mode_{pcmp_mode}.pkl")
        trx_pred_contact_map_lst = glob.glob(trx_tp_pred_contact_map_proc_path, recursive=False)
        if(len(trx_pred_contact_map_lst) == 0):
            print('No trx_pred_contact_map found in trx_tp_pred_contact_map_proc folder and next searching in trx_fn_pred_contact_map_proc folder...')
            # if the desired pred_contact_map does not exist inside tp_pred_contact_map_proc_path, then
            # search inside fn_pred_contact_map_proc_path for the same
            trx_fn_pred_contact_map_proc_path = os.path.join(trx_fn_pred_contact_map_proc_loc, f"pred_contact_map_{protein_name}_{chain_1_name}_{chain_2_name}_attnMode_{attr_mode}_minMaxMode_{min_max_mode}_pcmp_mode_{pcmp_mode}.pkl")
            trx_pred_contact_map_lst = glob.glob(trx_fn_pred_contact_map_proc_path, recursive=False)
            if(len(trx_pred_contact_map_lst) == 0):
                raise Exception(f"No trx_pred_contact_map found in trx_tp_pred_contact_map_proc and trx_fn_pred_contact_map_proc folders for \
                                protein_name: {protein_name} :: chain_1_name: {chain_1_name} :: chain_2_name: {chain_2_name}")
            else:  
                # desired trx_pred_contact_map exists inside trx_fn_pred_contact_map_proc folder
                print('desired trx_pred_contact_map exists inside trx_fn_pred_contact_map_proc folder')
                trx_tp_fn_lst.append('fn')
            # end of else block
        else:
            print('desired pred_contact_map exists inside tp_pred_contact_map_proc folder')
            trx_tp_fn_lst.append('tp')
        # end of if-else block: if(len(trx_pred_contact_map_lst) == 0):
        trx_pred_contact_map_name = trx_pred_contact_map_lst[0].split('/')[-1]
        trx_pcmp_mode = trx_pred_contact_map_name.split('.')[0].split('_')[-1]  # f"pred_contact_map_{protein_name}_{chain_1_name}_{chain_2_name}_attnMode_{attr_mode}_minMaxMode_{min_max_mode}_pcmp_mode_{pcmp_mode}.pkl"
        print(f'trx_pcmp_mode: {trx_pcmp_mode}')
        trx_pred_contact_map = joblib.load(trx_pred_contact_map_lst[0])
        # ################### Retrieve the trx_pred_contact_map -End
        print('################### Retrieve the trx_pred_contact_map -End \n')

        # ################### Carry out the weighted addition
        print('\n################### Carry out the weighted addition')
        pred_hyb_contact_map = DockUtils.weighted_addition_normalized(cnn_pred_contact_map, trx_pred_contact_map, trx_contrib_wt=trx_contrib_wt)

        print('Saving pred_hyb_contact_map')
        pred_hyb_contact_map_location = None
        if(trx_tp_fn_lst[-1] == 'tp'):  # tp and fn result is w.r.t. the transformer-based model prediction.
            pred_hyb_contact_map_location = os.path.join(tp_pred_hyb_contact_map_loc, f"pred_hyb_contact_map_{protein_name}_{chain_1_name}_{chain_2_name}_attrMode_{attr_mode}_minMaxMode_{min_max_mode}_pcmp_mode_{pcmp_mode}.pkl")
        else:
            pred_hyb_contact_map_location = os.path.join(fn_pred_hyb_contact_map_loc, f"pred_hyb_contact_map_{protein_name}_{chain_1_name}_{chain_2_name}_attrMode_{attr_mode}_minMaxMode_{min_max_mode}_pcmp_mode_{pcmp_mode}.pkl")

        joblib.dump(pred_hyb_contact_map, pred_hyb_contact_map_location)
        print(f'pred_hyb_contact_map is saved as {pred_hyb_contact_map_location}')
    # end of for loop: for itr, row in fn_df.iterrows():
    print('\n #############################\n inside the create_hybrid_contact_maps() method - End\n')


def calculate_emd_betwn_gt_and_pred_hyb_contact_maps(root_path='./', docking_version='5_5'
                                                 , consider_full=False, min_max_mode='R', attr_mode='total'
                                                 , pct=99.0, pcmp_mode='SCGB'):
    print('\n #############################\n inside the calculate_emd_betwn_gt_and_pred_hyb_contact_maps() method - Start\n')
    print('\n########## docking_version: ' + str(docking_version))
    gt_contact_map_dir = os.path.join(root_path, f"dataset/preproc_data_docking_BM_{docking_version}/prot_gt_contact_map")
    
    img_pip_xai_dock_hyb_data_path = os.path.join(root_path, f'dataset/proc_data_tl_feat_to_img/img_pip_hybrid/img_pip_xai_dock_{docking_version}_hybrid')
    tp_pred_hyb_contact_map_loc = os.path.join(img_pip_xai_dock_hyb_data_path, 'tp_pred_hyb_contact_map', f'attrMode_{attr_mode}', f'minMaxMode_{min_max_mode}', f'pcmp_mode_{pcmp_mode}')
    fn_pred_hyb_contact_map_loc = os.path.join(img_pip_xai_dock_hyb_data_path, 'fn_pred_hyb_contact_map', f'attrMode_{attr_mode}', f'minMaxMode_{min_max_mode}', f'pcmp_mode_{pcmp_mode}')
    emd_result_dir = os.path.join(img_pip_xai_dock_hyb_data_path, 'emd_result')
    PPIPUtils.createFolder(emd_result_dir)
    
    contact_heatmap_plt_dir = None
    if(consider_full):
        # consider full-version version of the interaction maps
        contact_heatmap_plt_dir = os.path.join(emd_result_dir, 'contact_heatmap_plt_full')
    else:
        # consider 99 percentile version of the interaction maps
        contact_heatmap_plt_dir = os.path.join(emd_result_dir, 'contact_heatmap_plt_9Xp')
    PPIPUtils.createFolder(contact_heatmap_plt_dir)

    # read the dock_test.tsv file in a pandas dataframe
    # doc_test_tsv_fl_nm_loc = os.path.join(root_path, f'dataset/preproc_data_docking_BM_{docking_version}/derived_feat/dock', 'dock_test.tsv')
    doc_test_tsv_fl_nm_loc = os.path.join(root_path, f'dataset/preproc_data_docking_BM_{docking_version}/derived_feat/dock', 'dock_test_lenLimit_400.tsv')
    dock_test_df = pd.read_csv(doc_test_tsv_fl_nm_loc, sep='\t', header=None, names=['prot_1_id', 'prot_2_id', 'test_labels'])
    # iterate over the dock_test_df and for each pair of participating proteins in each row calculate the EMD between the ground-truth (gt) contact
    # map and the hybrid interaction map
    dock_test_row_idx_lst, prot_1_id_lst, prot_2_id_lst, attr_mode_lst, min_max_mode_lst, tp_fn_lst, emd_lst, pcmp_mode_lst = [], [], [], [], [], [], [], []  # lists to create a dataframe later
    aupr_lst, auroc_lst, specificity_lst, precision_lst, recall_lst, f1_score_lst = [], [], [], [], [], []  # lists to create a dataframe later
    tp_cont_lst, fp_cont_lst, fn_cont_lst, tn_cont_lst = [], [], [], []  # lists to create a dataframe later

    for index, row in dock_test_df.iterrows():
        print(f"\n ################# starting {index}-th row out of {dock_test_df.shape[0]-1}\n")
        prot_1_id, prot_2_id = row['prot_1_id'], row['prot_2_id']
        # protein id has the format of [protein_name]_[chain_name]
        protein_name, chain_1_name = prot_1_id.split('_')
        protein_name, chain_2_name = prot_2_id.split('_')
        print(f"protein_name: {protein_name} :: chain_1_name: {chain_1_name} :: chain_2_name: {chain_2_name} :: attrMode: {attr_mode} :: minMaxMode: {min_max_mode} :: pcmp_mode: {pcmp_mode}")
        # retrieve the pred_hyb_contact_map
        print('retrieving the pred_hyb_contact_map...')
        # first check whether it belongs to tp_pred_hyb_contact_map_loc
        tp_pred_hyb_contact_map_proc_path = os.path.join(tp_pred_hyb_contact_map_loc, f"pred_hyb_contact_map_{protein_name}_{chain_1_name}_{chain_2_name}_attrMode_{attr_mode}_minMaxMode_{min_max_mode}_pcmp_mode_{pcmp_mode}.pkl")
        pred_hyb_contact_map_lst = glob.glob(tp_pred_hyb_contact_map_proc_path, recursive=False)
        if(len(pred_hyb_contact_map_lst) == 0):
            print('No pred_hyb_contact_map found in tp_pred_hyb_contact_map_proc folder and next searching in fn_pred_hyb_contact_map_proc folder...')
            # if the desired pred_hyb_contact_map does not exist inside tp_pred_hyb_contact_map_proc_path, then
            # search inside fn_pred_hyb_contact_map_proc_path for the same
            fn_pred_hyb_contact_map_proc_path = os.path.join(fn_pred_hyb_contact_map_loc, f"pred_hyb_contact_map_{protein_name}_{chain_1_name}_{chain_2_name}_attrMode_{attr_mode}_minMaxMode_{min_max_mode}_pcmp_mode_{pcmp_mode}.pkl")
            pred_hyb_contact_map_lst = glob.glob(fn_pred_hyb_contact_map_proc_path, recursive=False)
            if(len(pred_hyb_contact_map_lst) == 0):
                raise Exception(f"No pred_hyb_contact_map found in tp_pred_hyb_contact_map_proc and fn_pred_hyb_contact_map_proc folders for \
                                protein_name: {protein_name} :: chain_1_name: {chain_1_name} :: chain_2_name: {chain_2_name}")
            else:  
                # desired pred_hyb_contact_map exists inside fn_pred_hyb_contact_map_proc folder
                print('desired pred_hyb_contact_map exists inside fn_pred_hyb_contact_map_proc folder')
                tp_fn_lst.append('fn')
            # end of else block
        else:
            print('desired pred_hyb_contact_map exists inside tp_pred_hyb_contact_map_proc folder')
            tp_fn_lst.append('tp')
        # end of if-else block: if(len(pred_hyb_contact_map_lst) == 0):
        pred_hyb_contact_map_name = pred_hyb_contact_map_lst[0].split('/')[-1]
        pcmp_mode = pred_hyb_contact_map_name.split('.')[0].split('_')[-1]  # f"pred_hyb_contact_map_{protein_name}_{chain_1_name}_{chain_2_name}_attrMode_{attr_mode}_minMaxMode_{min_max_mode}_pcmp_mode_{pcmp_mode}.pkl"
        print(f'pcmp_mode: {pcmp_mode}')
        pred_hyb_contact_map = joblib.load(pred_hyb_contact_map_lst[0])

        # retrieve the gt_contact_map
        print('retrieving the gt_contact_map...')
        gt_contact_map_location = os.path.join(gt_contact_map_dir, f"contact_map_{protein_name}_{chain_1_name}_{chain_2_name}.pkl")
        gt_contact_map = joblib.load(gt_contact_map_location)

        # check whether dimensionally both the interaction maps are same
        if(pred_hyb_contact_map.shape == gt_contact_map.shape):
           print('dimensionally both the interaction maps (pred_hyb_contact_map and gt_contact_map) are same')
        else:
            print(f'!!!!! dimensionally both the interaction maps (pred_hyb_contact_map and gt_contact_map) are not the same')
            print(f"pred_hyb_contact_map.shape: {pred_hyb_contact_map.shape} :: gt_contact_map.shape: {gt_contact_map.shape}")
            raise Exception(f"Dimensionally both the interaction maps (pred_hyb_contact_map and gt_contact_map) are not the same for \
                                protein_name: {protein_name} :: chain_1_name: {chain_1_name} :: chain_2_name: {chain_2_name} \
                                pred_hyb_contact_map.shape: {pred_hyb_contact_map.shape} :: gt_contact_map.shape: {gt_contact_map.shape}")
        # calculate the EMD between pred_hyb_contact_map and gt_contact_map
        print('Calculating the EMD between pred_hyb_contact_map and gt_contact_map')
        if(not consider_full):
            # Calculate the 9X-th percentile value for pred_hyb_contact_map
            percentile = np.percentile(pred_hyb_contact_map, pct)
            # Set values below 9X-th percentile to zero
            pred_hyb_contact_map[pred_hyb_contact_map <= percentile] = 0
            # ##### Set values above 9X-th percentile to one for plot generation purpose
            pred_hyb_contact_map[pred_hyb_contact_map > percentile] = 1

            # # Calculate the 9X-th percentile value for gt_contact_map
            # percentile = np.percentile(gt_contact_map, 99)
            # # Set values below 9X-th percentile to zero
            # gt_contact_map[gt_contact_map <= percentile] = 0

        emd = DockUtils.calculateEMD(pred_hyb_contact_map, gt_contact_map)
        print(f'emd: {emd}')
         # calculate the absolute difference between pred_hyb_contact_map and gt_contact_map
        # abs_diff = DockUtils.calcMatrixAbsDiff(pred_hyb_contact_map, gt_contact_map)

        # Calculate all the performance metrics comparing pred_hyb_contact_map and gt_contact_map
        perf_metric_dict = DockUtils.evaluate_contact_maps(pred_hyb_contact_map, gt_contact_map, distance_metric='cityblock')
        print(f'perf_metric_dict: {perf_metric_dict}')

        # generate contact heatmap plot for pred_hyb_contact_map
        print('generating contact heatmap plot for pred_hyb_contact_map')
        pred_hyb_contact_map_plot_path = os.path.join(contact_heatmap_plt_dir, f'prot_{prot_1_id}_prot_{prot_2_id}_{tp_fn_lst[-1]}_attrMode_{attr_mode}_minMaxMode_{min_max_mode}_pcmp_mode_{pcmp_mode}_pred_hyb_contact_map.png')
        generate_pred_contact_heatmap_plot(pred_hyb_contact_map, title='Predicted interaction map', row = prot_1_id, col=prot_2_id, save_path=pred_hyb_contact_map_plot_path)

        # generate contact heatmap plot for gt_contact_map
        print('generating contact heatmap plot for gt_contact_map')
        gt_contact_map_plot_path = os.path.join(contact_heatmap_plt_dir, f'prot_{prot_1_id}_prot_{prot_2_id}_gt_contact_map.png')
        generate_gt_contact_heatmap_plot(gt_contact_map, title='gt interaction map', row = prot_1_id, col=prot_2_id, save_path=gt_contact_map_plot_path)

        # populate the lists required to create a df later
        dock_test_row_idx_lst.append(index); prot_1_id_lst.append(prot_1_id); prot_2_id_lst.append(prot_2_id)
        attr_mode_lst.append(attr_mode); min_max_mode_lst.append(min_max_mode); pcmp_mode_lst.append(pcmp_mode); emd_lst.append(emd)
    
        aupr_lst.append(perf_metric_dict['aupr']); auroc_lst.append(perf_metric_dict['auroc']); specificity_lst.append(perf_metric_dict['specificity'])
        precision_lst.append(perf_metric_dict['precision']); recall_lst.append(perf_metric_dict['recall']); f1_score_lst.append(perf_metric_dict['f1_score'])
        tp_cont_lst.append(perf_metric_dict['conf_matrix_dict']['tp']); fp_cont_lst.append(perf_metric_dict['conf_matrix_dict']['fp'])
        fn_cont_lst.append(perf_metric_dict['conf_matrix_dict']['fn']); tn_cont_lst.append(perf_metric_dict['conf_matrix_dict']['tn'])
    # end of for loop: for index, row in dock_test_df.iterrows():
    # create the dataframe and save it
    emd_res_df = pd.DataFrame({'dock_test_row_idx': dock_test_row_idx_lst, 'prot_1_id': prot_1_id_lst, 'prot_2_id': prot_2_id_lst, 'min_max_mode': min_max_mode_lst
                           , 'attr_mode': attr_mode_lst, 'tp_fn': tp_fn_lst, 'pcmp_mode': pcmp_mode, 'emd': emd_lst
                           , 'aupr': aupr_lst, 'auroc': auroc_lst, 'specificity': specificity_lst
                           , 'precision': precision_lst, 'recall': recall_lst, 'f1_score': f1_score_lst
                           , 'tp_cont': tp_cont_lst, 'fp_cont': fp_cont_lst, 'fn_cont': fn_cont_lst, 'tn_cont': tn_cont_lst})
    print('\n #################\n inside the calculate_emd_betwn_gt_and_pred_hyb_contact_maps() method - End\n')
    return emd_res_df


def create_consolidated_emd_res_csv(emd_res_df_lst=None, root_path='./', docking_version='5_5'
                                                 , consider_full=False):
    print('\n #############################\n inside the create_consolidated_emd_res_csv() method - Start\n')
    print('\n########## docking_version: ' + str(docking_version))
    img_pip_xai_dock_hyb_data_path = os.path.join(root_path, f'dataset/proc_data_tl_feat_to_img/img_pip_hybrid/img_pip_xai_dock_{docking_version}_hybrid')
    emd_result_dir = os.path.join(img_pip_xai_dock_hyb_data_path, 'emd_result')
    PPIPUtils.createFolder(emd_result_dir)
    
    # create the dataframe and save it
    consolidated_emd_res_df = pd.concat(emd_res_df_lst)
    # save emd_res_df
    if(consider_full):
        # consider full-version version of the interaction maps
        consolidated_emd_res_df.to_csv(os.path.join(emd_result_dir, f'gt_vs_pred_emd_full_hybrid.csv'), index=False)
    else:
        # consider 99 percentile version of the interaction maps
        consolidated_emd_res_df.to_csv(os.path.join(emd_result_dir, f'gt_vs_pred_emd_9Xp_hybrid.csv'), index=False)

    # Discard the redundant columns
    consolidated_emd_res_df = consolidated_emd_res_df.drop(columns=['dock_test_row_idx', 'prot_1_id', 'prot_2_id'])

    # Group by tp_fn, attr_mode, and attr_mode
    grouped_df = consolidated_emd_res_df.groupby(['tp_fn', 'attr_mode', 'min_max_mode', 'pcmp_mode']).agg(
                                aupr_avg=('aupr', 'mean'), auroc_avg=('auroc', 'mean'), specif_avg=('specificity', 'mean'),
                                prec_avg=('precision', 'mean'), recall_avg=('recall', 'mean'), f1_avg=('f1_score', 'mean'),
                                tp_cont_sum=('tp_cont', 'sum'), fp_cont_sum=('fp_cont', 'sum'), fn_cont_sum=('fn_cont', 'sum'), tn_cont_sum=('tn_cont', 'sum')
                            ).reset_index()

    grouped_df['aupr_avg'] = grouped_df['aupr_avg'].round(3); grouped_df['auroc_avg'] = grouped_df['auroc_avg'].round(3); grouped_df['specif_avg'] = grouped_df['specif_avg'].round(3)
    grouped_df['prec_avg'] = grouped_df['prec_avg'].round(3); grouped_df['recall_avg'] = grouped_df['recall_avg'].round(3); grouped_df['f1_avg'] = grouped_df['f1_avg'].round(3)

    if(consider_full):
        # consider full-version version of the interaction maps
        grouped_df.to_csv(os.path.join(emd_result_dir, f'overall_eval_res_full_hybrid.csv'), index=False)
    else:
        # Save the grouped dataframe as a CSV file
        grouped_df.to_csv(os.path.join(emd_result_dir, f'overall_eval_res_9Xp_hybrid.csv'), index=False)

    print('\n #################\n inside the create_consolidated_emd_res_csv() method - End\n')


def generate_pred_contact_heatmap_plot(data_array_2d, title='Heatmap Plot with Colormap', row='prot_1_id', col='prot_2_id', save_path='heatmap_plot.png'):
    """
    Generate a heatmap plot using Matplotlib for the given numpy array.

    Parameters:
    - data_array_2d (numpy.ndarray): Input 2D array of floats with values between 0.0 and 1.0.
    - save_path (str): Path to save the generated heatmap plot image (default: 'heatmap_plot.png').

    Returns:
    - None

    Example:
    >>> data_array_2d = np.random.rand(5, 8)  # Example 5x8 array of random floats
    >>> generate_pred_contact_heatmap_plot(data_array_2d, save_path='my_heatmap_plot.png')
    """
    # # Ensure that input array is a numpy array
    data_array_2d = np.array(data_array_2d)
    # Check if the array has the correct dimension (2D)
    if data_array_2d.ndim != 2:
        raise ValueError("Input array must be a 2D numpy array")
    # Calculate the 90th percentile value
    # percentile = np.percentile(data_array_2d, 99)
    # print(f'percentile: {percentile}')

    # Set values below 90th percentile to zero
    # data_array_2d[data_array_2d <= percentile] = 0

    # In this program, the colormap is set to gray_r (reverse of the grayscale colormap), and 
    # the normalization is adjusted such that values greater than the 90th percentile appear darker in the colormap.
    # Create a colormap with increasing density of black for values greater than 90th percentile
    cmap = plt.cm.gray_r
    # norm = Normalize(vmin=0, vmax=percentile)

    # Create a custom heatmap plot
    plt.imshow(data_array_2d, cmap=cmap, norm=None, interpolation=None, origin='upper', aspect=None)
    # Set colorbar
    cbar = plt.colorbar()
    # cbar.set_label('Color Scale (0.0 to 1.0)')
    # Invert Y-axis to start from the top
    # plt.gca().invert_yaxis()
    # Set labels and title
    plt.xlabel(f'{col} (Col Idx : Y Axis)')
    plt.ylabel(f'{row} (Row Indices : X Axis)')
    plt.title(f'{title}')
    # Save the plot
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    # Show the plot (optional)
    # plt.show()
    # Close the plot
    plt.close()


def generate_gt_contact_heatmap_plot(data_array_2d, title='Heatmap Plot with Colormap', row='prot_1_id', col='prot_2_id', save_path='heatmap_plot.png'):
    """
    Generate a heatmap plot using Matplotlib for the given numpy array.

    Parameters:
    - data_array_2d (numpy.ndarray): Input 2D array of floats with values between 0.0 and 1.0.
    - save_path (str): Path to save the generated heatmap plot image (default: 'heatmap_plot.png').

    Returns:
    - None

    Example:
    >>> data_array_2d = np.random.rand(5, 8)  # Example 5x8 array of random floats
    >>> generate_gt_contact_heatmap_plot(data_array_2d, save_path='my_heatmap_plot.png')
    """
    # # Ensure that input array is a numpy array
    data_array_2d = np.array(data_array_2d)
    # Check if the array has the correct dimension (2D)
    if data_array_2d.ndim != 2:
        raise ValueError("Input array must be a 2D numpy array")
    # Calculate the 99th percentile value
    # percentile = np.percentile(data_array_2d, 99)
    # print(f'percentile: {percentile}')

    # Set values below 99th percentile to zero
    # data_array_2d[data_array_2d < percentile] = 0

    # In this program, the colormap is set to gray_r (reverse of the grayscale colormap), and 
    # the normalization is adjusted such that values greater than the 90th percentile appear darker in the colormap.
    # Create a colormap with increasing density of black for values greater than 90th percentile
    cmap = plt.cm.gray_r
    # norm = Normalize(vmin=0, vmax=percentile)

    # Create a custom heatmap plot
    plt.imshow(data_array_2d, cmap=cmap, norm=None, interpolation=None, origin='upper', aspect=None)
    # Set colorbar
    cbar = plt.colorbar()
    # cbar.set_label('Color Scale (0.0 to 1.0)')
    # Invert Y-axis to start from the top
    # plt.gca().invert_yaxis()
    # Set labels and title
    plt.xlabel(f'{col} (Col Idx : Y Axis)')
    plt.ylabel(f'{row} (Row Indices : X Axis)')
    plt.title(title)
    # Save the plot
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    # Show the plot (optional)
    # plt.show()create_pred_attn_map_2d
    # Close the plot
    plt.close()


if __name__ == '__main__':
    root_path = os.path.join('/scratch/pralaycs/Shubh_Working_Remote/PPI_Wkspc/PPI_Code/mat_p2ip_prj_working/')
    cnn_model_path = os.path.join(root_path, 'dataset/proc_data_tl_feat_to_img/img_p2ip_cnn/train/ResNet_tlStructEsmc_r400n18DnoWS')
    trx_model_path = os.path.join(root_path, 'dataset/proc_data_tl_feat_to_img/img_p2ip_trx/train/CDAMViT_tlStructEsmc_r400p16')
    
    trx_contrib_wt = 0.5  # transformer's attention-map weightage in interaction-map hybridization
    pct = 99.0  # percentile
    docking_version_lst = ['5_5']  # '5_5', '5_5'
    consider_full_lst = [True]  # True, False  # consider full-version or 99 percentile version of the interaction maps
    attr_mode_lst = ['total']  # 'total'
    
    # ############# IN FINAL VERSION OF PUBLIC CODE, REMOVE  "min_max_mode" and perform only 'R' mode ############# #
    min_max_mode_lst = ['N', 'R']  # 'N' => normal min-max for TP and reverse min-max for FN; 'R' => reverse min-max for TP and normal min-max for FN;
    # ############# IN FINAL VERSION OF PUBLIC CODE, REMOVE  "min_max_mode" and perform only 'R' mode ############# #
    
    # predicted interaction map process (pcmp) mode
    # S => Statistical Potential; C => Clustering; G => Graph smoothing; B => Binarization
    # 'SCGB' is recommended; Biophysics first, removes impossible interactions early. Clustering then removes scattered noise. 
    # Graph ensures smoothness. Adaptive threshold binarization keeps structure intact.
    # 'CSGB' works well if pred_map is highly noisy. Removes false positives early.
    pcmp_mode_lst = ['SCGB', 'CSGB']  # 'SCGB', 'CSGB'

    for docking_version in docking_version_lst:
        print('\n########## docking_version: ' + str(docking_version))
        for consider_full in consider_full_lst:
            print('\n########## consider_full: ' + str(consider_full))
            emd_res_df_lst = []  # For the docking-version specific consolidated EMD result dataframe
            for attr_mode in attr_mode_lst:
                print('\n########## attr_mode: ' + str(attr_mode))
                for min_max_mode in min_max_mode_lst:
                    print('\n########## min_max_mode: ' + str(min_max_mode))
                    for pcmp_mode in pcmp_mode_lst:
                        print('\n########## pcmp_mode: ' + str(pcmp_mode))
                        create_hyb_predicted_contact_map(root_path=root_path, cnn_model_path=cnn_model_path, trx_model_path=trx_model_path, docking_version=docking_version
                                                   , trx_contrib_wt=trx_contrib_wt, consider_full=consider_full, min_max_mode=min_max_mode, attr_mode=attr_mode, pcmp_mode=pcmp_mode)
                        emd_res_df = calculate_emd_betwn_gt_and_pred_hyb_contact_maps(root_path=root_path, docking_version=docking_version, consider_full=consider_full
                                                                                , min_max_mode=min_max_mode, attr_mode=attr_mode, pct=pct, pcmp_mode=pcmp_mode)
                        emd_res_df_lst.append(emd_res_df)
                    # End of for loop: for pcmp_mode in pcmp_mode_lst:
                # end of for loop: for min_max_mode in min_max_mode_lst:
            # end of for loop: for attr_mode in attr_mode_lst:
            # Create docking-version specific consolidated EMD result csv file
            create_consolidated_emd_res_csv(emd_res_df_lst=emd_res_df_lst, root_path=root_path, docking_version=docking_version, consider_full=consider_full)
        # end of for loop: for consider_full in consider_full_lst:
    # end of for loop: for docking_version in docking_version_lst:
