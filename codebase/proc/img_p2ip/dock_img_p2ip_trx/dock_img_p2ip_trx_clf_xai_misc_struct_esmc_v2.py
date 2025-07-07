import os
import sys
from pathlib import Path

path_root = Path(__file__).parents[3]  # upto 'codebase' folder
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


def create_predicted_contact_map(root_path='./', model_path='./', docking_version='5_5', min_max_mode='R', attn_mode='total', pcmp_mode='SCGB'):
    print('\n #############################\n inside the create_predicted_contact_map() method - Start\n')
    print('\n########## docking_version: ' + str(docking_version))
    # The directory for the saved PDB files
    pdb_file_location = os.path.join(root_path, f'dataset/preproc_data_docking_BM_{docking_version}/pdb_files')
    # retrieve the docking_version specific concatenated prediction result
    print('retrieving the docking_version specific concatenated prediction result')
    test_tag = model_path.split('/')[-1]
    xai_result_dir = os.path.join(root_path, f'dataset/proc_data_tl_feat_to_img/img_p2ip_trx/xai_dock_{docking_version}/{test_tag}')
    concat_pred_res_file_nm_with_loc = os.path.join(xai_result_dir, f'concat_pred_res_dock_{docking_version}.csv')
    concat_pred_res_df = pd.read_csv(concat_pred_res_file_nm_with_loc)
    # separate true-positive (tp) and false-negative (fn) result
    print('separating true-positive (tp) and false-negative (fn) result')
    tp_df = concat_pred_res_df[ concat_pred_res_df['actual_res'] == concat_pred_res_df['pred_res']]
    tp_df = tp_df.reset_index(drop=True)
    fn_df = concat_pred_res_df[ concat_pred_res_df['actual_res'] != concat_pred_res_df['pred_res']]
    fn_df = fn_df.reset_index(drop=True)

    # create folders for the predicted interaction maps for the tp and fn result
    print('creating folders for the predicted interaction maps for the tp and fn result')
    tp_pred_contact_map_loc = os.path.join(xai_result_dir, 'tp_pred_contact_map', f'attnMode_{attn_mode}', f'minMaxMode_{min_max_mode}')
    PPIPUtils.createFolder(tp_pred_contact_map_loc)
    tp_pred_contact_map_proc_loc = os.path.join(xai_result_dir, 'tp_pred_contact_map_proc', f'attnMode_{attn_mode}', f'minMaxMode_{min_max_mode}', f'pcmp_mode_{pcmp_mode}')
    PPIPUtils.createFolder(tp_pred_contact_map_proc_loc)

    fn_pred_contact_map_loc = os.path.join(xai_result_dir, 'fn_pred_contact_map', f'attnMode_{attn_mode}', f'minMaxMode_{min_max_mode}')
    PPIPUtils.createFolder(fn_pred_contact_map_loc)
    fn_pred_contact_map_proc_loc = os.path.join(xai_result_dir, 'fn_pred_contact_map_proc', f'attnMode_{attn_mode}', f'minMaxMode_{min_max_mode}', f'pcmp_mode_{pcmp_mode}')
    PPIPUtils.createFolder(fn_pred_contact_map_proc_loc)

    postproc_attn_result_dir = os.path.join(xai_result_dir, 'attn_postproc')
    # create predicted interaction map for the tp cases
    print('\n creating predicted interaction map for the tp cases')
    # iterate over the tp_df and create predicted interaction map
    # for itr, row in tp_df.head(10).iterrows():
    for itr, row in tp_df.iterrows():
        print('\n#### tp: starting ' + str(itr) + '-th entry out of ' + str(tp_df.shape[0]-1) + ' ####\n')
        idx, prot1_id, prot2_id = row['idx'], row['prot1_id'], row['prot2_id']
        # fetch the attention map corresponding to idx
        postproc_res_dict_file_nm_with_loc = os.path.join(postproc_attn_result_dir, 'idx_' + str(idx) + '.pkl')
        postproc_res_dict = joblib.load(postproc_res_dict_file_nm_with_loc)
        # retrieve 'non_zero_pxl_indx_lst_lst' from postproc_res_dict
        non_zero_pxl_indx_lst_lst = postproc_res_dict['non_zero_pxl_indx_lst_lst']
        pxl_attn_2d_arr = postproc_res_dict['attn']
        
        # pxl_attn_lst = None
        # if(attn_mode == 'total'):
        #     # retrieve 'non_zero_pxl_tot_attn_lst' from postproc_res_dict
        #     pxl_attn_lst = postproc_res_dict['non_zero_pxl_tot_attn_lst']
        # else:
        #     # retrieve 'non_zero_pxl_attn_lst_lst' from postproc_res_dict
        #     pxl_attn_lst = postproc_res_dict['non_zero_pxl_attn_lst_lst']
        # # end of if-else block: if(attn_mode == 'total'):

        # create predicted 2d attention map
        print('tp: creating predicted 2d attention map')
        pred_attn_map_2d = create_pred_attn_map_2d(non_zero_pxl_indx_lst_lst, pxl_attn_2d_arr, attn_mode)
        print(f'tp: min_max_mode: {min_max_mode}')
        pred_contact_map = None
        min_val, max_val = pred_attn_map_2d.min(), pred_attn_map_2d.max()
        if(min_max_mode == 'N'):  # 'N' => normal min-max for TP and reverse min-max for FN;
            # create predicted interaction map for tp cases by performing an usual min-max normalization
            print('tp: creating predicted interaction map for tp cases by performing an usual min-max normalization')
            pred_contact_map = (pred_attn_map_2d - min_val) / (max_val - min_val)
        elif(min_max_mode == 'R'):  # 'R' => reverse min-max for TP and normal min-max for FN;
            # create predicted interaction map for tp cases by performing reversed min-max normalization
            print('tp: creating predicted interaction map for tp cases by performing reversed min-max normalization')
            pred_contact_map = (max_val - pred_attn_map_2d) / (max_val - min_val)
        # save pred_contact_map
        print('tp: saving pred_contact_map')
        protein_name, chain_1_name = prot1_id.split('_')
        protein_name, chain_2_name = prot2_id.split('_')
        print(f'protein_name: {protein_name} :: chain_1_name: {chain_1_name} :: chain_2_name: {chain_2_name} :: attnMode: {attn_mode}: minMaxMode: {min_max_mode} :: pcmp_mode: {pcmp_mode}')
        pred_contact_map_location = os.path.join(tp_pred_contact_map_loc, f"pred_contact_map_{protein_name}_{chain_1_name}_{chain_2_name}_attnMode_{attn_mode}_minMaxMode_{min_max_mode}.pkl")
        joblib.dump(pred_contact_map, pred_contact_map_location)
        print(f'tp: pred_contact_map is saved as {pred_contact_map_location}')
        # Postprocess the original pred_contact_map as per the pcmp_mode
        print(f"tp: Postprocess the original pred_contact_map as per the pcmp_mode: {pcmp_mode}")
        lambda_weight = 0.1  # λ is a hyperparameter controlling the weight of statistical potential incorporation.
        prot_contact_map_processor = ProteinContactMapUtils.ProteinContactMapProcessor(pdb_file_location=pdb_file_location, protein_name=protein_name
                                                                                       , chain_1_name=chain_1_name, chain_2_name=chain_2_name
                                                                                       , pred_contact_map=pred_contact_map, seq_order=pcmp_mode, lambda_weight=lambda_weight)
        pred_contact_map_proc = prot_contact_map_processor.process()
        pred_contact_map_proc_location = os.path.join(tp_pred_contact_map_proc_loc, f"pred_contact_map_{protein_name}_{chain_1_name}_{chain_2_name}_attnMode_{attn_mode}_minMaxMode_{min_max_mode}_pcmp_mode_{pcmp_mode}.pkl")
        joblib.dump(pred_contact_map_proc, pred_contact_map_proc_location)
        print(f'tp: pred_contact_map_proc is saved as {pred_contact_map_proc_location}')
    # end of for loop: for itr, row in tp_df.iterrows():
    # create predicted interaction map for the fn cases
    print('\n creating predicted interaction map for the fn cases')
    # iterate over the fn_df and create predicted interaction map
    for itr, row in fn_df.iterrows():
        print('\n#### fn: starting ' + str(itr) + '-th entry out of ' + str(fn_df.shape[0]-1) + ' ####\n')
        idx, prot1_id, prot2_id = row['idx'], row['prot1_id'], row['prot2_id']
        # fetch the attention map corresponding to idx
        postproc_res_dict_file_nm_with_loc = os.path.join(postproc_attn_result_dir, 'idx_' + str(idx) + '.pkl')
        postproc_res_dict = joblib.load(postproc_res_dict_file_nm_with_loc)
        # retrieve 'non_zero_pxl_indx_lst_lst' from postproc_res_dict
        non_zero_pxl_indx_lst_lst = postproc_res_dict['non_zero_pxl_indx_lst_lst']
        pxl_attn_2d_arr = postproc_res_dict['attn']
        
        # pxl_attn_lst = None
        # if(attn_mode == 'total'):
        #     # retrieve 'non_zero_pxl_tot_attn_lst' from postproc_res_dict
        #     pxl_attn_lst = postproc_res_dict['non_zero_pxl_tot_attn_lst']
        # else:
        #     # retrieve 'non_zero_pxl_attn_lst_lst' from postproc_res_dict
        #     pxl_attn_lst = postproc_res_dict['non_zero_pxl_attn_lst_lst']
        # # end of if-else block: if(attn_mode == 'total'):

        # create predicted 2d attention map
        print('fn: creating predicted 2d attention map')
        pred_attn_map_2d = create_pred_attn_map_2d(non_zero_pxl_indx_lst_lst, pxl_attn_2d_arr, attn_mode)
        print(f'fn: min_max_mode: {min_max_mode}')
        pred_contact_map = None
        min_val, max_val = pred_attn_map_2d.min(), pred_attn_map_2d.max()
        if(min_max_mode == 'N'):  # 'N' => normal min-max for TP and reverse min-max for FN;
            # create predicted interaction map for fn cases by performing reversed min-max normalization
            print('fn: creating predicted interaction map for fn cases by performing reversed min-max normalization')
            pred_contact_map = (max_val - pred_attn_map_2d) / (max_val - min_val)
        elif(min_max_mode == 'R'):  # 'R' => reverse min-max for TP and normal min-max for FN;
            # create predicted interaction map for fn cases by performing an usual min-max normalization
            print('fn: creating predicted interaction map for fn cases by performing an usual min-max normalization')
            pred_contact_map = (pred_attn_map_2d - min_val) / (max_val - min_val)
        # save pred_contact_map
        print('fn: saving pred_contact_map')
        protein_name, chain_1_name = prot1_id.split('_')
        protein_name, chain_2_name = prot2_id.split('_')
        print(f'protein_name: {protein_name} :: chain_1_name: {chain_1_name} :: chain_2_name: {chain_2_name} :: attnMode: {attn_mode} :: minMaxMode: {min_max_mode} :: pcmp_mode: {pcmp_mode}')
        pred_contact_map_location = os.path.join(fn_pred_contact_map_loc, f"pred_contact_map_{protein_name}_{chain_1_name}_{chain_2_name}_attnMode_{attn_mode}_minMaxMode_{min_max_mode}.pkl")
        joblib.dump(pred_contact_map, pred_contact_map_location)
        print(f'fn: pred_contact_map is saved as {pred_contact_map_location}')
        # Postprocess the original pred_contact_map as per the pcmp_mode
        print(f"fn: Postprocess the original pred_contact_map as per the pcmp_mode: {pcmp_mode}")
        lambda_weight = 0.3  # λ is a hyperparameter controlling the weight of statistical potential incorporation.
        prot_contact_map_processor = ProteinContactMapUtils.ProteinContactMapProcessor(pdb_file_location=pdb_file_location, protein_name=protein_name
                                                                                       , chain_1_name=chain_1_name, chain_2_name=chain_2_name
                                                                                       , pred_contact_map=pred_contact_map, seq_order=pcmp_mode, lambda_weight=lambda_weight)
        pred_contact_map_proc = prot_contact_map_processor.process()
        pred_contact_map_proc_location = os.path.join(fn_pred_contact_map_proc_loc, f"pred_contact_map_{protein_name}_{chain_1_name}_{chain_2_name}_attnMode_{attn_mode}_minMaxMode_{min_max_mode}_pcmp_mode_{pcmp_mode}.pkl")
        joblib.dump(pred_contact_map_proc, pred_contact_map_proc_location)
        print(f'tp: pred_contact_map_proc is saved as {pred_contact_map_proc_location}')
    # end of for loop: for itr, row in fn_df.iterrows():
    print('\n #############################\n inside the create_predicted_contact_map() method - End\n')


def create_pred_attn_map_2d(non_zero_pxl_indx_lst_lst, pxl_attn_2d_arr, attn_mode):
    """
    Create a 2D numpy array of attentions based on the given lists.

    Parameters:
    - non_zero_pxl_indx_lst_lst (list): A list of lists where each inner list contains the non-zero pixel index (H,W) i.e. 2 integers

    Returns:
    - numpy.ndarray: A 2D numpy array initiated with zeros and populated based on input argument lists.

    Example:
    >>> lst1 = [[3, 1], [4, 2], [2, 0], [5, 2]]
    >>> lst2 = [0.1, 0.2, 0.3, 0.4]
    >>> create_pred_attn_map_2d(lst1, lst2)
    array([[0. , 0. , 0. ],
           [0. , 0. , 0. ],
           [0.3, 0. , 0. ],
           [0. , 0.1, 0. ],
           [0. , 0., 0.2 ],
           [0. , 0., 0.4 ]])
    """
    # Find 'h_max' and 'w_max' from the input non_zero_pxl_indx_lst_lst
    h_max = max(non_zero_pxl_indx_lst_lst, key=lambda x: x[0])[0]
    w_max = max(non_zero_pxl_indx_lst_lst, key=lambda x: x[1])[1]

    # Create a numpy array initialized with zeros of dimensions (h_max + 1) by (w_max + 1)
    pred_attn_map_2d = np.zeros((h_max+1, w_max+1))

    # Iterate over non_zero_pxl_indx_lst_lst and populate pred_attn_map_2d based on non_zero_pxl_attn_lst
    for i in range(len(non_zero_pxl_indx_lst_lst)):
        x, y = non_zero_pxl_indx_lst_lst[i]
        if(attn_mode == 'total'):
            pred_attn_map_2d[x, y] = pxl_attn_2d_arr[x, y]
    # end of for loop: for i in range(len(non_zero_pxl_indx_lst_lst)):
    return pred_attn_map_2d


def calculate_emd_betwn_gt_and_pred_contact_maps(root_path='./', model_path='./', docking_version='5_5'
                                                 , consider_full=False, min_max_mode='R', attn_mode='total'
                                                 , pct=99.0, pcmp_mode='SCGB'):
    print('\n #############################\n inside the calculate_emd_betwn_gt_and_pred_contact_maps() method - Start\n')
    print('\n########## docking_version: ' + str(docking_version))
    gt_contact_map_dir = os.path.join(root_path, f"dataset/preproc_data_docking_BM_{docking_version}/prot_gt_contact_map")
    test_tag = model_path.split('/')[-1]
    xai_result_dir = os.path.join(root_path, f'dataset/proc_data_tl_feat_to_img/img_p2ip_trx/xai_dock_{docking_version}/{test_tag}')
    tp_pred_contact_map_proc_loc = os.path.join(xai_result_dir, 'tp_pred_contact_map_proc', f'attnMode_{attn_mode}', f'minMaxMode_{min_max_mode}', f'pcmp_mode_{pcmp_mode}')
    fn_pred_contact_map_proc_loc = os.path.join(xai_result_dir, 'fn_pred_contact_map_proc', f'attnMode_{attn_mode}', f'minMaxMode_{min_max_mode}', f'pcmp_mode_{pcmp_mode}')
    emd_result_dir = os.path.join(xai_result_dir, 'emd_result')
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
    # map and the predicted interaction map
    dock_test_row_idx_lst, prot_1_id_lst, prot_2_id_lst, attn_mode_lst, min_max_mode_lst, tp_fn_lst, emd_lst, pcmp_mode_lst = [], [], [], [], [], [], [], []  # lists to create a dataframe later
    aupr_lst, auroc_lst, specificity_lst, precision_lst, recall_lst, f1_score_lst = [], [], [], [], [], []  # lists to create a dataframe later
    tp_cont_lst, fp_cont_lst, fn_cont_lst, tn_cont_lst = [], [], [], []  # lists to create a dataframe later

    for index, row in dock_test_df.iterrows():
        print(f"\n ################# starting {index}-th row out of {dock_test_df.shape[0]-1}\n")
        prot_1_id, prot_2_id = row['prot_1_id'], row['prot_2_id']
        # protein id has the format of [protein_name]_[chain_name]
        protein_name, chain_1_name = prot_1_id.split('_')
        protein_name, chain_2_name = prot_2_id.split('_')
        print(f"protein_name: {protein_name} :: chain_1_name: {chain_1_name} :: chain_2_name: {chain_2_name} :: attnMode: {attn_mode} :: minMaxMode: {min_max_mode} :: pcmp_mode: {pcmp_mode}")
        # retrieve the pred_contact_map
        print('retrieving the pred_contact_map...')
        # first check whether it belongs to tp_pred_contact_map_proc_loc
        tp_pred_contact_map_proc_path = os.path.join(tp_pred_contact_map_proc_loc, f"pred_contact_map_{protein_name}_{chain_1_name}_{chain_2_name}_attnMode_{attn_mode}_minMaxMode_{min_max_mode}_pcmp_mode_{pcmp_mode}.pkl")
        pred_contact_map_lst = glob.glob(tp_pred_contact_map_proc_path, recursive=False)
        if(len(pred_contact_map_lst) == 0):
            print('No pred_contact_map found in tp_pred_contact_map_proc folder and next searching in fn_pred_contact_map_proc folder...')
            # if the desired pred_contact_map does not exist inside tp_pred_contact_map_proc_path, then
            # search inside fn_pred_contact_map_proc_path for the same
            fn_pred_contact_map_proc_path = os.path.join(fn_pred_contact_map_proc_loc, f"pred_contact_map_{protein_name}_{chain_1_name}_{chain_2_name}_attnMode_{attn_mode}_minMaxMode_{min_max_mode}_pcmp_mode_{pcmp_mode}.pkl")
            pred_contact_map_lst = glob.glob(fn_pred_contact_map_proc_path, recursive=False)
            if(len(pred_contact_map_lst) == 0):
                raise Exception(f"No pred_contact_map found in tp_pred_contact_map_proc and fn_pred_contact_map_proc folders for \
                                protein_name: {protein_name} :: chain_1_name: {chain_1_name} :: chain_2_name: {chain_2_name}")
            else:  
                # desired pred_contact_map exists inside fn_pred_contact_map_proc folder
                print('desired pred_contact_map exists inside fn_pred_contact_map_proc folder')
                tp_fn_lst.append('fn')
            # end of else block
        else:
            print('desired pred_contact_map exists inside tp_pred_contact_map_proc folder')
            tp_fn_lst.append('tp')
        # end of if-else block: if(len(pred_contact_map_lst) == 0):
        pred_contact_map_name = pred_contact_map_lst[0].split('/')[-1]
        pcmp_mode = pred_contact_map_name.split('.')[0].split('_')[-1]  # f"pred_contact_map_{protein_name}_{chain_1_name}_{chain_2_name}_attnMode_{attn_mode}_minMaxMode_{min_max_mode}_pcmp_mode_{pcmp_mode}.pkl"
        print(f'pcmp_mode: {pcmp_mode}')
        pred_contact_map = joblib.load(pred_contact_map_lst[0])

        # retrieve the gt_contact_map
        print('retrieving the gt_contact_map...')
        gt_contact_map_location = os.path.join(gt_contact_map_dir, f"contact_map_{protein_name}_{chain_1_name}_{chain_2_name}.pkl")
        gt_contact_map = joblib.load(gt_contact_map_location)

        # check whether dimensionally both the interaction maps are same
        if(pred_contact_map.shape == gt_contact_map.shape):
           print('dimensionally both the interaction maps (pred_contact_map and gt_contact_map) are same')
        else:
            print(f'!!!!! dimensionally both the interaction maps (pred_contact_map and gt_contact_map) are not the same')
            print(f"pred_contact_map.shape: {pred_contact_map.shape} :: gt_contact_map.shape: {gt_contact_map.shape}")
            raise Exception(f"Dimensionally both the interaction maps (pred_contact_map and gt_contact_map) are not the same for \
                                protein_name: {protein_name} :: chain_1_name: {chain_1_name} :: chain_2_name: {chain_2_name} \
                                pred_contact_map.shape: {pred_contact_map.shape} :: gt_contact_map.shape: {gt_contact_map.shape}")
        # calculate the EMD between pred_contact_map and gt_contact_map
        print('Calculating the EMD between pred_contact_map and gt_contact_map')
        if(not consider_full):
            # Calculate the 9X-th percentile value for pred_contact_map
            percentile = np.percentile(pred_contact_map, pct)
            # Set values below 9X-th percentile to zero
            pred_contact_map[pred_contact_map <= percentile] = 0
            # ##### Set values above 9X-th percentile to one for plot generation purpose
            pred_contact_map[pred_contact_map > percentile] = 1

            # # Calculate the 9X-th percentile value for gt_contact_map
            # percentile = np.percentile(gt_contact_map, 99)
            # # Set values below 9X-th percentile to zero
            # gt_contact_map[gt_contact_map <= percentile] = 0

        emd = DockUtils.calculateEMD(pred_contact_map, gt_contact_map)
        print(f'emd: {emd}')
         # calculate the absolute difference between pred_contact_map and gt_contact_map
        # abs_diff = DockUtils.calcMatrixAbsDiff(pred_contact_map, gt_contact_map)

        # Calculate all the performance metrics comparing pred_contact_map and gt_contact_map
        perf_metric_dict = DockUtils.evaluate_contact_maps(pred_contact_map, gt_contact_map, distance_metric='cityblock')
        print(f'perf_metric_dict: {perf_metric_dict}')

        # generate contact heatmap plot for pred_contact_map
        print('generating contact heatmap plot for pred_contact_map')
        pred_contact_map_plot_path = os.path.join(contact_heatmap_plt_dir, f'prot_{prot_1_id}_prot_{prot_2_id}_{tp_fn_lst[-1]}_attnMode_{attn_mode}_minMaxMode_{min_max_mode}_pcmp_mode_{pcmp_mode}_pred_contact_map.png')
        generate_pred_contact_heatmap_plot(pred_contact_map, title='Predicted interaction map', row = prot_1_id, col=prot_2_id, save_path=pred_contact_map_plot_path)

        # generate contact heatmap plot for gt_contact_map
        print('generating contact heatmap plot for gt_contact_map')
        gt_contact_map_plot_path = os.path.join(contact_heatmap_plt_dir, f'prot_{prot_1_id}_prot_{prot_2_id}_gt_contact_map.png')
        generate_gt_contact_heatmap_plot(gt_contact_map, title='gt interaction map', row = prot_1_id, col=prot_2_id, save_path=gt_contact_map_plot_path)

        # populate the lists required to create a df later
        dock_test_row_idx_lst.append(index); prot_1_id_lst.append(prot_1_id); prot_2_id_lst.append(prot_2_id)
        attn_mode_lst.append(attn_mode); min_max_mode_lst.append(min_max_mode); pcmp_mode_lst.append(pcmp_mode); emd_lst.append(emd)
    
        aupr_lst.append(perf_metric_dict['aupr']); auroc_lst.append(perf_metric_dict['auroc']); specificity_lst.append(perf_metric_dict['specificity'])
        precision_lst.append(perf_metric_dict['precision']); recall_lst.append(perf_metric_dict['recall']); f1_score_lst.append(perf_metric_dict['f1_score'])
        tp_cont_lst.append(perf_metric_dict['conf_matrix_dict']['tp']); fp_cont_lst.append(perf_metric_dict['conf_matrix_dict']['fp'])
        fn_cont_lst.append(perf_metric_dict['conf_matrix_dict']['fn']); tn_cont_lst.append(perf_metric_dict['conf_matrix_dict']['tn'])
    # end of for loop: for index, row in dock_test_df.iterrows():
    # create the dataframe and save it
    emd_res_df = pd.DataFrame({'dock_test_row_idx': dock_test_row_idx_lst, 'prot_1_id': prot_1_id_lst, 'prot_2_id': prot_2_id_lst, 'min_max_mode': min_max_mode_lst
                           , 'attn_mode': attn_mode_lst, 'tp_fn': tp_fn_lst, 'pcmp_mode': pcmp_mode, 'emd': emd_lst
                           , 'aupr': aupr_lst, 'auroc': auroc_lst, 'specificity': specificity_lst
                           , 'precision': precision_lst, 'recall': recall_lst, 'f1_score': f1_score_lst
                           , 'tp_cont': tp_cont_lst, 'fp_cont': fp_cont_lst, 'fn_cont': fn_cont_lst, 'tn_cont': tn_cont_lst})
    print('\n #################\n inside the calculate_emd_betwn_gt_and_pred_contact_maps() method - End\n')
    return emd_res_df


def create_consolidated_emd_res_csv(emd_res_df_lst=None, root_path='./', model_path='./', docking_version='5_5'
                                                 , consider_full=False):
    print('\n #############################\n inside the create_consolidated_emd_res_csv() method - Start\n')
    print('\n########## docking_version: ' + str(docking_version))
    test_tag = model_path.split('/')[-1]
    xai_result_dir = os.path.join(root_path, f'dataset/proc_data_tl_feat_to_img/img_p2ip_trx/xai_dock_{docking_version}/{test_tag}')
    emd_result_dir = os.path.join(xai_result_dir, 'emd_result')
    PPIPUtils.createFolder(emd_result_dir)
    
    # create the dataframe and save it
    consolidated_emd_res_df = pd.concat(emd_res_df_lst)
    # save emd_res_df
    if(consider_full):
        # consider full-version version of the interaction maps
        consolidated_emd_res_df.to_csv(os.path.join(emd_result_dir, f'gt_vs_pred_emd_full.csv'), index=False)
    else:
        # consider 99 percentile version of the interaction maps
        consolidated_emd_res_df.to_csv(os.path.join(emd_result_dir, f'gt_vs_pred_emd_9Xp.csv'), index=False)

    # Discard the redundant columns
    consolidated_emd_res_df = consolidated_emd_res_df.drop(columns=['dock_test_row_idx', 'prot_1_id', 'prot_2_id'])

    # Group by tp_fn, attn_mode, and attn_mode
    grouped_df = consolidated_emd_res_df.groupby(['tp_fn', 'attn_mode', 'min_max_mode', 'pcmp_mode']).agg(
                                aupr_avg=('aupr', 'mean'), auroc_avg=('auroc', 'mean'), specif_avg=('specificity', 'mean'),
                                prec_avg=('precision', 'mean'), recall_avg=('recall', 'mean'), f1_avg=('f1_score', 'mean'),
                                tp_cont_sum=('tp_cont', 'sum'), fp_cont_sum=('fp_cont', 'sum'), fn_cont_sum=('fn_cont', 'sum'), tn_cont_sum=('tn_cont', 'sum')
                            ).reset_index()

    grouped_df['aupr_avg'] = grouped_df['aupr_avg'].round(3); grouped_df['auroc_avg'] = grouped_df['auroc_avg'].round(3); grouped_df['specif_avg'] = grouped_df['specif_avg'].round(3)
    grouped_df['prec_avg'] = grouped_df['prec_avg'].round(3); grouped_df['recall_avg'] = grouped_df['recall_avg'].round(3); grouped_df['f1_avg'] = grouped_df['f1_avg'].round(3)

    if(consider_full):
        # consider full-version version of the interaction maps
        grouped_df.to_csv(os.path.join(emd_result_dir, f'overall_eval_res_full.csv'), index=False)
    else:
        # Save the grouped dataframe as a CSV file
        grouped_df.to_csv(os.path.join(emd_result_dir, f'overall_eval_res_9Xp.csv'), index=False)

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
    model_path = os.path.join(root_path, 'dataset/proc_data_tl_feat_to_img/img_p2ip_trx/train/CDAMViT_tlStructEsmc_r400p16')
    # partial_model_name = 'ImgP2ipTrx'

    pct = 99.0  # percentile
    docking_version_lst = ['5_5']  # '5_5', '5_5'
    consider_full_lst = [True]  # True, False  # consider full-version or 99 percentile version of the interaction maps
    attn_mode_lst = ['total']  # 'total'
    
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
            for attn_mode in attn_mode_lst:
                print('\n########## attn_mode: ' + str(attn_mode))
                for min_max_mode in min_max_mode_lst:
                    print('\n########## min_max_mode: ' + str(min_max_mode))
                    for pcmp_mode in pcmp_mode_lst:
                        print('\n########## pcmp_mode: ' + str(pcmp_mode))
                        create_predicted_contact_map(root_path=root_path, model_path=model_path, docking_version=docking_version, min_max_mode=min_max_mode, attn_mode=attn_mode, pcmp_mode=pcmp_mode)
                        emd_res_df = calculate_emd_betwn_gt_and_pred_contact_maps(root_path=root_path, model_path=model_path, docking_version=docking_version, consider_full=consider_full
                                                                                , min_max_mode=min_max_mode, attn_mode=attn_mode, pct=pct, pcmp_mode=pcmp_mode)
                        emd_res_df_lst.append(emd_res_df)
                    # End of for loop: for pcmp_mode in pcmp_mode_lst:
                # end of for loop: for min_max_mode in min_max_mode_lst:
            # end of for loop: for attn_mode in attn_mode_lst:
            # Create docking-version specific consolidated EMD result csv file
            create_consolidated_emd_res_csv(emd_res_df_lst=emd_res_df_lst, root_path=root_path, model_path=model_path, docking_version=docking_version, consider_full=consider_full)
        # end of for loop: for consider_full in consider_full_lst:
    # end of for loop: for docking_version in docking_version_lst:
