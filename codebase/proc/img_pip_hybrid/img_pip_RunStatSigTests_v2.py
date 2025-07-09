import sys, os
import pandas as pd
from pathlib import Path
path_root = Path(__file__).parents[2]  
sys.path.insert(0, str(path_root))
from utils import dl_reproducible_result_util
import numpy as np
import joblib
from mlxtend.evaluate import mcnemar_table, mcnemar
from utils import PPIPUtils


def exec_statSigTests_DS(root_path='./'): 
    spec_type_lst = ['human', 'ecoli', 'fly', 'mouse', 'worm', 'yeast']  
    statSigTests_res_ds_dict = {}
    resultsFolderName = os.path.join(root_path, 'dataset/proc_data_tl_feat_to_img/stat_sig_tests_res/ds')
    PPIPUtils.createFolder(resultsFolderName)
    for spec_type in spec_type_lst:
        statSigTests_res_ds_dict[spec_type] = {}
        img_pip_res_path = os.path.join(root_path, 'dataset/proc_data_tl_feat_to_img/img_pip_hybrid/img_pip_cnn_trx_hybrid', f'{spec_type}',  f'combined_df_{spec_type}.csv')
        img_pip_res_df = pd.read_csv(img_pip_res_path)
        y_target_arr = (img_pip_res_df['actual_res'].to_numpy()).astype(int)
        y_img_pip_pred_prob_arr = img_pip_res_df['hybrid_pred_prob_1'].to_numpy()
        y_img_pip_arr = (np.round(y_img_pip_pred_prob_arr, decimals=0)).astype(int)
        ds_res_path = os.path.join(root_path, 'dscript_full/cross_spec_pred_result_seq_len400', spec_type + '.tsv')
        ds_res_df = pd.read_csv(ds_res_path,  delimiter='\t', header=None, names=['prot1', 'prot2', 'ds_pred_prob'])
        y_ds_pred_prob_arr = ds_res_df['ds_pred_prob'].to_numpy()
        y_ds_arr = (np.round(y_ds_pred_prob_arr, decimals=0)).astype(int)
        cont_matrix = mcnemar_table(y_target=y_target_arr, y_model1=y_img_pip_arr, y_model2=y_ds_arr)
        statSigTests_res_ds_dict[spec_type]['cont_matrix'] = cont_matrix
        statistic, p = mcnemar(cont_matrix, exact=False, corrected=True)
        statistic = round(statistic, ndigits=4)
        statSigTests_res_ds_dict[spec_type]['statistic_chi_squared'] = statistic
        statSigTests_res_ds_dict[spec_type]['p'] = p
        alpha = 0.05  
        if p > alpha:
            statSigTests_res_ds_dict[spec_type]['H0_rejected'] = 'NO'
        else:
            statSigTests_res_ds_dict[spec_type]['H0_rejected'] = 'YES'
    statSigTests_res_ds_dict_file_path = os.path.join(resultsFolderName, 'statSigTests_res_ds_dict.pkl')
    joblib.dump(value=statSigTests_res_ds_dict, filename=statSigTests_res_ds_dict_file_path, compress=3)


def exec_statSigTests_matpip(root_path='./'): 
    spec_type_lst = ['human', 'ecoli', 'fly', 'mouse', 'worm', 'yeast']  
    statSigTests_res_matpip_dict = {}
    resultsFolderName = os.path.join(root_path, 'dataset/proc_data_tl_feat_to_img/stat_sig_tests_res/matpip')
    PPIPUtils.createFolder(resultsFolderName)
    for spec_type in spec_type_lst:
        statSigTests_res_matpip_dict[spec_type] = {}
        img_pip_res_path = os.path.join(root_path, 'dataset/proc_data_tl_feat_to_img/img_pip_hybrid/img_pip_cnn_trx_hybrid', f'{spec_type}',  f'combined_df_{spec_type}.csv')
        img_pip_res_df = pd.read_csv(img_pip_res_path)
        y_target_arr = (img_pip_res_df['actual_res'].to_numpy()).astype(int)
        y_img_pip_pred_prob_arr = img_pip_res_df['hybrid_pred_prob_1'].to_numpy()
        y_img_pip_arr = (np.round(y_img_pip_pred_prob_arr, decimals=0)).astype(int)
        matpip_res_path = os.path.join(root_path, 'dataset/proc_data_DS/', f'mat_res_origMan_auxTlOtherMan_{spec_type}_model_len400', f'pred_{spec_type}_DS.tsv')
        matpip_res_df = pd.read_csv(matpip_res_path,  delimiter='\t', header=None, names=['matpip_pred_prob', 'actual_res'])
        y_matpip_pred_prob_arr = matpip_res_df['matpip_pred_prob'].to_numpy()
        y_matpip_arr = (np.round(y_matpip_pred_prob_arr, decimals=0)).astype(int)
        cont_matrix = mcnemar_table(y_target=y_target_arr, y_model1=y_img_pip_arr, y_model2=y_matpip_arr)
        statSigTests_res_matpip_dict[spec_type]['cont_matrix'] = cont_matrix
        statistic, p = mcnemar(cont_matrix, exact=False, corrected=True)
        statistic = round(statistic, ndigits=4)
        statSigTests_res_matpip_dict[spec_type]['statistic_chi_squared'] = statistic
        statSigTests_res_matpip_dict[spec_type]['p'] = p
        alpha = 0.05  
        if p > alpha:
            statSigTests_res_matpip_dict[spec_type]['H0_rejected'] = 'NO'
        else:
            statSigTests_res_matpip_dict[spec_type]['H0_rejected'] = 'YES'
    statSigTests_res_matpip_dict_file_path = os.path.join(resultsFolderName, 'statSigTests_res_matpip_dict.pkl')
    joblib.dump(value=statSigTests_res_matpip_dict, filename=statSigTests_res_matpip_dict_file_path, compress=3)


if __name__ == '__main__':
    root_path = os.path.join('/project/root/directory/path/here')
    exec_statSigTests_matpip(root_path=root_path)
