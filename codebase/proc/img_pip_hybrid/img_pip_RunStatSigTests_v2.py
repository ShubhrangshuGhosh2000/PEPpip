import sys, os
import pandas as pd

from pathlib import Path
path_root = Path(__file__).parents[2]  # upto 'codebase' folder
sys.path.insert(0, str(path_root))
# print(sys.path)

from utils import dl_reproducible_result_util
import numpy as np
import joblib
from mlxtend.evaluate import mcnemar_table, mcnemar
from utils import PPIPUtils


def exec_statSigTests_DS(root_path='./'): 
    print('inside exec_statSigTests_DS() method - start')
    spec_type_lst = ['human', 'ecoli', 'fly', 'mouse', 'worm', 'yeast']  # human, ecoli, fly, mouse, worm, yeast
    
    # declare the dictionary containing the statistical significance test results for all the species
    statSigTests_res_ds_dict = {}
    resultsFolderName = os.path.join(root_path, 'dataset/proc_data_tl_feat_to_img/stat_sig_tests_res/ds')
    # create results folders if they do not exist
    PPIPUtils.createFolder(resultsFolderName)

    for spec_type in spec_type_lst:
        print('\n########## spec_type: ' + str(spec_type))
        statSigTests_res_ds_dict[spec_type] = {}
        # retrieve the correct target (class) labels
        img_pip_res_path = os.path.join(root_path, 'dataset/proc_data_tl_feat_to_img/img_pip_hybrid/img_pip_cnn_trx_hybrid', f'{spec_type}',  f'combined_df_{spec_type}.csv')
        
        # img_pip_res_df = pd.read_csv(img_pip_res_path,  delimiter='\t', header=None, names=['img_pip_pred_prob', 'label'])
        img_pip_res_df = pd.read_csv(img_pip_res_path)
        y_target_arr = (img_pip_res_df['actual_res'].to_numpy()).astype(int)

        # #### retrieve the class labels predicted by img_pip (model-1)
        y_img_pip_pred_prob_arr = img_pip_res_df['hybrid_pred_prob_1'].to_numpy()
        y_img_pip_arr = (np.round(y_img_pip_pred_prob_arr, decimals=0)).astype(int)

        # #### retrieve the class labels predicted by dscript (model-2)
        ds_res_path = os.path.join(root_path, 'dscript_full/cross_spec_pred_result_seq_len400', spec_type + '.tsv')
        ds_res_df = pd.read_csv(ds_res_path,  delimiter='\t', header=None, names=['prot1', 'prot2', 'ds_pred_prob'])
        y_ds_pred_prob_arr = ds_res_df['ds_pred_prob'].to_numpy()
        y_ds_arr = (np.round(y_ds_pred_prob_arr, decimals=0)).astype(int)

        # define contingency matrix. 
        # ############### Order of model-1 and model-2 in the figure given 'Example 1 - Creating 2x2 contingency tables'
        # in https://rasbt.github.io/mlxtend/user_guide/evaluate/mcnemar/ should be reversed (model-1 will be at the upper half 
        # and model-2 will be at the left). ########################## 
        cont_matrix = mcnemar_table(y_target=y_target_arr, y_model1=y_img_pip_arr, y_model2=y_ds_arr)
        print('cont_matrix: ' + str(cont_matrix))
        statSigTests_res_ds_dict[spec_type]['cont_matrix'] = cont_matrix
        # calculate mcnemar test
        statistic, p = mcnemar(cont_matrix, exact=False, corrected=True)
        # round the values upto 4 decimal places
        statistic = round(statistic, ndigits=4)
        # p = round(p, ndigits=5)
        statSigTests_res_ds_dict[spec_type]['statistic_chi_squared'] = statistic
        statSigTests_res_ds_dict[spec_type]['p'] = p
        # summarize the finding
        print('statistic:' + str(statistic) + ' : p-value: ' + str(p))
        # interpret the p-value
        alpha = 0.05  # significance level
        if p > alpha:
            print('Same proportions of errors (fail to reject H0)')
            statSigTests_res_ds_dict[spec_type]['H0_rejected'] = 'NO'
        else:
            print('Different proportions of errors (reject H0)')
            statSigTests_res_ds_dict[spec_type]['H0_rejected'] = 'YES'
    # end of for loop: for spec_type in spec_type_lst:
    print('Final statSigTests_res_ds_dict: ' + str(statSigTests_res_ds_dict))
    # save statSigTests_res_ds_dict as a pkl file
    statSigTests_res_ds_dict_file_path = os.path.join(resultsFolderName, 'statSigTests_res_ds_dict.pkl')
    joblib.dump(value=statSigTests_res_ds_dict, filename=statSigTests_res_ds_dict_file_path, compress=3)
    print("\n The statSigTests_res_ds_dict is saved as: " + statSigTests_res_ds_dict_file_path)
    print('inside exec_statSigTests_DS() method - end')


def exec_statSigTests_matpip(root_path='./'): 
    print('inside exec_statSigTests_matpip() method - start')
    spec_type_lst = ['human', 'ecoli', 'fly', 'mouse', 'worm', 'yeast']  # human, ecoli, fly, mouse, worm, yeast
    
    # declare the dictionary containing the statistical significance test results for all the species
    statSigTests_res_matpip_dict = {}
    resultsFolderName = os.path.join(root_path, 'dataset/proc_data_tl_feat_to_img/stat_sig_tests_res/matpip')
    # create results folders if they do not exist
    PPIPUtils.createFolder(resultsFolderName)

    for spec_type in spec_type_lst:
        print('\n########## spec_type: ' + str(spec_type))
        statSigTests_res_matpip_dict[spec_type] = {}
        # retrieve the correct target (class) labels
        img_pip_res_path = os.path.join(root_path, 'dataset/proc_data_tl_feat_to_img/img_pip_hybrid/img_pip_cnn_trx_hybrid', f'{spec_type}',  f'combined_df_{spec_type}.csv')
        
        # img_pip_res_df = pd.read_csv(img_pip_res_path,  delimiter='\t', header=None, names=['img_pip_pred_prob', 'label'])
        img_pip_res_df = pd.read_csv(img_pip_res_path)
        y_target_arr = (img_pip_res_df['actual_res'].to_numpy()).astype(int)

        # #### retrieve the class labels predicted by img_pip (model-1)
        y_img_pip_pred_prob_arr = img_pip_res_df['hybrid_pred_prob_1'].to_numpy()
        y_img_pip_arr = (np.round(y_img_pip_pred_prob_arr, decimals=0)).astype(int)

        # #### retrieve the class labels predicted by matpip (model-2)
        # mat_res_path = os.path.join(root_path, 'dataset/proc_data_e2e', 'mat_res_origMan_auxTlOtherMan_' + spec_type, 'pred_' + spec_type + '_DS.tsv')
        matpip_res_path = os.path.join(root_path, 'dataset/proc_data_DS/', f'mat_res_origMan_auxTlOtherMan_{spec_type}_model_len400', f'pred_{spec_type}_DS.tsv')
        
        matpip_res_df = pd.read_csv(matpip_res_path,  delimiter='\t', header=None, names=['matpip_pred_prob', 'actual_res'])
        y_matpip_pred_prob_arr = matpip_res_df['matpip_pred_prob'].to_numpy()
        y_matpip_arr = (np.round(y_matpip_pred_prob_arr, decimals=0)).astype(int)

        # define contingency matrix. 
        # ############### Order of model-1 and model-2 in the figure given 'Example 1 - Creating 2x2 contingency tables'
        # in https://rasbt.github.io/mlxtend/user_guide/evaluate/mcnemar/ should be reversed (model-1 will be at the upper half 
        # and model-2 will be at the left). ########################## 
        cont_matrix = mcnemar_table(y_target=y_target_arr, y_model1=y_img_pip_arr, y_model2=y_matpip_arr)
        print('cont_matrix: ' + str(cont_matrix))
        statSigTests_res_matpip_dict[spec_type]['cont_matrix'] = cont_matrix
        # calculate mcnemar test
        statistic, p = mcnemar(cont_matrix, exact=False, corrected=True)
        # round the values upto 4 decimal places
        statistic = round(statistic, ndigits=4)
        # p = round(p, ndigits=5)
        statSigTests_res_matpip_dict[spec_type]['statistic_chi_squared'] = statistic
        statSigTests_res_matpip_dict[spec_type]['p'] = p
        # summarize the finding
        print('statistic:' + str(statistic) + ' : p-value: ' + str(p))
        # interpret the p-value
        alpha = 0.05  # significance level
        if p > alpha:
            print('Same proportions of errors (fail to reject H0)')
            statSigTests_res_matpip_dict[spec_type]['H0_rejected'] = 'NO'
        else:
            print('Different proportions of errors (reject H0)')
            statSigTests_res_matpip_dict[spec_type]['H0_rejected'] = 'YES'
    # end of for loop: for spec_type in spec_type_lst:
    print('Final statSigTests_res_matpip_dict: ' + str(statSigTests_res_matpip_dict))
    # save statSigTests_res_matpip_dict as a pkl file
    statSigTests_res_matpip_dict_file_path = os.path.join(resultsFolderName, 'statSigTests_res_matpip_dict.pkl')
    joblib.dump(value=statSigTests_res_matpip_dict, filename=statSigTests_res_matpip_dict_file_path, compress=3)
    print("\n The statSigTests_res_matpip_dict is saved as: " + statSigTests_res_matpip_dict_file_path)
    print('inside exec_statSigTests_matpip() method - end')



if __name__ == '__main__':
    root_path = os.path.join('/project/root/directory/path/here')
    root_path = os.path.join('/scratch/pralaycs/Shubh_Working_Remote/PPI_Wkspc/PPI_Code/mat_p2ip_prj_working')  

    # ############ for Dscript #############
    # exec_statSigTests_DS(root_path=root_path)
    
    exec_statSigTests_matpip(root_path=root_path)

