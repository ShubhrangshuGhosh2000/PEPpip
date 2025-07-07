import sys, os
import pandas as pd

from pathlib import Path
path_root = Path(__file__).parents[3]  # upto 'MAT_P2IP_PRJ' folder
sys.path.insert(0, str(path_root))
# print(sys.path)

from codebase.utils import dl_reproducible_result_util
from codebase.utils import PPIPUtils
import numpy as np
import joblib
from mlxtend.evaluate import mcnemar_table, mcnemar


def exec_statSigTests_DS(root_path='./'): 
    print('inside exec_statSigTests_DS() method - start')
    spec_type_lst = ['ecoli', 'fly', 'mouse', 'worm', 'yeast']  # human, ecoli, fly, mouse, worm, yeast
    # declare the dictionary containing the statistical significance test results for all the species
    statSigTests_res_ds_dict = {}
    resultsFolderName = os.path.join(root_path, 'dataset/proc_data_e2e/stat_sig_tests_res/ds')
    # create results folders if they do not exist
    PPIPUtils.makeDir(resultsFolderName)

    for spec_type in spec_type_lst:
        print('\n########## spec_type: ' + str(spec_type))
        statSigTests_res_ds_dict[spec_type] = {}
        # retrieve the correct target (class) labels
        mat_res_path = os.path.join(root_path, 'dataset/proc_data_e2e', 'mat_res_origMan_auxTlOtherMan_' + spec_type, 'pred_' + spec_type + '_DS.tsv')
        mat_res_df = pd.read_csv(mat_res_path,  delimiter='\t', header=None, names=['mat_pred_prob', 'label'])
        y_target_arr = (mat_res_df['label'].to_numpy()).astype(int)

        # #### retrieve the class labels predicted by mat_pip (model-1)
        y_mat_pred_prob_arr = mat_res_df['mat_pred_prob'].to_numpy()
        y_mat_pip_arr = (np.round(y_mat_pred_prob_arr, decimals=0)).astype(int)

        # #### retrieve the class labels predicted by dscript (model-2)
        ds_res_path = os.path.join(root_path, 'dscript_full/cross_spec_pred_result', spec_type + '.tsv')
        ds_res_df = pd.read_csv(ds_res_path,  delimiter='\t', header=None, names=['prot1', 'prot2', 'ds_pred_prob'])
        y_ds_pred_prob_arr = ds_res_df['ds_pred_prob'].to_numpy()
        y_ds_arr = (np.round(y_ds_pred_prob_arr, decimals=0)).astype(int)

        # define contingency matrix. 
        # ############### Order of model-1 and model-2 in the figure given 'Example 1 - Creating 2x2 contingency tables'
        # in https://rasbt.github.io/mlxtend/user_guide/evaluate/mcnemar/ should be reversed (model-1 will be at the upper half 
        # and model-2 will be at the left). ########################## 
        cont_matrix = mcnemar_table(y_target=y_target_arr, y_model1=y_mat_pip_arr, y_model2=y_ds_arr)
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


def exec_statSigTests_humanBenchmark(root_path='./', other_algo_result_path='./'): 
    print('inside exec_statSigTests_humanBenchmark() method - start')
    resultsFolderName = os.path.join(root_path, 'dataset/proc_data_e2e/stat_sig_tests_res/human_benchmark')
    # create results folders if they do not exist
    PPIPUtils.makeDir(resultsFolderName)
    human_dataset_types_lst = ['50_pos_full', '10_pos_full', '03_pos_full', '50_pos_held', '10_pos_held', '03_pos_held']
    # human_dataset_types_lst = ['50_pos_full', '10_pos_full', '03_pos_full']
    # declare the dictionary containing the statistical significance test results for the different human dataset types
    statSigTests_res_humanBenchmark_dict = {}

    for human_dataset_type in human_dataset_types_lst:
        print('\n########## human_dataset_type: ' + str(human_dataset_type))
        statSigTests_res_humanBenchmark_dict[human_dataset_type] = {}

        if(human_dataset_type == '50_pos_full'):
            con_y_target_arr, con_y_mat_pip_arr, con_y_other_algo_arr = retrieve_50_pos_full_res(root_path, other_algo_result_path)
        elif(human_dataset_type == '10_pos_full'):
            con_y_target_arr, con_y_mat_pip_arr, con_y_other_algo_arr = retrieve_10_pos_full_res(root_path, other_algo_result_path)
        elif(human_dataset_type == '03_pos_full'):
            con_y_target_arr, con_y_mat_pip_arr, con_y_other_algo_arr = retrieve_03_pos_full_res(root_path, other_algo_result_path)
        elif(human_dataset_type == '50_pos_held'):
            con_y_target_arr, con_y_mat_pip_arr, con_y_other_algo_arr = retrieve_50_pos_held_res(root_path, other_algo_result_path)
        elif(human_dataset_type == '10_pos_held'):
            con_y_target_arr, con_y_mat_pip_arr, con_y_other_algo_arr = retrieve_10_pos_held_res(root_path, other_algo_result_path)
        elif(human_dataset_type == '03_pos_held'):
            con_y_target_arr, con_y_mat_pip_arr, con_y_other_algo_arr = retrieve_03_pos_held_res(root_path, other_algo_result_path)
        
        # define contingency matrix. 
        # ############### Order of model-1 and model-2 in the figure given 'Example 1 - Creating 2x2 contingency tables'
        # in https://rasbt.github.io/mlxtend/user_guide/evaluate/mcnemar/ should be reversed (model-1 will be at the upper half 
        # and model-2 will be at the left). ########################## 
        cont_matrix = mcnemar_table(y_target=con_y_target_arr, y_model1=con_y_mat_pip_arr, y_model2=con_y_other_algo_arr)
        print('cont_matrix: ' + str(cont_matrix))
        statSigTests_res_humanBenchmark_dict[human_dataset_type]['cont_matrix'] = cont_matrix
        # calculate mcnemar test
        statistic, p = mcnemar(cont_matrix, exact=False, corrected=True)
        # round the values upto 4 decimal places
        statistic = round(statistic, ndigits=4)
        # p = round(p, ndigits=5)
        statSigTests_res_humanBenchmark_dict[human_dataset_type]['statistic_chi_squared'] = statistic
        statSigTests_res_humanBenchmark_dict[human_dataset_type]['p'] = p
        # summarize the finding
        print('statistic:' + str(statistic) + ' : p-value: ' + str(p))
        # interpret the p-value
        alpha = 0.05  # significance level
        if p > alpha:
            print('Same proportions of errors (fail to reject H0)')
            statSigTests_res_humanBenchmark_dict[human_dataset_type]['H0_rejected'] = 'NO'
        else:
            print('Different proportions of errors (reject H0)')
            statSigTests_res_humanBenchmark_dict[human_dataset_type]['H0_rejected'] = 'YES'
    # end of for loop: for human_dataset_type in human_dataset_types_lst:
    print('Final statSigTests_res_humanBenchmark_dict: ' + str(statSigTests_res_humanBenchmark_dict))
    # save statSigTests_res_humanBenchmark_dict as a pkl file
    statSigTests_res_humanBenchmark_dict_file_path = os.path.join(resultsFolderName, 'statSigTests_res_humanBenchmark_dict.pkl')
    joblib.dump(value=statSigTests_res_humanBenchmark_dict, filename=statSigTests_res_humanBenchmark_dict_file_path, compress=3)
    print("\n The statSigTests_res_humanBenchmark_dict is saved as: " + statSigTests_res_humanBenchmark_dict_file_path)
    print('inside exec_statSigTests_humanBenchmark() method - end')


def retrieve_50_pos_full_res(oot_path='./', other_algo_result_path='./'):
    print('inside retrieve_50_pos_full_res() method - start')
    mat_pip_res_path = os.path.join(root_path, 'dataset/proc_data/mat_res/mat_res_origMan_auxTlOtherMan')
    y_target_lst, y_mat_pip_lst, y_other_algo_lst = [], [], []

    for i in range(0,5):
        # ############# for mat_pip algo -start #############
        mat_pip_res_file_nm = os.path.join(mat_pip_res_path, 'R50_'+str(i)+'_predict.tsv')
        mat_res_df = pd.read_csv(mat_pip_res_file_nm,  delimiter='\t', header=None, names=['mat_pred_prob', 'label'])
        # retrieve the correct target (class) labels
        y_target_arr = (mat_res_df['label'].to_numpy()).astype(int)
        y_target_lst += y_target_arr.tolist()
        # retrieve the class labels predicted by mat_pip (model-1)
        y_mat_pred_prob_arr = mat_res_df['mat_pred_prob'].to_numpy()
        y_mat_pip_arr = (np.round(y_mat_pred_prob_arr, decimals=0)).astype(int)
        y_mat_pip_lst += y_mat_pip_arr.tolist()
        # ############# for mat_pip algo -end #############

        # ############# for other  algo -start #############
        other_algo_res_file_nm = os.path.join(other_algo_result_path, 'R50_'+str(i)+'_predict.tsv')
        other_algo_res_df = pd.read_csv(other_algo_res_file_nm,  delimiter='\t', header=None, names=['other_algo_pred_prob', 'label'])
        # retrieve the correct target (class) labels
        other_algo_y_target_arr = (other_algo_res_df['label'].to_numpy()).astype(int)
        # check the equality with y_target_arr from mat_pip
        same = np.array_equal(y_target_arr, other_algo_y_target_arr)
        if(not same):
            print('######### ERROR!! ERROR!! y_target_arr and other_algo_y_target_arr not same for R50_'+str(i)+'_predict.tsv')
            sys.exit(0)
        # retrieve the class labels predicted by other_algo (model-2)
        y_other_algo_pred_prob_arr = other_algo_res_df['other_algo_pred_prob'].to_numpy()
        y_other_algo_arr = (np.round(y_other_algo_pred_prob_arr, decimals=0)).astype(int)
        y_other_algo_lst += y_other_algo_arr.tolist()
        # ############# for other  algo -end #############
    # end of for loop: for i in range(0,5):
    print('inside retrieve_50_pos_full_res() method - end')
    return (np.array(y_target_lst), np.array(y_mat_pip_lst), np.array(y_other_algo_lst))


def retrieve_10_pos_full_res(oot_path='./', other_algo_result_path='./'):
    print('inside retrieve_10_pos_full_res() method - start')
    mat_pip_res_path = os.path.join(root_path, 'dataset/proc_data/mat_res/mat_res_origMan_auxTlOtherMan')
    y_target_lst, y_mat_pip_lst, y_other_algo_lst = [], [], []

    for i in range(0,5):
        # ############# for mat_pip algo -start #############
        mat_pip_res_file_nm = os.path.join(mat_pip_res_path, 'R20_'+str(i)+'_predict1.tsv')
        mat_res_df = pd.read_csv(mat_pip_res_file_nm,  delimiter='\t', header=None, names=['mat_pred_prob', 'label'])
        # retrieve the correct target (class) labels
        y_target_arr = (mat_res_df['label'].to_numpy()).astype(int)
        y_target_lst += y_target_arr.tolist()
        # retrieve the class labels predicted by mat_pip (model-1)
        y_mat_pred_prob_arr = mat_res_df['mat_pred_prob'].to_numpy()
        y_mat_pip_arr = (np.round(y_mat_pred_prob_arr, decimals=0)).astype(int)
        y_mat_pip_lst += y_mat_pip_arr.tolist()
        # ############# for mat_pip algo -end #############

        # ############# for other algo -start #############
        other_algo_res_file_nm = os.path.join(other_algo_result_path, 'R20_'+str(i)+'_predict1.tsv')
        other_algo_res_df = pd.read_csv(other_algo_res_file_nm,  delimiter='\t', header=None, names=['other_algo_pred_prob', 'label'])
        # retrieve the correct target (class) labels
        other_algo_y_target_arr = (other_algo_res_df['label'].to_numpy()).astype(int)
        # check the equality with y_target_arr from mat_pip
        same = np.array_equal(y_target_arr, other_algo_y_target_arr)
        if(not same):
            print('######### ERROR!! ERROR!! y_target_arr and other_algo_y_target_arr not same for R20_'+str(i)+'_predict1.tsv')
            sys.exit(0)
        # retrieve the class labels predicted by other_algo (model-2)
        y_other_algo_pred_prob_arr = other_algo_res_df['other_algo_pred_prob'].to_numpy()
        y_other_algo_arr = (np.round(y_other_algo_pred_prob_arr, decimals=0)).astype(int)
        y_other_algo_lst += y_other_algo_arr.tolist()
        # ############# for other  algo -end #############
    # end of for loop: for i in range(0,5):
    print('inside retrieve_10_pos_full_res() method - end')
    return (np.array(y_target_lst), np.array(y_mat_pip_lst), np.array(y_other_algo_lst))


def retrieve_03_pos_full_res(oot_path='./', other_algo_result_path='./'):
    print('inside retrieve_03_pos_full_res() method - start')
    mat_pip_res_path = os.path.join(root_path, 'dataset/proc_data/mat_res/mat_res_origMan_auxTlOtherMan')
    y_target_lst, y_mat_pip_lst, y_other_algo_lst = [], [], []

    for i in range(0,5):
        # ############# for mat_pip algo -start #############
        mat_pip_res_file_nm = os.path.join(mat_pip_res_path, 'R20_'+str(i)+'_predict2.tsv')
        mat_res_df = pd.read_csv(mat_pip_res_file_nm,  delimiter='\t', header=None, names=['mat_pred_prob', 'label'])
        # retrieve the correct target (class) labels
        y_target_arr = (mat_res_df['label'].to_numpy()).astype(int)
        y_target_lst += y_target_arr.tolist()
        # retrieve the class labels predicted by mat_pip (model-1)
        y_mat_pred_prob_arr = mat_res_df['mat_pred_prob'].to_numpy()
        y_mat_pip_arr = (np.round(y_mat_pred_prob_arr, decimals=0)).astype(int)
        y_mat_pip_lst += y_mat_pip_arr.tolist()
        # ############# for mat_pip algo -end #############

        # ############# for other  algo -start #############
        other_algo_res_file_nm = os.path.join(other_algo_result_path, 'R20_'+str(i)+'_predict2.tsv')
        other_algo_res_df = pd.read_csv(other_algo_res_file_nm,  delimiter='\t', header=None, names=['other_algo_pred_prob', 'label'])
        # retrieve the correct target (class) labels
        other_algo_y_target_arr = (other_algo_res_df['label'].to_numpy()).astype(int)
        # check the equality with y_target_arr from mat_pip
        same = np.array_equal(y_target_arr, other_algo_y_target_arr)
        if(not same):
            print('######### ERROR!! ERROR!! y_target_arr and other_algo_y_target_arr not same for R20_'+str(i)+'_predict2.tsv')
            sys.exit(0)
        # retrieve the class labels predicted by other_algo (model-2)
        y_other_algo_pred_prob_arr = other_algo_res_df['other_algo_pred_prob'].to_numpy()
        y_other_algo_arr = (np.round(y_other_algo_pred_prob_arr, decimals=0)).astype(int)
        y_other_algo_lst += y_other_algo_arr.tolist()
        # ############# for other  algo -end #############
    # end of for loop: for i in range(0,5):
    print('inside retrieve_03_pos_full_res() method - end')
    return (np.array(y_target_lst), np.array(y_mat_pip_lst), np.array(y_other_algo_lst))


def retrieve_50_pos_held_res(oot_path='./', other_algo_result_path='./'):
    print('inside retrieve_50_pos_held_res() method - start')
    mat_pip_res_path = os.path.join(root_path, 'dataset/proc_data/mat_res/mat_res_origMan_auxTlOtherMan')
    y_target_lst, y_mat_pip_lst, y_other_algo_lst = [], [], []

    for i in range(0,6):
        for j in range(i,6):
            # ############# for mat_pip algo -start #############
            mat_pip_res_file_nm = os.path.join(mat_pip_res_path, 'H50_'+str(i)+'_'+str(j)+'_predict.tsv')
            mat_res_df = pd.read_csv(mat_pip_res_file_nm,  delimiter='\t', header=None, names=['mat_pred_prob', 'label'])
            # retrieve the correct target (class) labels
            y_target_arr = (mat_res_df['label'].to_numpy()).astype(int)
            y_target_lst += y_target_arr.tolist()
            # retrieve the class labels predicted by mat_pip (model-1)
            y_mat_pred_prob_arr = mat_res_df['mat_pred_prob'].to_numpy()
            y_mat_pip_arr = (np.round(y_mat_pred_prob_arr, decimals=0)).astype(int)
            y_mat_pip_lst += y_mat_pip_arr.tolist()
            # ############# for mat_pip algo -end #############

            # ############# for other  algo -start #############
            other_algo_res_file_nm = os.path.join(other_algo_result_path, 'H50_'+str(i)+'_'+str(j)+'_predict.tsv')
            other_algo_res_df = pd.read_csv(other_algo_res_file_nm,  delimiter='\t', header=None, names=['other_algo_pred_prob', 'label'])
            # retrieve the correct target (class) labels
            other_algo_y_target_arr = (other_algo_res_df['label'].to_numpy()).astype(int)
            # check the equality with y_target_arr from mat_pip
            same = np.array_equal(y_target_arr, other_algo_y_target_arr)
            if(not same):
                print('######### ERROR!! ERROR!! y_target_arr and other_algo_y_target_arr not same for H50_'+str(i)+'_'+str(j)+'_predict.tsv')
                sys.exit(0)
            # retrieve the class labels predicted by other_algo (model-2)
            y_other_algo_pred_prob_arr = other_algo_res_df['other_algo_pred_prob'].to_numpy()
            y_other_algo_arr = (np.round(y_other_algo_pred_prob_arr, decimals=0)).astype(int)
            y_other_algo_lst += y_other_algo_arr.tolist()
            # ############# for other  algo -end #############
        # end of for loop: for j in range(i,6):
    # end of for loop: for i in range(0,6):
    print('inside retrieve_50_pos_held_res() method - end')
    return (np.array(y_target_lst), np.array(y_mat_pip_lst), np.array(y_other_algo_lst))


def retrieve_10_pos_held_res(oot_path='./', other_algo_result_path='./'):
    print('inside retrieve_10_pos_held_res() method - start')
    mat_pip_res_path = os.path.join(root_path, 'dataset/proc_data/mat_res/mat_res_origMan_auxTlOtherMan')
    y_target_lst, y_mat_pip_lst, y_other_algo_lst = [], [], []

    for i in range(0,6):
        for j in range(i,6):
            # ############# for mat_pip algo -start #############
            mat_pip_res_file_nm = os.path.join(mat_pip_res_path, 'H20_'+str(i)+'_'+str(j)+'_predict1.tsv')
            mat_res_df = pd.read_csv(mat_pip_res_file_nm,  delimiter='\t', header=None, names=['mat_pred_prob', 'label'])
            # retrieve the correct target (class) labels
            y_target_arr = (mat_res_df['label'].to_numpy()).astype(int)
            y_target_lst += y_target_arr.tolist()
            # retrieve the class labels predicted by mat_pip (model-1)
            y_mat_pred_prob_arr = mat_res_df['mat_pred_prob'].to_numpy()
            y_mat_pip_arr = (np.round(y_mat_pred_prob_arr, decimals=0)).astype(int)
            y_mat_pip_lst += y_mat_pip_arr.tolist()
            # ############# for mat_pip algo -end #############

            # ############# for other  algo -start #############
            other_algo_res_file_nm = os.path.join(other_algo_result_path, 'H20_'+str(i)+'_'+str(j)+'_predict1.tsv')
            other_algo_res_df = pd.read_csv(other_algo_res_file_nm,  delimiter='\t', header=None, names=['other_algo_pred_prob', 'label'])
            # retrieve the correct target (class) labels
            other_algo_y_target_arr = (other_algo_res_df['label'].to_numpy()).astype(int)
            # check the equality with y_target_arr from mat_pip
            same = np.array_equal(y_target_arr, other_algo_y_target_arr)
            if(not same):
                print('######### ERROR!! ERROR!! y_target_arr and other_algo_y_target_arr not same for H20_'+str(i)+'_'+str(j)+'_predict1.tsv')
                sys.exit(0)
            # retrieve the class labels predicted by other_algo (model-2)
            y_other_algo_pred_prob_arr = other_algo_res_df['other_algo_pred_prob'].to_numpy()
            y_other_algo_arr = (np.round(y_other_algo_pred_prob_arr, decimals=0)).astype(int)
            y_other_algo_lst += y_other_algo_arr.tolist()
            # ############# for other  algo -end #############
        # end of for loop: for j in range(i,6):
    # end of for loop: for i in range(0,6):
    print('inside retrieve_10_pos_held_res() method - end')
    return (np.array(y_target_lst), np.array(y_mat_pip_lst), np.array(y_other_algo_lst))


def retrieve_03_pos_held_res(oot_path='./', other_algo_result_path='./'):
    print('inside retrieve_03_pos_held_res() method - start')
    mat_pip_res_path = os.path.join(root_path, 'dataset/proc_data/mat_res/mat_res_origMan_auxTlOtherMan')
    y_target_lst, y_mat_pip_lst, y_other_algo_lst = [], [], []

    for i in range(0,6):
        for j in range(i,6):
            # ############# for mat_pip algo -start #############
            mat_pip_res_file_nm = os.path.join(mat_pip_res_path, 'H20_'+str(i)+'_'+str(j)+'_predict2.tsv')
            mat_res_df = pd.read_csv(mat_pip_res_file_nm,  delimiter='\t', header=None, names=['mat_pred_prob', 'label'])
            # retrieve the correct target (class) labels
            y_target_arr = (mat_res_df['label'].to_numpy()).astype(int)
            y_target_lst += y_target_arr.tolist()
            # retrieve the class labels predicted by mat_pip (model-1)
            y_mat_pred_prob_arr = mat_res_df['mat_pred_prob'].to_numpy()
            y_mat_pip_arr = (np.round(y_mat_pred_prob_arr, decimals=0)).astype(int)
            y_mat_pip_lst += y_mat_pip_arr.tolist()
            # ############# for mat_pip algo -end #############

            # ############# for other  algo -start #############
            other_algo_res_file_nm = os.path.join(other_algo_result_path, 'H20_'+str(i)+'_'+str(j)+'_predict2.tsv')
            other_algo_res_df = pd.read_csv(other_algo_res_file_nm,  delimiter='\t', header=None, names=['other_algo_pred_prob', 'label'])
            # retrieve the correct target (class) labels
            other_algo_y_target_arr = (other_algo_res_df['label'].to_numpy()).astype(int)
            # check the equality with y_target_arr from mat_pip
            same = np.array_equal(y_target_arr, other_algo_y_target_arr)
            if(not same):
                print('######### ERROR!! ERROR!! y_target_arr and other_algo_y_target_arr not same for H20_'+str(i)+'_'+str(j)+'_predict2.tsv')
                sys.exit(0)
            # retrieve the class labels predicted by other_algo (model-2)
            y_other_algo_pred_prob_arr = other_algo_res_df['other_algo_pred_prob'].to_numpy()
            y_other_algo_arr = (np.round(y_other_algo_pred_prob_arr, decimals=0)).astype(int)
            y_other_algo_lst += y_other_algo_arr.tolist()
            # ############# for other  algo -end #############
        # end of for loop: for j in range(i,6):
    # end of for loop: for i in range(0,6):
    print('inside retrieve_03_pos_held_res() method - end')
    return (np.array(y_target_lst), np.array(y_mat_pip_lst), np.array(y_other_algo_lst))



if __name__ == '__main__':
    root_path = os.path.join('/project/root/directory/path/here')
    

    # ############ for CROSS-SPECIES #############
    exec_statSigTests_DS(root_path=root_path)

    # ############ for human-benchmark ############
    # pipr result path
    other_algo_result_path = '/scratch/pralaycs/Shubh_Working_Remote/PPI_Wkspc/PPI_Code/only_seq_prj_v1/dataset/proc_data/benchmark_res/pipr_res/pipr_res_man_orig'
    exec_statSigTests_humanBenchmark(root_path=root_path, other_algo_result_path=other_algo_result_path)


