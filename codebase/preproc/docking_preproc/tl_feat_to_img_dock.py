import os
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

path_root = Path(__file__).parents[2]  # upto 'codebase' folder
sys.path.insert(0, str(path_root))
# print(sys.path)


# Use protTrans TL features (2d) on Dscript datasets (max prot length = 800,  per protein feature matrix 800 x 1024).
# Then median pool amino acid wise to get 800 x 1 feature vector. 
def prep_2dTlFeat_to_1dAaFeat(root_path='./', max_prot_len = 800, docking_version = '4_0'):
    print('In prep_2dTlFeat_to_1dAaFeat() method - Start')
    print('\n########## docking_version: ' + str(docking_version))
    # load already saved protTrans TL feature dictionary
    print('loading already saved protTrans TL feature dictionary ...')
    tl_feat_dict_fl_nm = os.path.join(root_path, 'dataset/preproc_data_docking_BM_' + str(docking_version), 'dock_seq_feat_dict_prot_t5_xl_uniref50_img.pkl')
    tl_feat_dict = joblib.load(tl_feat_dict_fl_nm)
    print('loaded already saved protTrans TL feature dictionary ...')

    # load already saved overall_aa_feat_summ_stat_dict dictionary
    print('loading already saved overall_aa_feat_summ_stat_dict dictionary ...')
    summ_stat_dict_file_nm_path = os.path.join(root_path, 'dataset/preproc_data_tl_feat_to_img/aa_feat_summ_stat', 'overall_aa_feat_summ_stat.pkl')
    overall_aa_feat_summ_stat_dict = joblib.load(summ_stat_dict_file_nm_path)
    print('loaded already saved overall_aa_feat_summ_stat_dict dictionary ...')

    # iterate through tl_feat_dict and populate oneD_aa_feat_dict
    print('iterating through tl_feat_dict and populating oneD_aa_feat_dict')
    oneD_aa_feat_dict = {}
    tl_feat_dict_key_lst = list(tl_feat_dict.keys())
    for itr, prot_id in enumerate(tl_feat_dict_key_lst):
        if(itr % 10 == 0) : print('performing pooling aa-wise :: itr = ' + str(itr) + ' out of ' + str(len(tl_feat_dict_key_lst)))
        # retrieve the sequence of the respective protein
        seq = tl_feat_dict[prot_id]['seq']
        # retrieve the sequence length of the respective protein
        seq_len = tl_feat_dict[prot_id]['seq_len']
        # retrieve the 2D-tl feature matrix
        seq_2d_feat_arr = tl_feat_dict[prot_id]['seq_2d_feat']
        # perform the min-max normalization per amino acid for the individual protein sequence using overall_aa_feat_summ_stat_dict and 
        # setting maxRange=1 and minRange=0.1 in the formula:
        # ((value - minVal) / (maxVal - minVal)) * (maxRange - minRange) + minRange
        for aa_idx in range(len(seq)):
            aa = seq[aa_idx]
            value = seq_2d_feat_arr[aa_idx,:]
            minVal = overall_aa_feat_summ_stat_dict[aa]['overall_min']
            maxVal = overall_aa_feat_summ_stat_dict[aa]['overall_max']
            maxRange = 1
            minRange = 0.1
            min_max_normalized_val = ((value - minVal) / (maxVal - minVal)) * (maxRange - minRange) + minRange
            seq_2d_feat_arr[aa_idx,:] = min_max_normalized_val
        # end of for loop: for aa_idx in range(len(seq)):
        # apply pooling amino-acid wise
        aa_wise_1d_feat_arr = np.apply_along_axis(np.median, axis=1, arr=seq_2d_feat_arr)  # can apply np.mean/max/min etc. in place np.median 
        # adjust the length of aa_wise_1d_feat_arr to max_prot_len
        aa_wise_1d_feat_arr_max_len = np.zeros(max_prot_len)
        if(len(aa_wise_1d_feat_arr) <= max_prot_len):
            # if the current protein sequence length <= max_prot_len, then apply zero-padding
            aa_wise_1d_feat_arr_max_len[:len(aa_wise_1d_feat_arr)] = aa_wise_1d_feat_arr
        else:
            # if the current protein sequence length > max_prot_len, then subset from the beginning
            aa_wise_1d_feat_arr_max_len = aa_wise_1d_feat_arr[:max_prot_len]
        # populate tl_seq_feat_for_img_dict
        oneD_aa_feat_dict[str(prot_id)] = {'seq': seq, 'seq_len': seq_len, 'aa_wise_1d_feat': aa_wise_1d_feat_arr_max_len}
    # end of for loop
    # save oneD_aa_feat_dict
    print("\n Saving oneD_aa_feat_dict to a .pkl file...")
    oneD_aa_feat_dict_fl_nm = os.path.join(root_path, 'dataset/preproc_data_docking_BM_' + str(docking_version), 'dock_oneD_aa_feat', 'oneD_aa_feat_dict_len_' + str(max_prot_len) + '.pkl')
    joblib.dump(value=oneD_aa_feat_dict, filename=oneD_aa_feat_dict_fl_nm, compress=3)
    print("\n The DS_seq_feat_dict is saved as: " + oneD_aa_feat_dict_fl_nm)
    print('In prep_2dTlFeat_to_1dAaFeat() method - End')


def make_test_pp_pair_list(root_path='./', docking_version = '4_0'):
    print('In make_test_pp_pair_list() method - Start')
    test_pair_list_path = os.path.join(root_path, 'dataset/preproc_data_docking_BM_' + str(docking_version), 'derived_feat/dock', 'dock_test.tsv')
    spec_test_pairs_df = pd.read_csv(test_pair_list_path, delimiter='\t', header=None, names=['prot1_id', 'prot2_id', 'label'])
    # declare a list of lists where each inner list contains prot1_id and prot2_id
    pp_test_pair_lst_lsts = []
    # decalare a list of class labels for each pair
    test_class_label_lst = []
    # iterate over spec_test_pairs_df and populate pp_test_pair_lst_lsts and test_class_label_lst
    for index, row in spec_test_pairs_df.iterrows():
        if(index % 20 == 0): print('starting ' + str(index) + '-th protein out of ' + str(spec_test_pairs_df.shape[0]))
        pp_test_pair_lst_lsts.append([row['prot1_id'], row['prot2_id']])
        test_class_label_lst.append(float(row['label']))
    # save pp_test_pair_lst_lsts and test_class_label_lst
    print('saving pp_test_pair_lst_lsts and test_class_label_lst')
    pp_pair_fl_nm = os.path.join(root_path, 'dataset/preproc_data_docking_BM_' + str(docking_version), 'dock_test_list_dump', 'dock_pp_pair_test.pkl')
    joblib.dump(value=pp_test_pair_lst_lsts, filename=pp_pair_fl_nm, compress=3)
    class_label_fl_nm = os.path.join(root_path, 'dataset/preproc_data_docking_BM_' + str(docking_version), 'dock_test_list_dump', 'dock_class_label_test.pkl')
    joblib.dump(value=test_class_label_lst, filename=class_label_fl_nm, compress=3)
    print('In make_test_pp_pair_list() method - End')


if __name__ == '__main__':
    root_path = os.path.join('/project/root/directory/path/here')
    root_path = os.path.join('/scratch/pralaycs/Shubh_Working_Remote/PPI_Wkspc/PPI_Code/mat_p2ip_prj_working')

    max_prot_len_lst = [400]  # 256, 400, 800
    docking_version_lst = ['4_0', '5_5']  # '4_0', '5_5'

    for docking_version in docking_version_lst:
        for max_prot_len in max_prot_len_lst:
            prep_2dTlFeat_to_1dAaFeat(root_path=root_path, max_prot_len = max_prot_len, docking_version = docking_version)
        make_test_pp_pair_list(root_path=root_path, docking_version = docking_version)
