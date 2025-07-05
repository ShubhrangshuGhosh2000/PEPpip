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


def make_train_test_pp_pair_list(root_path='./', spec_type = 'human', restricted_len = 400):
    print('In make_train_test_pp_pair_list() method - Start')
    print('\n########## spec_type: ' + str(spec_type))
    test_pair_list_path = os.path.join(root_path, 'dataset/orig_data_DS/pairs', spec_type + '_test.tsv' )
    spec_test_pairs_df = pd.read_csv(test_pair_list_path, delimiter='\t', header=None, names=['prot1_id', 'prot2_id', 'label'])
    
    # Filter out spec_test_pairs_df as per restricted_len
    len_restricted_DS_seq_df = pd.read_csv(os.path.join(root_path, 'dataset/preproc_data_DS/seqs', 'DS_' + spec_type + '_seq_len' + str(restricted_len) + '.csv'))
    valid_prot_ids = set(len_restricted_DS_seq_df['prot_id'])
    len_restrct_spec_test_pairs_df = spec_test_pairs_df[(spec_test_pairs_df['prot1_id'].isin(valid_prot_ids)) & (spec_test_pairs_df['prot2_id'].isin(valid_prot_ids))]
    # save len_restrct_spec_test_pairs_df
    len_restrct_spec_test_pairs_df.to_csv(os.path.join(root_path, 'dataset/preproc_data_DS/pairs', 'DS_test_pairs_' + spec_type + '_seq_len' + str(restricted_len) + '.csv'), index=False)

    # declare a list of lists where each inner list contains prot1_id and prot2_id
    pp_test_pair_lst_lsts = []
    # decalare a list of class labels for each pair
    test_class_label_lst = []
    # iterate over len_restrct_spec_test_pairs_df and populate pp_test_pair_lst_lsts and test_class_label_lst
    for index, row in len_restrct_spec_test_pairs_df.iterrows():
        if(index % 200 == 0): print(spec_type + ' :: starting ' + str(index+1) + '-th protein out of ' + str(spec_test_pairs_df.shape[0]))
        pp_test_pair_lst_lsts.append([row['prot1_id'], row['prot2_id']])
        test_class_label_lst.append(float(row['label']))
    # save pp_test_pair_lst_lsts and test_class_label_lst
    print('saving pp_test_pair_lst_lsts and test_class_label_lst')
    pp_pair_fl_nm = os.path.join(root_path, 'dataset/preproc_data_DS/train_test_list_dump', 'pp_pair_test_' + spec_type + '.pkl')
    joblib.dump(value=pp_test_pair_lst_lsts, filename=pp_pair_fl_nm, compress=3)
    class_label_fl_nm = os.path.join(root_path, 'dataset/preproc_data_DS/train_test_list_dump', 'class_label_test_' + spec_type + '.pkl')
    joblib.dump(value=test_class_label_lst, filename=class_label_fl_nm, compress=3)

    if(spec_type == 'human'):
        # perform special processing for human train pairs
        print('performing special processing for human train pairs')
        train_pair_list_path = os.path.join(root_path, 'dataset/orig_data_DS/pairs', spec_type + '_train.tsv' )
        human_train_pairs_df = pd.read_csv(train_pair_list_path, delimiter='\t', header=None, names=['prot1_id', 'prot2_id', 'label'])
        
        # Filter out human_train_pairs_df as per restricted_len
        len_restricted_DS_seq_df = pd.read_csv(os.path.join(root_path, 'dataset/preproc_data_DS/seqs', 'DS_' + spec_type + '_seq_len' + str(restricted_len) + '.csv'))
        valid_prot_ids = set(len_restricted_DS_seq_df['prot_id'])
        len_restrct_human_train_pairs_df = human_train_pairs_df[(human_train_pairs_df['prot1_id'].isin(valid_prot_ids)) & (human_train_pairs_df['prot2_id'].isin(valid_prot_ids))]
        # save len_restrct_human_train_pairs_df
        len_restrct_human_train_pairs_df.to_csv(os.path.join(root_path, 'dataset/preproc_data_DS/pairs', 'DS_train_pairs_' + spec_type + '_seq_len' + str(restricted_len) + '.csv'), index=False)
        
        # declare a list of lists where each inner list contains prot1_id and prot2_id
        pp_train_pair_lst_lsts = []
        # declare a list of lists where each inner list contains prot1_id, prot2_id and prot2_id, prot1_id
        pp_train_pair_dbl_lst_lsts = []
        # decalare a list of class labels for each pair
        train_class_label_lst = []
        train_class_label_dbl_lst = []
        # iterate over len_restrct_human_train_pairs_df and populate pp_train_pair_lst_lsts and train_class_label_lst
        for index, row in len_restrct_human_train_pairs_df.iterrows():
            if(index % 200 == 0): print('human train :: starting ' + str(index) + '-th protein out of ' + str(human_train_pairs_df.shape[0]))
            pp_train_pair_lst_lsts.append([row['prot1_id'], row['prot2_id']])
            train_class_label_lst.append(float(row['label']))

            pp_train_pair_dbl_lst_lsts.append([row['prot1_id'], row['prot2_id']])
            train_class_label_dbl_lst.append(float(row['label']))
            pp_train_pair_dbl_lst_lsts.append([row['prot2_id'], row['prot1_id']])
            train_class_label_dbl_lst.append(float(row['label']))
        # End of for loop: for index, row in len_restrct_human_train_pairs_df.iterrows():

        # save pp_train_pair_lst_lsts and train_class_label_lst
        print('saving pp_train_pair_lst_lsts and train_class_label_lst')
        pp_pair_fl_nm = os.path.join(root_path, 'dataset/preproc_data_DS/train_test_list_dump', 'pp_pair_train_' + spec_type + '.pkl')
        joblib.dump(value=pp_train_pair_lst_lsts, filename=pp_pair_fl_nm, compress=3)
        class_label_fl_nm = os.path.join(root_path, 'dataset/preproc_data_DS/train_test_list_dump', 'class_label_train_' + spec_type + '.pkl')
        joblib.dump(value=train_class_label_lst, filename=class_label_fl_nm, compress=3)
        
        # print('saving pp_train_pair_dbl_lst_lsts and train_class_label_dbl_lst')
        pp_pair_dbl_fl_nm = os.path.join(root_path, 'dataset/preproc_data_DS/train_test_list_dump', 'pp_pair_dbl_train_' + spec_type + '.pkl')
        joblib.dump(value=pp_train_pair_dbl_lst_lsts, filename=pp_pair_dbl_fl_nm, compress=3)
        class_label_dbl_fl_nm = os.path.join(root_path, 'dataset/preproc_data_DS/train_test_list_dump', 'class_label_dbl_train_' + spec_type + '.pkl')
        joblib.dump(value=train_class_label_dbl_lst, filename=class_label_dbl_fl_nm, compress=3)
    # End of if block: if(spec_type == 'human'):
    print('In make_train_test_pp_pair_list() method - End')


if __name__ == '__main__':
    root_path = os.path.join('/project/root/directory/path/here')
    root_path = os.path.join('/scratch/pralaycs/Shubh_Working_Remote/PPI_Wkspc/PPI_Code/mat_p2ip_prj_working')

    restricted_len = 400
    # spec_type_lst = ['ecoli', 'fly', 'mouse', 'worm', 'yeast', 'human']
    spec_type_lst = ['ecoli', 'fly', 'mouse', 'worm', 'yeast', 'human']
    for spec_type in spec_type_lst:
        make_train_test_pp_pair_list(root_path, spec_type = spec_type, restricted_len=restricted_len)
        
