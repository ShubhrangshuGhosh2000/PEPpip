import os
import sys
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
path_root = Path(__file__).parents[2]  
sys.path.insert(0, str(path_root))


def make_train_test_pp_pair_list(root_path='./', spec_type = 'human', restricted_len = 400):
    test_pair_list_path = os.path.join(root_path, 'dataset/orig_data_DS/pairs', spec_type + '_test.tsv' )
    spec_test_pairs_df = pd.read_csv(test_pair_list_path, delimiter='\t', header=None, names=['prot1_id', 'prot2_id', 'label'])
    len_restricted_DS_seq_df = pd.read_csv(os.path.join(root_path, 'dataset/preproc_data_DS/seqs', 'DS_' + spec_type + '_seq_len' + str(restricted_len) + '.csv'))
    valid_prot_ids = set(len_restricted_DS_seq_df['prot_id'])
    len_restrct_spec_test_pairs_df = spec_test_pairs_df[(spec_test_pairs_df['prot1_id'].isin(valid_prot_ids)) & (spec_test_pairs_df['prot2_id'].isin(valid_prot_ids))]
    len_restrct_spec_test_pairs_df.to_csv(os.path.join(root_path, 'dataset/preproc_data_DS/pairs', 'DS_test_pairs_' + spec_type + '_seq_len' + str(restricted_len) + '.csv'), index=False)
    pp_test_pair_lst_lsts = []
    test_class_label_lst = []
    for index, row in len_restrct_spec_test_pairs_df.iterrows():
        pp_test_pair_lst_lsts.append([row['prot1_id'], row['prot2_id']])
        test_class_label_lst.append(float(row['label']))
    pp_pair_fl_nm = os.path.join(root_path, 'dataset/preproc_data_DS/train_test_list_dump', 'pp_pair_test_' + spec_type + '.pkl')
    joblib.dump(value=pp_test_pair_lst_lsts, filename=pp_pair_fl_nm, compress=3)
    class_label_fl_nm = os.path.join(root_path, 'dataset/preproc_data_DS/train_test_list_dump', 'class_label_test_' + spec_type + '.pkl')
    joblib.dump(value=test_class_label_lst, filename=class_label_fl_nm, compress=3)
    if(spec_type == 'human'):
        train_pair_list_path = os.path.join(root_path, 'dataset/orig_data_DS/pairs', spec_type + '_train.tsv' )
        human_train_pairs_df = pd.read_csv(train_pair_list_path, delimiter='\t', header=None, names=['prot1_id', 'prot2_id', 'label'])
        len_restricted_DS_seq_df = pd.read_csv(os.path.join(root_path, 'dataset/preproc_data_DS/seqs', 'DS_' + spec_type + '_seq_len' + str(restricted_len) + '.csv'))
        valid_prot_ids = set(len_restricted_DS_seq_df['prot_id'])
        len_restrct_human_train_pairs_df = human_train_pairs_df[(human_train_pairs_df['prot1_id'].isin(valid_prot_ids)) & (human_train_pairs_df['prot2_id'].isin(valid_prot_ids))]
        len_restrct_human_train_pairs_df.to_csv(os.path.join(root_path, 'dataset/preproc_data_DS/pairs', 'DS_train_pairs_' + spec_type + '_seq_len' + str(restricted_len) + '.csv'), index=False)
        pp_train_pair_lst_lsts = []
        pp_train_pair_dbl_lst_lsts = []
        train_class_label_lst = []
        train_class_label_dbl_lst = []
        for index, row in len_restrct_human_train_pairs_df.iterrows():
            pp_train_pair_lst_lsts.append([row['prot1_id'], row['prot2_id']])
            train_class_label_lst.append(float(row['label']))
            pp_train_pair_dbl_lst_lsts.append([row['prot1_id'], row['prot2_id']])
            train_class_label_dbl_lst.append(float(row['label']))
            pp_train_pair_dbl_lst_lsts.append([row['prot2_id'], row['prot1_id']])
            train_class_label_dbl_lst.append(float(row['label']))
        pp_pair_fl_nm = os.path.join(root_path, 'dataset/preproc_data_DS/train_test_list_dump', 'pp_pair_train_' + spec_type + '.pkl')
        joblib.dump(value=pp_train_pair_lst_lsts, filename=pp_pair_fl_nm, compress=3)
        class_label_fl_nm = os.path.join(root_path, 'dataset/preproc_data_DS/train_test_list_dump', 'class_label_train_' + spec_type + '.pkl')
        joblib.dump(value=train_class_label_lst, filename=class_label_fl_nm, compress=3)
        pp_pair_dbl_fl_nm = os.path.join(root_path, 'dataset/preproc_data_DS/train_test_list_dump', 'pp_pair_dbl_train_' + spec_type + '.pkl')
        joblib.dump(value=pp_train_pair_dbl_lst_lsts, filename=pp_pair_dbl_fl_nm, compress=3)
        class_label_dbl_fl_nm = os.path.join(root_path, 'dataset/preproc_data_DS/train_test_list_dump', 'class_label_dbl_train_' + spec_type + '.pkl')
        joblib.dump(value=train_class_label_dbl_lst, filename=class_label_dbl_fl_nm, compress=3)


if __name__ == '__main__':
    root_path = os.path.join('/project/root/directory/path/here')
    restricted_len = 400
    spec_type_lst = ['ecoli', 'fly', 'mouse', 'worm', 'yeast', 'human']
    for spec_type in spec_type_lst:
        make_train_test_pp_pair_list(root_path, spec_type = spec_type, restricted_len=restricted_len)
