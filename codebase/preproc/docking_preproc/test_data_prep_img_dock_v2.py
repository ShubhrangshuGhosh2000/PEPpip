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


def make_test_pp_pair_list(root_path='./', docking_version = '4_0', restricted_len = 400):
    print('In make_test_pp_pair_list() method - Start')
    test_pair_list_path = os.path.join(root_path, 'dataset/preproc_data_docking_BM_' + str(docking_version), 'derived_feat/dock', 'dock_test_lenLimit_' + str(restricted_len) + '.tsv')
    spec_test_pairs_df = pd.read_csv(test_pair_list_path, delimiter='\t', header=None, names=['prot1_id', 'prot2_id', 'label'])
    # declare a list of lists where each inner list contains prot1_id and prot2_id
    pp_test_pair_lst_lsts = []
    # decalare a list of class labels for each pair
    test_class_label_lst = []
    # iterate over spec_test_pairs_df and populate pp_test_pair_lst_lsts and test_class_label_lst
    for index, row in spec_test_pairs_df.iterrows():
        if(index % 20 == 0): print('starting ' + str(index+1) + '-th protein out of ' + str(spec_test_pairs_df.shape[0]))
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
    

    restricted_len = 400
    docking_version_lst = ['4_0', '5_5']  # '4_0', '5_5'

    for docking_version in docking_version_lst:
        make_test_pp_pair_list(root_path=root_path, docking_version = docking_version, restricted_len=restricted_len)
