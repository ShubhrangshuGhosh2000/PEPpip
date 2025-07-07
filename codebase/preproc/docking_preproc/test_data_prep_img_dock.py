import os
import sys
from pathlib import Path
import joblib
import pandas as pd
path_root = Path(__file__).parents[2]  
sys.path.insert(0, str(path_root))


def make_test_pp_pair_list(root_path='./', docking_version = '5_5', restricted_len = 400):
    test_pair_list_path = os.path.join(root_path, 'dataset/preproc_data_docking_BM_' + str(docking_version), 'derived_feat/dock', 'dock_test_lenLimit_' + str(restricted_len) + '.tsv')
    spec_test_pairs_df = pd.read_csv(test_pair_list_path, delimiter='\t', header=None, names=['prot1_id', 'prot2_id', 'label'])
    pp_test_pair_lst_lsts = []
    test_class_label_lst = []
    for index, row in spec_test_pairs_df.iterrows():
        pp_test_pair_lst_lsts.append([row['prot1_id'], row['prot2_id']])
        test_class_label_lst.append(float(row['label']))
    pp_pair_fl_nm = os.path.join(root_path, 'dataset/preproc_data_docking_BM_' + str(docking_version), 'dock_test_list_dump', 'dock_pp_pair_test.pkl')
    joblib.dump(value=pp_test_pair_lst_lsts, filename=pp_pair_fl_nm, compress=3)
    class_label_fl_nm = os.path.join(root_path, 'dataset/preproc_data_docking_BM_' + str(docking_version), 'dock_test_list_dump', 'dock_class_label_test.pkl')
    joblib.dump(value=test_class_label_lst, filename=class_label_fl_nm, compress=3)


if __name__ == '__main__':
    root_path = os.path.join('/project/root/directory/path/here')
    restricted_len = 400
    docking_version_lst = ['5_5']
    for docking_version in docking_version_lst:
        make_test_pp_pair_list(root_path=root_path, docking_version = docking_version, restricted_len=restricted_len)
