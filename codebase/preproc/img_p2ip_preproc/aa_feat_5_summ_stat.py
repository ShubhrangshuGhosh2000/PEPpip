import os
import sys
from pathlib import Path

import joblib
import pandas as pd
import numpy as np

path_root = Path(__file__).parents[2]  
sys.path.insert(0, str(path_root))


def derive_5_summary_stats_for_aa_feat(root_path='./', spec_type = 'human'):
    tl_feat_dict_fl_nm = os.path.join(root_path, 'dataset/preproc_data_DS', 'DS_seq_feat_dict_prot_t5_xl_uniref50' +  '_' + spec_type + '_img.pkl')
    tl_feat_dict = joblib.load(tl_feat_dict_fl_nm)
    aa_set = set()
    for prot_id in list(tl_feat_dict.keys()):
        prot_seq_asStr = tl_feat_dict[prot_id]['seq']
        prot_seq_asSet= {aa for aa in prot_seq_asStr} 
        aa_set = aa_set | prot_seq_asSet
    aa_feat_dict = {}
    for aa in aa_set:
        aa_feat_dict[aa] = []
    tl_feat_dict_key_lst = list(tl_feat_dict.keys())
    for itr, prot_id in enumerate(tl_feat_dict_key_lst):
        prot_seq_asStr = tl_feat_dict[prot_id]['seq']
        seq_2d_feat = tl_feat_dict[prot_id]['seq_2d_feat']
        for aa_idx in range(len(prot_seq_asStr)):
            aa = prot_seq_asStr[aa_idx]
            aa_feat_dict[aa].append(seq_2d_feat[aa_idx,:])
    aa_feat_summ_stat_dict = {}
    for aa_itr, aa in enumerate(aa_set):
        aa_feat_summ_stat_dict[aa] = {}
        if(len(aa_feat_dict[aa]) > 0 ):
            aa_feat_summ_stat_dict[aa]['overall_min'] = np.amin(aa_feat_dict[aa], axis=0)
            aa_feat_summ_stat_dict[aa]['overall_max'] = np.amax(aa_feat_dict[aa], axis=0)
            aa_feat_summ_stat_dict[aa]['std'] = np.std(aa_feat_dict[aa], axis=0)
            aa_feat_summ_stat_dict[aa]['mean'] = np.mean(aa_feat_dict[aa], axis=0)
            aa_feat_summ_stat_dict[aa]['median'] = np.median(aa_feat_dict[aa], axis=0)
    dict_file_nm_path = os.path.join(root_path, 'dataset/preproc_data_tl_feat_to_img/aa_feat_summ_stat', 'aa_feat_summ_stat_' + spec_type + '.pkl')
    joblib.dump(value=aa_feat_summ_stat_dict, filename=dict_file_nm_path, compress=3)


if __name__ == '__main__':
    root_path = os.path.join('/project/root/directory/path/here')
    spec_type = 'human'
    derive_5_summary_stats_for_aa_feat(root_path, spec_type = spec_type)
