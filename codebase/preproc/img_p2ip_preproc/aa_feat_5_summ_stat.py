import os
import sys
from pathlib import Path

import joblib
import pandas as pd
import numpy as np

path_root = Path(__file__).parents[2]  # upto 'codebase' folder
sys.path.insert(0, str(path_root))
# print(sys.path)


def derive_5_summary_stats_for_aa_feat(root_path='./', spec_type = 'human'):
    print('In derive_5_summary_stats_for_aa_feat() method - Start')
    print('\n########## spec_type: ' + str(spec_type))
    # load already saved protTrans TL feature dictionary
    print('loading already saved protTrans TL feature dictionary ...')
    tl_feat_dict_fl_nm = os.path.join(root_path, 'dataset/preproc_data_DS', 'DS_seq_feat_dict_prot_t5_xl_uniref50' +  '_' + spec_type + '_img.pkl')
    tl_feat_dict = joblib.load(tl_feat_dict_fl_nm)
    print('loaded already saved protTrans TL feature dictionary ...')
    # iterate through protein sequences to create a set of distinct amino-acids(aa)
    aa_set = set()
    for prot_id in list(tl_feat_dict.keys()):
        prot_seq_asStr = tl_feat_dict[prot_id]['seq']
        prot_seq_asSet= {aa for aa in prot_seq_asStr} 
        aa_set = aa_set | prot_seq_asSet
    print('distinct aa_set: ' + str(aa_set))
    # create a dictionary whose key is an amino acid (aa) and 
    # value is a list of 1d arrays each with length 1024 (representing one of 1024 aa features)
    aa_feat_dict = {}
    for aa in aa_set:
        aa_feat_dict[aa] = []
    print('populating aa_feat_dict...')
    tl_feat_dict_key_lst = list(tl_feat_dict.keys())
    for itr, prot_id in enumerate(tl_feat_dict_key_lst):
        print('itr: ' + str(itr) + ' out of: ' + str(len(tl_feat_dict_key_lst)-1))
        prot_seq_asStr = tl_feat_dict[prot_id]['seq']
        seq_2d_feat = tl_feat_dict[prot_id]['seq_2d_feat']
        # iterate through the protein sequence and update aa_feat_dict
        for aa_idx in range(len(prot_seq_asStr)):
            aa = prot_seq_asStr[aa_idx]
            aa_feat_dict[aa].append(seq_2d_feat[aa_idx,:])
        # end of for loop: for aa_idx in range(len(prot_seq_asStr)):
    # end of for loop: for itr, prot_id in enumerate(tl_feat_dict_key_lst):
    print('populated aa_feat_dict...')
    # calculate 5 summary stat for each of 1024 features of each amino acid
    print('\n calculating 5 summary stat for each of 1024 features of each amino acid')
    aa_feat_summ_stat_dict = {}
    print('aa_set: ' + str(sorted(aa_set)))
    for aa_itr, aa in enumerate(aa_set):
        aa_feat_summ_stat_dict[aa] = {}
        if(len(aa_feat_dict[aa]) > 0 ):
            aa_feat_summ_stat_dict[aa]['min'] = np.amin(aa_feat_dict[aa], axis=0)
            aa_feat_summ_stat_dict[aa]['max'] = np.amax(aa_feat_dict[aa], axis=0)
            aa_feat_summ_stat_dict[aa]['std'] = np.std(aa_feat_dict[aa], axis=0)
            aa_feat_summ_stat_dict[aa]['mean'] = np.mean(aa_feat_dict[aa], axis=0)
            aa_feat_summ_stat_dict[aa]['median'] = np.median(aa_feat_dict[aa], axis=0)
            print("### completed aa_itr: " + str(aa_itr) + " out of: " + str(len(aa_set)-1))
    # end of for loop: for aa in aa_set:
    # save aa_feat_summ_stat_dict
    print("\n saving aa_feat_summ_stat_dict")
    dict_file_nm_path = os.path.join(root_path, 'dataset/preproc_data_tl_feat_to_img/aa_feat_summ_stat', 'aa_feat_summ_stat_' + spec_type + '.pkl')
    joblib.dump(value=aa_feat_summ_stat_dict, filename=dict_file_nm_path, compress=3)
    print("\n The aa_feat_summ_stat_dict is saved as: " + dict_file_nm_path)
    print('In derive_5_summary_stats_for_aa_feat() method - End')


# ############ NOT REQUIRED TO BE SHARED 
# ############ as only human related statistics (i.e. min, max, etc.) per amino-acid should be used for the training but here, we are considering 
# ############ all the species for calculating the statistics (min, max, etc.).
# ############ We can tactfully say that after caculating the human specific statistics (min and max), we have kept some rooms for any other future species
# ############ and so, instead of using the exact min, max values obtained for the human specific amino-acids, we took more flexible range for the min and max 
# ############ values to be used for the min-max normalization purpose.
def derive_overall_5_summary_stats_for_aa_feat(root_path='./'):
    print('In derive_overall_5_summary_stats_for_aa_feat() method - Start')
    # iterate all the species to derive the overall 5 summary stat
    spec_type_lst = ['human', 'ecoli', 'fly', 'mouse', 'worm', 'yeast']
    
    # create a dictionary to store the overall 5 summary stat where the key is an amino acid (aa) and the value is 
    # another dictionary with keys are 'min', 'max', 'std', 'mean', 'median' and the value for each key in 
    # the inner dictionary is a list of 1d arrays of length 1024. This list can have maximum length of 6 if
    # the respective amino acid appears in all the 6 species. The other keys of this inner dictionary are 
    # 'overall_min', 'overall_max', 'overall_std', 'overall_mean', 'overall_median' and value for each of them 
    # is a 1d-array.
    overall_aa_feat_summ_stat_dict = {}
    for spec_type in spec_type_lst:
        print('\n########## spec_type: ' + str(spec_type))
        spec_stat_dict_fl_nm_path = os.path.join(root_path, 'dataset/preproc_data_tl_feat_to_img/aa_feat_summ_stat', 'aa_feat_summ_stat_' + spec_type + '.pkl')
        spec_stat_dict = joblib.load(spec_stat_dict_fl_nm_path)
    
        for aa in list(spec_stat_dict.keys()):
            if(aa not in overall_aa_feat_summ_stat_dict.keys()):
                print('aa: ' + str(aa) + ' is NOT already present in overall_aa_feat_summ_stat_dict')
                overall_aa_feat_summ_stat_dict[aa]= {'min': [], 'max': [], 'std': [], 'mean': [], 'median': []}
            # end of if block
            overall_aa_feat_summ_stat_dict[aa]['min'].append(spec_stat_dict[aa]['min'])
            overall_aa_feat_summ_stat_dict[aa]['max'].append(spec_stat_dict[aa]['max'])
            overall_aa_feat_summ_stat_dict[aa]['std'].append(spec_stat_dict[aa]['std'])
            overall_aa_feat_summ_stat_dict[aa]['mean'].append(spec_stat_dict[aa]['mean'])
            overall_aa_feat_summ_stat_dict[aa]['median'].append(spec_stat_dict[aa]['median'])
        # end of for loop: for aa in list(spec_stat_dict.keys()):
    # end of for loop: for spec in spec_type_lst:
    # calculate the overall stat
    print('calculating the overall stat')
    for aa_itr, aa in enumerate(list(overall_aa_feat_summ_stat_dict.keys())):
        print(' aa_itr: ' + str(aa_itr) + ' :: aa: ' + str(aa))
        overall_aa_feat_summ_stat_dict[aa]['overall_min'] = np.amin(overall_aa_feat_summ_stat_dict[aa]['min'], axis=0)
        overall_aa_feat_summ_stat_dict[aa]['overall_max'] = np.amax(overall_aa_feat_summ_stat_dict[aa]['max'], axis=0)
        overall_aa_feat_summ_stat_dict[aa]['overall_std'] = np.std(overall_aa_feat_summ_stat_dict[aa]['std'], axis=0)
        overall_aa_feat_summ_stat_dict[aa]['overall_mean'] = np.mean(overall_aa_feat_summ_stat_dict[aa]['mean'], axis=0)
        overall_aa_feat_summ_stat_dict[aa]['overall_median'] = np.median(overall_aa_feat_summ_stat_dict[aa]['median'], axis=0)
    # end for loop

    # # next, calculate the 'overall_single_min' for each amino-acid, which is the minimum of 1d-array (of length 1024) for that amino-acid stored in 
    # # overall_aa_feat_summ_stat_dict with key as 'overall_min'. Similarly, calculate 'overall_single_max'.
    # print('calculating overall_single_min and overall_single_max for each amino-acid')
    # for aa in list(overall_aa_feat_summ_stat_dict.keys()):
    #     overall_aa_feat_summ_stat_dict[aa]['overall_single_min'] = np.amin(overall_aa_feat_summ_stat_dict[aa]['overall_min'])
    #     overall_aa_feat_summ_stat_dict[aa]['overall_single_max'] = np.amax(overall_aa_feat_summ_stat_dict[aa]['overall_max'])
    #     print('aa: ' + str(aa) + ' :: overall_single_min: ' + str(overall_aa_feat_summ_stat_dict[aa]['overall_single_min']) + ' :: \
    #           overall_single_max: ' + str(overall_aa_feat_summ_stat_dict[aa]['overall_single_max']))

    # # next, calculate 'overall_single_min_human' and 'overall_single_max_human' for each amino-acid using human specific 
    # # spec_stat_dict with keys as 'min' and 'max' respectively.
    # print('calculating overall_single_min_human and overall_single_max_human for each amino-acid')
    # human_spec_stat_dict_fl_nm_path = os.path.join(root_path, 'dataset/preproc_data_tl_feat_to_img/aa_feat_summ_stat', 'aa_feat_summ_stat_' + 'human' + '.pkl')
    # human_spec_stat_dict = joblib.load(human_spec_stat_dict_fl_nm_path)
    # for aa in list(human_spec_stat_dict.keys()):
    #     overall_aa_feat_summ_stat_dict[aa]['overall_single_min_human'] = np.amin(human_spec_stat_dict[aa]['min'])
    #     overall_aa_feat_summ_stat_dict[aa]['overall_single_max_human'] = np.amax(human_spec_stat_dict[aa]['max'])
    #     print('aa: ' + str(aa) + ' :: overall_single_min_human: ' + str(overall_aa_feat_summ_stat_dict[aa]['overall_single_min_human']) + ' :: \
    #           overall_single_max_human: ' + str(overall_aa_feat_summ_stat_dict[aa]['overall_single_max_human']))

    # save overall_aa_feat_summ_stat_dict
    print("\n saving overall_aa_feat_summ_stat_dict")
    dict_file_nm_path = os.path.join(root_path, 'dataset/preproc_data_tl_feat_to_img/aa_feat_summ_stat', 'overall_aa_feat_summ_stat.pkl')
    joblib.dump(value=overall_aa_feat_summ_stat_dict, filename=dict_file_nm_path, compress=3)
    print("\n The overall_aa_feat_summ_stat_dict is saved as: " + dict_file_nm_path)
    print('In derive_overall_5_summary_stats_for_aa_feat() method - End')


if __name__ == '__main__':
    root_path = os.path.join('/project/root/directory/path/here')
    root_path = os.path.join('/scratch/pralaycs/Shubh_Working_Remote/PPI_Wkspc/PPI_Code/mat_p2ip_prj_working')

    max_prot_len = 800
    spec_type = 'yeast'  # human, ecoli, fly, mouse, worm, yeast 
    # derive_5_summary_stats_for_aa_feat(root_path, spec_type = spec_type)

    derive_overall_5_summary_stats_for_aa_feat(root_path)

