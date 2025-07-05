import os
import sys
from pathlib import Path

import pandas as pd

path_root = Path(__file__).parents[2]  # upto 'codebase' folder
sys.path.insert(0, str(path_root))
# print(sys.path)

from utils import preproc_tl_struct_util_DS_img


# Extract features using prose model
def prepare_tl_struct_feat_for_DS_seq_for_img(root_path='./', prose_model_path='./', prose_model_name = 'prose_mt_3x1024', spec_type = 'human', restricted_len=400):
    print('\n########## spec_type: ' + str(spec_type))
    # fetch the already saved DS_sequence df
    print('\n ########## fetch the already saved DS_sequence df ######g#### ')
    DS_seq_df = pd.read_csv(os.path.join(root_path,'dataset/preproc_data_DS/seqs', 'DS_' + spec_type + '_seq_len' + str(restricted_len) + '.csv'))
    # extract features using the prose model for the DS_sequence list
    print('\n ########## extract features using the prose model (tl_struct model) for the DS_sequence list ########## ')
    preproc_tl_struct_util_DS_img.extract_feat_from_prose(DS_seq_df['prot_id'].tolist(), DS_seq_df['seq'].tolist(), prose_model_path, prose_model_name, spec_type)
    print("######## prepare_tl_struct_feat_for_DS_seq_for_img - DONE ########")


if __name__ == '__main__':
    root_path = os.path.join('/project/root/directory/path/here')
    root_path = os.path.join('/scratch/pralaycs/Shubh_Working_Remote/PPI_Wkspc/PPI_Code/mat_p2ip_prj_working')
    restricted_len = 400

    # spec_type_lst = ['ecoli', 'fly', 'mouse', 'worm', 'yeast', 'human']
    spec_type_lst = ['fly', 'mouse', 'worm']
    for spec_type in spec_type_lst:
        prepare_tl_struct_feat_for_DS_seq_for_img(root_path
                                        ,prose_model_path=os.path.join(root_path, '../prose_Models/')
                                        , prose_model_name = 'prose_mt_3x1024'
                                        , spec_type = spec_type, restricted_len=restricted_len)
