import os
import sys
from pathlib import Path
import pandas as pd
path_root = Path(__file__).parents[2]  
sys.path.insert(0, str(path_root))
from codebase.utils import preproc_prose_util_DS_img


def prepare_tl_struct_feat_for_DS_seq_for_img(root_path='./', prose_model_path='./', prose_model_name = 'prose_mt_3x1024', spec_type = 'human', restricted_len=400):
    DS_seq_df = pd.read_csv(os.path.join(root_path,'dataset/preproc_data_DS/seqs', 'DS_' + spec_type + '_seq_len' + str(restricted_len) + '.csv'))
    preproc_prose_util_DS_img.extract_feat_from_prose(DS_seq_df['prot_id'].tolist(), DS_seq_df['seq'].tolist(), prose_model_path, prose_model_name, spec_type)


if __name__ == '__main__':
    root_path = os.path.join('/project/root/directory/path/here')
    restricted_len = 400
    spec_type_lst = ['ecoli', 'fly', 'mouse', 'worm', 'yeast', 'human']
    for spec_type in spec_type_lst:
        prepare_tl_struct_feat_for_DS_seq_for_img(root_path
                                        ,prose_model_path=os.path.join(root_path, '../prose_Models/')
                                        , prose_model_name = 'prose_mt_3x1024'
                                        , spec_type = spec_type, restricted_len=restricted_len)
