import os
import sys
from pathlib import Path

import pandas as pd

path_root = Path(__file__).parents[2]  # upto 'codebase' folder
sys.path.insert(0, str(path_root))
# print(sys.path)

from utils import preproc_tl_esmc_util_dock_img


# Extract features using emc-Cambrian model
def prepare_tl_esmc_feat_for_dock_seq_for_img(root_path='./', esmc_model_path='./', esmc_model_name = 'esmc_600m', docking_version = '4_0', restricted_len = 400):
    print('\n########## docking_version: ' + str(docking_version))
    # fetch the already saved dock_sequence df
    print('\n ########## fetch the already saved dock_sequence df ######g#### ')
    dock_seq_df = pd.read_csv(os.path.join(root_path,f'dataset/preproc_data_docking_BM_{docking_version}', f'dock_seq_lenLimit_{restricted_len}.csv'))
    # extract features using the esmc model for the dock_sequence list
    print('\n ########## extract features using the esmc model (tl_esmc model) for the dock_sequence list ########## ')
    preproc_tl_esmc_util_dock_img.extract_feat_from_esmc(dock_seq_df['prot_id'].tolist(), dock_seq_df['seq'].tolist(), esmc_model_path, esmc_model_name, docking_version)
    print("######## prepare_tl_esmc_feat_for_dock_seq_for_img - DONE ########")


if __name__ == '__main__':
    root_path = os.path.join('/project/root/directory/path/here')
    
    restricted_len = 400

    docking_version = '4_0'  # '4_0', '5_5'
    restricted_len = 400
    prepare_tl_esmc_feat_for_dock_seq_for_img(root_path
                                    ,esmc_model_path=os.path.join(root_path, '../esmc_Models/')
                                    , esmc_model_name = 'esmc_600m'
                                    , docking_version = docking_version, restricted_len = restricted_len)
