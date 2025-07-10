import os
import sys
from pathlib import Path
import pandas as pd
path_root = Path(__file__).parents[2]  
sys.path.insert(0, str(path_root))
from codebase.utils import preproc_esmc_util_dock_img


def prepare_tl_esmc_feat_for_dock_seq_for_img(root_path='./', esmc_model_path='./', esmc_model_name = 'esmc_600m', docking_version = '5_5', restricted_len = 400):
    dock_seq_df = pd.read_csv(os.path.join(root_path,f'dataset/preproc_data_docking_BM_{docking_version}', f'dock_seq_lenLimit_{restricted_len}.csv'))
    preproc_esmc_util_dock_img.extract_feat_from_esmc(dock_seq_df['prot_id'].tolist(), dock_seq_df['seq'].tolist(), esmc_model_path, esmc_model_name, docking_version)


if __name__ == '__main__':
    root_path = os.path.join('/project/root/directory/path/here')
    restricted_len = 400
    docking_version = '5_5'  
    restricted_len = 400
    prepare_tl_esmc_feat_for_dock_seq_for_img(root_path
                                    ,esmc_model_path=os.path.join(root_path, '../esmc_Models/')
                                    , esmc_model_name = 'esmc_600m'
                                    , docking_version = docking_version, restricted_len = restricted_len)
