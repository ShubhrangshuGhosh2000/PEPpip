import os
import sys
from pathlib import Path
import joblib
import pandas as pd
path_root = Path(__file__).parents[2]  
sys.path.insert(0, str(path_root))
from codebase.utils import preproc_protT5_util_dock_img


def prepare_tl_feat_for_dock_seq_for_img(root_path='./', protTrans_model_path='./', protTrans_model_name = 'prot_t5_xl_uniref50'
                                         , docking_version = '5_5', restricted_len = 400):
    dock_seq_df = pd.read_csv(os.path.join(root_path,f'dataset/preproc_data_docking_BM_{docking_version}', f'dock_seq_lenLimit_{restricted_len}.csv'))
    features_lst, features_2d_lst  = preproc_protT5_util_dock_img.extract_feat_from_protTrans(dock_seq_df['seq'].tolist(), protTrans_model_path, protTrans_model_name)
    for index, row in dock_seq_df.iterrows():
        prot_id = row['prot_id']
        prot_2dArr = features_2d_lst[index]
        prot_2dArr_file_nm_loc = os.path.join(root_path, f'dataset/preproc_data_docking_BM_{docking_version}/dock_tl_2d_feat_dict_dump_img',  f"prot_id_{prot_id}.pkl")
        joblib.dump(value=prot_2dArr, filename=prot_2dArr_file_nm_loc, compress=0)
    temp_result_dir = os.path.join(root_path, 'temp_result') 
    for temp_file in os.listdir(temp_result_dir):
        os.remove(os.path.join(temp_result_dir, temp_file))
    temp_per_prot_emb_result_dir = os.path.join(root_path, 'temp_per_prot_emb_result') 
    for temp_file in os.listdir(temp_per_prot_emb_result_dir):
        os.remove(os.path.join(temp_per_prot_emb_result_dir, temp_file))


if __name__ == '__main__':
    root_path = os.path.join('/project/root/directory/path/here')
    docking_version = '5_5'  
    restricted_len = 400
    prepare_tl_feat_for_dock_seq_for_img(root_path, protTrans_model_path=os.path.join(root_path, '../ProtTrans_Models/')
                                         , protTrans_model_name = 'prot_t5_xl_uniref50'
                                         , docking_version = docking_version, restricted_len = restricted_len)
