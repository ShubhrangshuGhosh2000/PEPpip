import os
import sys
from pathlib import Path
import joblib
import pandas as pd
path_root = Path(__file__).parents[2]  
sys.path.insert(0, str(path_root))
from utils import preproc_tl_util_DS_img


def parse_DS_to_fasta(root_path='./', spec_type = 'human', restricted_len=400):
    f = open(os.path.join(root_path, 'dataset/orig_data_DS/seqs', spec_type + '.fasta'))
    prot_lst, seq_lst = [], []
    idx = 0
    for line in f:
        if idx == 0:
            prot_lst.append(line.strip().strip('>'))
        elif idx == 1:
            seq_lst.append(line.strip())
        idx += 1
        idx = idx % 2
    f.close()
    DS_seq_df = pd.DataFrame(data = {'prot_id': prot_lst, 'seq': seq_lst})
    DS_seq_df['seq_len'] = DS_seq_df['seq'].str.len()
    DS_seq_df.to_csv(os.path.join(root_path, 'dataset/preproc_data_DS/seqs', 'DS_' + spec_type + '_seq_orig.csv'), index=False)
    len_restricted_DS_seq_df = DS_seq_df[DS_seq_df['seq_len'] <= 400]
    len_restricted_DS_seq_df.to_csv(os.path.join(root_path, 'dataset/preproc_data_DS/seqs', 'DS_' + spec_type + '_seq_len' + str(restricted_len) + '.csv'), index=False)


def prepare_tl_feat_for_DS_seq_for_img_v2(root_path='./', protTrans_model_path='./', protTrans_model_name = 'prot_t5_xl_uniref50', spec_type = 'human', restricted_len=400):
    DS_seq_df = pd.read_csv(os.path.join(root_path,'dataset/preproc_data_DS/seqs', 'DS_' + spec_type + '_seq_len' + str(restricted_len) + '.csv'))
    features_lst, features_2d_lst  = preproc_tl_util_DS_img.extract_feat_from_protTrans(DS_seq_df['seq'].tolist(), protTrans_model_path, protTrans_model_name, spec_type)
    for index, row in DS_seq_df.iterrows():
        prot_id = row['prot_id']
        prot_2dArr = features_2d_lst[index]
        prot_2dArr_file_nm_loc = os.path.join(root_path, 'dataset/preproc_data_DS/tl_2d_feat_dict_dump_img', spec_type, f"prot_id_{prot_id}.pkl")
        joblib.dump(value=prot_2dArr, filename=prot_2dArr_file_nm_loc, compress=0)
    temp_result_dir = os.path.join(root_path, 'temp_result_' + spec_type) 
    for temp_file in os.listdir(temp_result_dir):
        os.remove(os.path.join(temp_result_dir, temp_file))
    temp_per_prot_emb_result_dir = os.path.join(root_path, 'temp_per_prot_emb_result_' + spec_type) 
    for temp_file in os.listdir(temp_per_prot_emb_result_dir):
        os.remove(os.path.join(temp_per_prot_emb_result_dir, temp_file))


if __name__ == '__main__':
    root_path = os.path.join('/project/root/directory/path/here')
    restricted_len = 400
    spec_type_lst = ['ecoli', 'fly', 'mouse', 'worm', 'yeast', 'human']
    for spec_type in spec_type_lst:
        prepare_tl_feat_for_DS_seq_for_img_v2(root_path
                                        ,protTrans_model_path=os.path.join(root_path, '../ProtTrans_Models/')
                                        , protTrans_model_name = 'prot_t5_xl_uniref50'
                                        , spec_type = spec_type, restricted_len=restricted_len)
