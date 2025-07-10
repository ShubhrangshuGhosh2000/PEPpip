import os
import sys
from pathlib import Path
path_root = Path(__file__).parents[1]  
sys.path.insert(0, str(path_root))
import joblib
import numpy as np
import torch
import glob
from utils.prose.multitask import ProSEMT
from utils.prose.alphabets import Uniprot21

root_path = os.path.join('/project/root/directory/path/here')


def load_prose_tl_struct_model(prose_model_path='./', prose_model_name = 'prose_mt_3x1024'):
    prose_model_name_with_path = os.path.join(prose_model_path, f"{prose_model_name}.sav")
    model = ProSEMT.load_pretrained(prose_model_name_with_path)
    return (model) 


def extract_feat_from_prose(prot_id_lst, seq_lst, prose_model_path='./', prose_model_name = 'prose_mt_3x1024', spec_type = 'human'):
    model = load_prose_tl_struct_model(prose_model_path, prose_model_name)
    extract_feat_from_preloaded_prose(prot_id_lst, seq_lst, model, spec_type, device='gpu')


def extract_feat_from_preloaded_prose(prot_id_lst, seq_lst, model, spec_type, device='gpu'):
    compact_seq_lst = [seq.replace(" ", "") for seq in seq_lst]
    if(device == 'cpu'):
        device = 'cpu'
    else:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model = model.eval()
    part_size = 1
    part_lst = [compact_seq_lst[i: i + part_size] for i in range(0, len(compact_seq_lst), part_size)]
    tot_no_of_parts = len(part_lst)
    spec_result_dir = os.path.join(root_path, 'dataset/preproc_data_DS/tl_struct_2d_feat_dict_dump_img', spec_type)
    latest_pkl_file_index = -1
    all_pkl_fl_nm_lst = glob.glob(os.path.join(spec_result_dir, 'prot_id_*.pkl'))
    if(len(all_pkl_fl_nm_lst) > 0):  
        latest_pkl_file_index = len(all_pkl_fl_nm_lst) -2
    cuda_error = False
    for itr in range(latest_pkl_file_index + 1, tot_no_of_parts):
        x = part_lst[itr][0]  
        z = None  
        if len(x) == 0:
            n = model.embedding.proj.weight.size(1)
            z = np.zeros((1,n), dtype=np.float32)
        else:
            alphabet = Uniprot21()
            x = x.upper()
            x = x.encode('utf-8')
            x = alphabet.encode(x)
            x = torch.from_numpy(x)
            x = x.to(device)
            try:
                with torch.no_grad():
                    x = x.long().unsqueeze(0)
                    z = model.transform(x)
                    z = z.cpu().numpy()
            except Exception as ex:
                cuda_error = True  
                break  
        prot_id = prot_id_lst[itr]
        prot_2dArr_file_nm_loc = os.path.join(spec_result_dir, f"prot_id_{prot_id}.pkl")
        z = z.squeeze()
        joblib.dump(value=z, filename=prot_2dArr_file_nm_loc, compress=3)
        if(device == 'cpu'):
            return extract_feat_from_preloaded_prose(prot_id_lst, seq_lst, model, spec_type, device='gpu')
    if(cuda_error):
        return extract_feat_from_preloaded_prose(prot_id_lst, seq_lst, model, spec_type, device='cpu')


if __name__ == '__main__':
    pass
