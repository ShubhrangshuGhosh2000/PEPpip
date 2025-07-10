import os
import sys
from pathlib import Path
path_root = Path(__file__).parents[1]  
sys.path.insert(0, str(path_root))
import joblib
import torch
import glob
from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig

root_path = os.path.join('/project/root/directory/path/here')


def load_esmc_model(esmc_model_path='./', esmc_model_name = 'esmc_600m'):
    model = ESMC.from_pretrained(esmc_model_name)
    return model


def extract_feat_from_esmc(prot_id_lst, seq_lst, esmc_model_path='./', esmc_model_name = 'esmc_600m', docking_version = '5_5'):
    model = load_esmc_model(esmc_model_path, esmc_model_name)
    extract_feat_from_preloaded_esmc(prot_id_lst, seq_lst, model, docking_version, device='gpu')


def extract_feat_from_preloaded_esmc(prot_id_lst, seq_lst, model, docking_version, device='gpu'):
    compact_seq_lst = [seq.replace(" ", "") for seq in seq_lst]
    if(device == 'cpu'):
        device = 'cpu'
    else:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    part_size = 1
    part_lst = [compact_seq_lst[i: i + part_size] for i in range(0, len(compact_seq_lst), part_size)]
    tot_no_of_parts = len(part_lst)
    spec_result_dir = os.path.join(root_path, 'dataset/preproc_data_docking_BM_' + str(docking_version), 'dock_tl_esmc_2d_feat_dict_dump_img')
    latest_pkl_file_index = -1
    all_pkl_fl_nm_lst = glob.glob(os.path.join(spec_result_dir, 'prot_id_*.pkl'))
    if(len(all_pkl_fl_nm_lst) > 0):  
        latest_pkl_file_index = len(all_pkl_fl_nm_lst) -2
    cuda_error = False
    for itr in range(latest_pkl_file_index + 1, tot_no_of_parts):
        x = part_lst[itr][0]  
        z = None  
        x = x.upper()
        try:
            protein = ESMProtein(sequence=x)
            protein_tensor = model.encode(protein)
            logits_output = model.logits(protein_tensor, LogitsConfig(sequence=True, return_embeddings=True))
            z = logits_output.embeddings
            z = z.cpu().numpy()
        except Exception as ex:
            cuda_error = True  
            break  
        prot_id = prot_id_lst[itr]
        prot_2dArr_file_nm_loc = os.path.join(spec_result_dir, f"prot_id_{prot_id}.pkl")
        z = z.squeeze()
        joblib.dump(value=z, filename=prot_2dArr_file_nm_loc, compress=3)
        if(device == 'cpu'):
            return extract_feat_from_preloaded_esmc(prot_id_lst, seq_lst, model, docking_version, device='gpu')
    if(cuda_error):
        return extract_feat_from_preloaded_esmc(prot_id_lst, seq_lst, model, docking_version, device='cpu')


if __name__ == '__main__':
    pass
