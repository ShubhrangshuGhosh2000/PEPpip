import os
import sys
from pathlib import Path

path_root = Path(__file__).parents[1]  # upto 'codebase' folder
sys.path.insert(0, str(path_root))
# print(sys.path)

import joblib
import numpy as np
import torch
import glob
from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig


root_path = os.path.join('/project/root/directory/path/here')



# load esm-Cambrian model
def load_esmc_model(esmc_model_path='./', esmc_model_name = 'esmc_600m'):
    print("\n ################## loading esmc model ##################")
    model = ESMC.from_pretrained(esmc_model_name)
    return model


# use esmc model to extract the features for a given list of sequences 
def extract_feat_from_esmc(prot_id_lst, seq_lst, esmc_model_path='./', esmc_model_name = 'esmc_600m', docking_version = '4_0'):
    # first load esmc Model which will be used subsequently
    model = load_esmc_model(esmc_model_path, esmc_model_name)
    # next extract the features using the loaded model
    extract_feat_from_preloaded_esmc(prot_id_lst, seq_lst, model, docking_version, device='gpu')


# use preloaded esmc model to extract the features for a given list of sequences 
def extract_feat_from_preloaded_esmc(prot_id_lst, seq_lst, model, docking_version, device='gpu'):
    print(' Inside extract_feat_from_preloaded_esmc() method - Start')
    print('\n########## docking_version: ' + str(docking_version))
    # first create a list of compact sequences without the whitespace in between the characters
    compact_seq_lst = [seq.replace(" ", "") for seq in seq_lst]
    
    # invoke esmc model for the feature extraction
    print("\n ################## invoking esmc model for the feature extraction ##################")

    # 3. Load the model into the GPU if avilabile
    if(device == 'cpu'):
        device = 'cpu'
    else:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    model = model.to(device)

    # partition size indicates the number of the sequences which will be present in each partition (except might be for the last partition)
    part_size = 1
    print("partitioning the original sequence list into a number of sublists where each sublist (except the last) contains " + str(part_size) + " sequences...")
    part_lst = [compact_seq_lst[i: i + part_size] for i in range(0, len(compact_seq_lst), part_size)]
    tot_no_of_parts = len(part_lst)
    print('original sequence list length = ' + str(len(compact_seq_lst)))
    print('total number of partitions (each of size ' + str(part_size) + ') = ' + str(tot_no_of_parts))
    spec_result_dir = os.path.join(root_path, 'dataset/preproc_data_docking_BM_' + str(docking_version), 'dock_tl_esmc_2d_feat_dict_dump_img')
    # # this latest_pkl_file_index will help in skipping iteration, if it is already done (see below)
    latest_pkl_file_index = -1
    # fetching all the pkl files with name starting with 'prot_id_'
    all_pkl_fl_nm_lst = glob.glob(os.path.join(spec_result_dir, 'prot_id_*.pkl'))
    if(len(all_pkl_fl_nm_lst) > 0):  # temp_result_dir is not empty
        latest_pkl_file_index = len(all_pkl_fl_nm_lst) -2
    print('##### latest_pkl_file_index: ' + str(latest_pkl_file_index))

    cuda_error = False
    # for itr in range(0, tot_no_of_parts):
    for itr in range(latest_pkl_file_index + 1, tot_no_of_parts):
        # print('ítr: ' + str(itr))
        x = part_lst[itr][0]  # indiv_sequence
        z = None  # seq_emd
        x = x.upper()
        try:
            protein = ESMProtein(sequence=x)
            protein_tensor = model.encode(protein)
            logits_output = model.logits(protein_tensor, LogitsConfig(sequence=True, return_embeddings=True))
            z = logits_output.embeddings
            z = z.cpu().numpy()
        except Exception as ex:
            print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@ inside exception@@@@@@@@@@@@@@@@@@')
            err_message = str(ex) + ".....For details, please enable 'verbose' argument as True (if not already enabled) and see the log."
            print("Error message :", err_message, sep="")
            cuda_error = True  # 'cpu' should be used for the next iteration
            break  # jump out of the for loop so that 'cpu' can be used for the next iteration
        if(itr % 25 == 0): print('###### docking_version: ' + docking_version + ' ### completed ' + str(itr) + '-th iteration out of ' + str(tot_no_of_parts-1))
        # save z as a pkl file
        prot_id = prot_id_lst[itr]
        prot_2dArr_file_nm_loc = os.path.join(spec_result_dir, f"prot_id_{prot_id}.pkl")
        # squeeze z to 2d-array (e.g. from (1, 128, 6165) to (128, 6165))
        z = z.squeeze()
        # reduce the dimension of z from (#row, 6165) to (#row, 1024) for the memory efficient saving in the hard-disk
        # z_reduced = z[:, :3000]
        # joblib.dump(value=z_reduced, filename=prot_2dArr_file_nm_loc, compress=3)

        joblib.dump(value=z, filename=prot_2dArr_file_nm_loc, compress=3)
        # cpu should be used only in that iteration for which gpu memory is not sufficient and
        # after that again gpu will be used
        if(device == 'cpu'):
            # call this method again but indicate the device should be gpu
            return extract_feat_from_preloaded_esmc(prot_id_lst, seq_lst, model, docking_version, device='gpu')
    # end of for itr in range(0, tot_no_of_parts)

    # check for the cuda-error
    if(cuda_error):
        # call this method again but indicate device should be cpu
        return extract_feat_from_preloaded_esmc(prot_id_lst, seq_lst, model, docking_version, device='cpu')
    print(' Inside extract_feat_from_preloaded_esmc() method -End')


if __name__ == '__main__':
    # print('in main')
    pass
