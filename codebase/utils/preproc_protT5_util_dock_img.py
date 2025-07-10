import gc
import glob
import os
import re
import joblib
import numpy as np
import torch
from transformers import T5EncoderModel, T5Tokenizer

root_path = os.path.join('/project/root/directory/path/here')


def load_protTrans_tl_model(protTrans_model_path='./', protTrans_model_name = 'prot_t5_xl_uniref50'):
    protTrans_model_name_with_path = os.path.join(protTrans_model_path, protTrans_model_name)
    tokenizer = T5Tokenizer.from_pretrained(protTrans_model_name_with_path, do_lower_case=False )
    model = T5EncoderModel.from_pretrained(protTrans_model_name_with_path)
    gc.collect()
    return (model, tokenizer) 


def extract_feat_from_protTrans(seq_lst, protTrans_model_path='./', protTrans_model_name = 'prot_t5_xl_uniref50'):
    model, tokenizer = load_protTrans_tl_model(protTrans_model_path, protTrans_model_name)
    features, features_2d = extract_feat_from_preloaded_protTrans(seq_lst, model, tokenizer, device='gpu')
    return (features, features_2d)


def extract_feat_from_preloaded_protTrans(seq_lst, model, tokenizer, device='gpu'):
    compact_seq_lst = [seq.replace(" ", "") for seq in seq_lst]
    seq_lst_with_space = [' '.join(seq) for seq in compact_seq_lst]
    if(device == 'cpu'):
        device = 'cpu'
    else:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model = model.eval()
    seq_lst_with_space = [re.sub(r"[UZOB]", "X", sequence) for sequence in seq_lst_with_space]
    part_size = 1
    part_lst = [seq_lst_with_space[i: i + part_size] for i in range(0, len(seq_lst_with_space), part_size)]
    tot_no_of_parts = len(part_lst)
    temp_feat_lst = []  
    temp_result_dir = os.path.join(root_path, 'temp_result')  
    temp_per_prot_emb_result_dir = os.path.join(root_path, 'temp_per_prot_emb_result')  
    latest_pkl_file_index = -1
    all_pkl_fl_nm_lst = glob.glob(os.path.join(temp_result_dir, 'feat_lst_*.pkl'))
    if(len(all_pkl_fl_nm_lst) > 0):  
        pkl_file_ind_lst = [int(indiv_fl_nm.replace(os.path.join(temp_result_dir, 'feat_lst_'), '').replace('.pkl', '')) for indiv_fl_nm in all_pkl_fl_nm_lst]
        latest_pkl_file_index = max(pkl_file_ind_lst)
    cuda_error = False
    for itr in range(latest_pkl_file_index + 1, tot_no_of_parts):
        indiv_part_lst = part_lst[itr]
        ids = tokenizer.batch_encode_plus(indiv_part_lst, add_special_tokens=True, padding=True)
        input_ids = torch.tensor(ids['input_ids']).to(device)
        attention_mask = torch.tensor(ids['attention_mask']).to(device)
        try:
            with torch.no_grad():
                embedding = model(input_ids=input_ids,attention_mask=attention_mask)
            embedding = embedding.last_hidden_state.cpu().numpy()
        except Exception as ex:
            cuda_error = True  
            break  
        for seq_num in range(len(embedding)):
            seq_len = (attention_mask[seq_num] == 1).sum()
            seq_emd = embedding[seq_num][:seq_len-1]
            temp_feat_lst.append(seq_emd)
        if((itr % 1 == 0) or (itr in [(tot_no_of_parts - 1)])):
            filename = os.path.join(temp_result_dir, 'feat_lst_' + str(itr) + '.pkl')
            joblib.dump(value=temp_feat_lst, filename=filename, compress=0)
            temp_feat_lst = []
        if(device == 'cpu'):
            return extract_feat_from_preloaded_protTrans(seq_lst, model, tokenizer, device='gpu')
    if(cuda_error):
        return extract_feat_from_preloaded_protTrans(seq_lst, model, tokenizer, device='cpu')
    features = []  
    features_2d = []  
    loop_start_index = 0  
    for itr in range(loop_start_index, tot_no_of_parts):
        if((itr % 1 == 0) or (itr in [(tot_no_of_parts - 1)])):
            temp_feat_lst = joblib.load(os.path.join(temp_result_dir, 'feat_lst_' + str(itr) + '.pkl'))
            for prot_residue_embed in temp_feat_lst:  
                features_2d.append(prot_residue_embed)
                features.append(np.apply_along_axis(np.median, axis=0, arr=prot_residue_embed))  
            if(itr % 20000 == 0): 
                temp_feat_lst_file_nm = os.path.join(temp_per_prot_emb_result_dir, 'features_' + str(loop_start_index) + '_' + str(itr) + '.pkl')
                joblib.dump(value=features, filename=temp_feat_lst_file_nm, compress=0)
                temp_feat_2d_lst_file_nm = os.path.join(temp_per_prot_emb_result_dir, 'features_2d_' + str(loop_start_index) + '_' + str(itr) + '.pkl')
                joblib.dump(value=features_2d, filename=temp_feat_2d_lst_file_nm, compress=0)
    joblib.dump(value=features, filename=os.path.join(temp_per_prot_emb_result_dir, 'features.pkl'), compress=0)
    joblib.dump(value=features_2d, filename=os.path.join(temp_per_prot_emb_result_dir, 'features_2d.pkl'), compress=0)
    return (features, features_2d)


if __name__ == '__main__':
    pass
