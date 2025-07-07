import os
import sys
from pathlib import Path

import joblib
import pandas as pd

path_root = Path(__file__).parents[2]  # upto 'codebase' folder
sys.path.insert(0, str(path_root))
# print(sys.path)

from utils import preproc_tl_util_DS_img


# parse the content of allSeqs.fasta and create a dataframe containing 'prot_id' and 'seq' columns
def parse_DS_to_fasta(root_path='./', spec_type = 'human', restricted_len=400):
    print('\n########## spec_type: ' + str(spec_type))
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

    # create dataframe
    DS_seq_df = pd.DataFrame(data = {'prot_id': prot_lst, 'seq': seq_lst})
    DS_seq_df['seq_len'] = DS_seq_df['seq'].str.len()

    # save DS_seq_df
    DS_seq_df.to_csv(os.path.join(root_path, 'dataset/preproc_data_DS/seqs', 'DS_' + spec_type + '_seq_orig.csv'), index=False)

    # Apply length restriction
    len_restricted_DS_seq_df = DS_seq_df[DS_seq_df['seq_len'] <= 400]

    # save len_restricted_DS_seq_df
    len_restricted_DS_seq_df.to_csv(os.path.join(root_path, 'dataset/preproc_data_DS/seqs', 'DS_' + spec_type + '_seq_len' + str(restricted_len) + '.csv'), index=False)
# End of parse_DS_to_fasta() method


def calc_class_dist(root_path='./', restricted_len=400):
    spec_type_lst = ['ecoli', 'fly', 'mouse', 'worm', 'yeast', 'human']
    # spec_type_lst = ['fly']
    overall_class_dist_dict_lst = []
    for spec_type in spec_type_lst:
        print('\n########## spec_type: ' + str(spec_type))
        print('\n ########## fetch the candidate pairs ######g#### ')
        DS_seq_df = pd.read_csv(os.path.join(root_path,'dataset/preproc_data_DS/pairs', 'DS_test_pairs_' + spec_type + '_seq_len' + str(restricted_len) + '.csv'))
        # Calculate the number of 1s in 'label' column
        num_of_ones = DS_seq_df[DS_seq_df['label'] == 1.0]['label'].count()
        num_of_zeroes = DS_seq_df.shape[0] - num_of_ones
        class_ratio = num_of_zeroes / num_of_ones
        class_dist_dict = {'spec_type': spec_type, 'num_of_ones': num_of_ones, 'num_of_zeroes': num_of_zeroes, 'class_ratio': class_ratio}
        overall_class_dist_dict_lst.append(class_dist_dict)
    # End of for loop: for spec_type in spec_type_lst:

    # Convert overall_class_dist_dict_lst into a pandas dataframe and save it as a csv
    overall_class_dist_df = pd.DataFrame(overall_class_dist_dict_lst)
    overall_class_dist_df.to_csv(os.path.join(root_path, 'dataset/preproc_data_DS/pairs/', 'overall_class_dist.csv'), index=False)
# End of calc_class_distribution() method


# Extract features using protTrans model (tl model)
def prepare_tl_feat_for_DS_seq_for_img_v2(root_path='./', protTrans_model_path='./', protTrans_model_name = 'prot_t5_xl_uniref50', spec_type = 'human', restricted_len=400):
    print('\n########## spec_type: ' + str(spec_type))
    # fetch the already saved DS_sequence df
    print('\n ########## fetch the already saved DS_sequence df ######g#### ')
    DS_seq_df = pd.read_csv(os.path.join(root_path,'dataset/preproc_data_DS/seqs', 'DS_' + spec_type + '_seq_len' + str(restricted_len) + '.csv'))
    # extract features using the protTrans model for the DS_sequence list
    print('\n ########## extract features using the protTrans model (tl model) for the DS_sequence list ########## ')
    features_lst, features_2d_lst  = preproc_tl_util_DS_img.extract_feat_from_protTrans(DS_seq_df['seq'].tolist(), protTrans_model_path, protTrans_model_name, spec_type)
    
    for index, row in DS_seq_df.iterrows():
        prot_id = row['prot_id']
        prot_2dArr = features_2d_lst[index]
        # save prot_2dArr as a pkl file
        prot_2dArr_file_nm_loc = os.path.join(root_path, 'dataset/preproc_data_DS/tl_2d_feat_dict_dump_img', spec_type, f"prot_id_{prot_id}.pkl")
        joblib.dump(value=prot_2dArr, filename=prot_2dArr_file_nm_loc, compress=0)
    # end of for loop: for index, row in DS_seq_df.iterrows():

    print("\n######## cleaning all the intermediate stuffs - START ########")
    # remove all the intermediate files in the 'temp_result' and 'temp_per_prot_emb_result' directories which
    # were used in extract_feat_from_preloaded_protTrans() method
    temp_result_dir = os.path.join(root_path, 'temp_result_' + spec_type) 
    for temp_file in os.listdir(temp_result_dir):
        os.remove(os.path.join(temp_result_dir, temp_file))
    temp_per_prot_emb_result_dir = os.path.join(root_path, 'temp_per_prot_emb_result_' + spec_type) 
    for temp_file in os.listdir(temp_per_prot_emb_result_dir):
        os.remove(os.path.join(temp_per_prot_emb_result_dir, temp_file))
    print("######## cleaning all the intermediate stuffs - DONE ########")



if __name__ == '__main__':
    root_path = os.path.join('/project/root/directory/path/here')
    

    restricted_len = 400
    # spec_type_lst = ['ecoli', 'fly', 'mouse', 'worm', 'yeast', 'human']
    spec_type_lst = ['fly']
    for spec_type in spec_type_lst:
        # ### parse_DS_to_fasta(root_path=root_path, spec_type=spec_type, restricted_len=restricted_len)
        prepare_tl_feat_for_DS_seq_for_img_v2(root_path
                                        ,protTrans_model_path=os.path.join(root_path, '../ProtTrans_Models/')
                                        , protTrans_model_name = 'prot_t5_xl_uniref50'
                                        , spec_type = spec_type, restricted_len=restricted_len)
    # End of for loop: for spec_type in spec_type_lst:

    # ### calc_class_dist(root_path=root_path, restricted_len=restricted_len)


