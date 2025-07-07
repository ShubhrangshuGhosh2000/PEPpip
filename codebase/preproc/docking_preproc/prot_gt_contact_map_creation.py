import os
import joblib
import numpy as np
import pandas as pd
from Bio import PDB


def calc_residue_dist(residue_one, residue_two) :
    diff_vector  = residue_one["CA"].coord - residue_two["CA"].coord
    return np.sqrt(np.sum(diff_vector * diff_vector))


def calc_dist_matrix(chain_one, chain_two) :
    chain_1_lst, chain_2_lst = [], []
    for residue in chain_one:
        three_letter_res_name = residue.get_resname()
        is_standard_aa = PDB.Polypeptide.is_aa(three_letter_res_name, standard = True)
        if(is_standard_aa):
            chain_1_lst.append(residue)
    for residue in chain_two:
        three_letter_res_name = residue.get_resname()
        is_standard_aa = PDB.Polypeptide.is_aa(three_letter_res_name, standard = True)
        if(is_standard_aa):
            chain_2_lst.append(residue)
    answer = np.zeros((len(chain_1_lst), len(chain_2_lst)), np.float64)
    for row, residue_one in enumerate(chain_1_lst) :
        for col, residue_two in enumerate(chain_2_lst) :
            answer[row, col] = calc_residue_dist(residue_one, residue_two)
    return answer


def create_prot_gt_contact_map(root_path='./', contact_threshold = 8.0, docking_version = '5_5'):
    pdb_file_location = os.path.join(root_path, f'dataset/preproc_data_docking_BM_{docking_version}/pdb_files')
    output_location = os.path.join(root_path, f'dataset/preproc_data_docking_BM_{docking_version}/prot_gt_contact_map')
    doc_test_tsv_fl_nm_loc = os.path.join(root_path, f'dataset/preproc_data_docking_BM_{docking_version}/derived_feat/dock', 'dock_test.tsv')
    dock_test_df = pd.read_csv(doc_test_tsv_fl_nm_loc, sep='\t', header=None, names=['prot_1_id', 'prot_2_id', 'test_labels'])
    for index, row in dock_test_df.iterrows():
        prot_1_id, prot_2_id = row['prot_1_id'], row['prot_2_id']
        protein_name, chain_1_name = prot_1_id.split('_')
        protein_name, chain_2_name = prot_2_id.split('_')
        pdb_file_path = os.path.join(pdb_file_location, f"{protein_name}.pdb")
        parser = PDB.PDBParser(QUIET=False)
        structure = parser.get_structure(protein_name, pdb_file_path)
        model = structure[0]
        dist_matrix = calc_dist_matrix(model[chain_1_name], model[chain_2_name])
        dist_matrix_location = os.path.join(output_location, f"dist_matrix_{protein_name}_{chain_1_name}_{chain_2_name}.pkl")
        joblib.dump(dist_matrix, dist_matrix_location)
        min_val = dist_matrix.min()
        max_val = dist_matrix.max()
        gt_contact_map = (max_val - dist_matrix) / (max_val - min_val)
        gt_contact_map_location = os.path.join(output_location, f"gt_contact_map_{protein_name}_{chain_1_name}_{chain_2_name}.pkl")
        joblib.dump(gt_contact_map, gt_contact_map_location)


if __name__ == "__main__":
    root_path = os.path.join('/project/root/directory/path/here')
    docking_version = '5_5'  
    create_prot_gt_contact_map(root_path, docking_version = docking_version)
