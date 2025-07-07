import os

import joblib
import numpy as np
import pandas as pd
from Bio import PDB


def calc_residue_dist(residue_one, residue_two) :
    """Returns the C-alpha distance between two residues"""
    diff_vector  = residue_one["CA"].coord - residue_two["CA"].coord
    return np.sqrt(np.sum(diff_vector * diff_vector))


def calc_dist_matrix(chain_one, chain_two) :
    """Returns a matrix of C-alpha distances between two chains"""
    # use only standard residue
    chain_1_lst, chain_2_lst = [], []
    for residue in chain_one:
        three_letter_res_name = residue.get_resname()
        is_standard_aa = PDB.Polypeptide.is_aa(three_letter_res_name, standard = True)
        # print(f"residue: {three_letter_res_name} : standard: {is_standard_aa}")
        if(is_standard_aa):
            chain_1_lst.append(residue)
    for residue in chain_two:
        three_letter_res_name = residue.get_resname()
        is_standard_aa = PDB.Polypeptide.is_aa(three_letter_res_name, standard = True)
        # print(f"residue: {three_letter_res_name} : standard: {is_standard_aa}")
        if(is_standard_aa):
            chain_2_lst.append(residue)

    answer = np.zeros((len(chain_1_lst), len(chain_2_lst)), np.float64)
    for row, residue_one in enumerate(chain_1_lst) :
        for col, residue_two in enumerate(chain_2_lst) :
            # print(f"row: {row} :: col: {col}")
            answer[row, col] = calc_residue_dist(residue_one, residue_two)
    return answer


def create_prot_gt_contact_map(root_path='./', contact_threshold = 8.0, docking_version = '4_0'):
    """
    Create ground-truth interaction map for the specific pair of chains for the protein complex 

    Args:
        root_path (str): root_path. Defaults to './'.
        contact_threshold (float, optional): Contact threshold in Angstrom. Defaults to 8.0.

    Returns:
        None
    """
    print('\n#### inside the create_prot_gt_contact_map() method - Start\n')
    # The directory for the saved PDB files
    pdb_file_location = os.path.join(root_path, f'dataset/preproc_data_docking_BM_{docking_version}/pdb_files')
    # The directory to save the output interaction maps as pickle (pkl) file.
    output_location = os.path.join(root_path, f'dataset/preproc_data_docking_BM_{docking_version}/prot_gt_contact_map')

    # read the dock_test.tsv file in a pandas dataframe
    doc_test_tsv_fl_nm_loc = os.path.join(root_path, f'dataset/preproc_data_docking_BM_{docking_version}/derived_feat/dock', 'dock_test.tsv')
    dock_test_df = pd.read_csv(doc_test_tsv_fl_nm_loc, sep='\t', header=None, names=['prot_1_id', 'prot_2_id', 'test_labels'])
    
    # iterate over the dock_test_df and for each pair of participating proteins in each row derive the interaction map
    for index, row in dock_test_df.iterrows():
        ###############
        # remove 1QFW_I, 1QFW_M, 1FC2, 1GPW, 1OFU_X, 1US7, 2CFH, 1IBR, 2IDO for missing CA for BM_4_0
        # remove 1US7, 1GPW, 1OFU_X, 1FC2, 2CFH, 2IDO, 1IBR for missing CA for BM_5_5
        # if(index <= 376): continue
        ###############
        print(f"\n ################# starting {index}-th row out of {dock_test_df.shape[0]-1}\n")
        prot_1_id, prot_2_id = row['prot_1_id'], row['prot_2_id']
        # protein id has the format of [protein_name]_[chain_name]
        protein_name, chain_1_name = prot_1_id.split('_')
        protein_name, chain_2_name = prot_2_id.split('_')
        print(f"protein_name: {protein_name} :: chain_1_name: {chain_1_name} :: chain_2_name: {chain_2_name}")

        # now, read the PDB file corresponding to the protein_name
        pdb_file_path = os.path.join(pdb_file_location, f"{protein_name}.pdb")
        # create a PDBParser object
        parser = PDB.PDBParser(QUIET=False)
        # parse already saved PDB file
        structure = parser.get_structure(protein_name, pdb_file_path)
        # extract the model from the structure (usually there's only one model)
        model = structure[0]
        # calculate distance matrix between chain_1 and chain_2 by invoking calc_dist_matrix() method
        dist_matrix = calc_dist_matrix(model[chain_1_name], model[chain_2_name])
        # now save the dist_matrix as pkl file
        print('\n saving dist_matrix as pkl')
        dist_matrix_location = os.path.join(output_location, f"dist_matrix_{protein_name}_{chain_1_name}_{chain_2_name}.pkl")
        joblib.dump(dist_matrix, dist_matrix_location)

        # create ground-truth contact_map
        # compare each member of dist_matrix with the contact_threshold.
        # If the member is greater than the contact_threshold then the member is replaced with the contact_threshold.
        # processed_matrix = np.where(dist_matrix > contact_threshold, contact_threshold, dist_matrix)

        # perform a min-max normalization in a reverse way so that each member value will be within 0.0 and 1 .0 and 
        # the minimum member value will be treated as having highest probability i.e. 1.0 and the maximum value will be 
        # treated as a minimum probability i.e. 0 and the member value in between minimum and maximum will be proportionately mapped.
        min_val = dist_matrix.min()
        max_val = dist_matrix.max()
        gt_contact_map = (max_val - dist_matrix) / (max_val - min_val)
        # now save the gt_contact_map as pkl file
        print('\n saving gt_contact_map as pkl')
        gt_contact_map_location = os.path.join(output_location, f"gt_contact_map_{protein_name}_{chain_1_name}_{chain_2_name}.pkl")
        joblib.dump(gt_contact_map, gt_contact_map_location)
    # end of for loop: for index, row in dock_test_df.iterrows():
    print('\n#### inside the create_prot_gt_contact_map() method - End\n')


if __name__ == "__main__":
    root_path = os.path.join('/project/root/directory/path/here')
    
    # root_path = os.path.join('/home/Shubh_Working_Ubuntu/Workspaces/PPI_Wkspc/PPI_Code/mat_p2ip_prj_working')

    docking_version = '4_0'  # '4_0', '5_5'
    create_prot_gt_contact_map(root_path, docking_version = docking_version)
