import os
import sys
from pathlib import Path

path_root = Path(__file__).parents[2]  # upto 'codebase' folder
sys.path.insert(0, str(path_root))
# print(sys.path)

import joblib
import pandas as pd
import requests
from Bio import PDB

from utils import PPIPUtils

def process_dock_bm_csv_file(dock_bm_csv_file_location, output_location):
    """
    Reads the protein docking benchmark CSV file, extracts specific columns, and processes the data into a Python dictionary.
    The processed data is saved as a pickle (pkl) file.

    Args:
        dock_bm_csv_file_location (str): The location of the protein docking benchmark CSV file.
        output_location (str): The directory to save the output pickle (pkl) file.

    Returns:
        dict: A Python dictionary containing processed data.
    """
    print('\n#### inside the process_dock_bm_csv_file() method - Start')
    # Read the protein docking benchmark CSV file into a DataFrame with the first row as header
    df = pd.read_csv(dock_bm_csv_file_location)

    # Extract the 'Complex' column and convert it into a list
    complex_column = df['Complex'].tolist()

    # Initialize an empty dictionary to store processed data
    processed_data = {}

    tot_no_chain_pairs = 0
    # Iterate through the list of complex entries
    for entry in complex_column:
        # each entry is in the format [protein_name]_[entity_1]:[entity_2]
        # Split the entry into parts using '_' as the separator
        parts = entry.strip().split('_')

        # Ensure that there are exactly two parts (protein_name and entities)
        if len(parts) == 2:
            protein_name, entities = parts[0], parts[1]

            # Split the entities into entity_1 and entity_2
            entity_1, entity_2 = entities.split(':')

            # Split entity_1 and entity_2 into lists of characters
            entity_1_chain_nm_lst = list(entity_1)
            entity_2_chain_nm_lst = list(entity_2)

            # Combine both lists into a single list of characters (chain names list)
            chain_names_list = entity_1_chain_nm_lst + entity_2_chain_nm_lst

            # Create a list of tuples representing chain pairs
            chain_pair_names_list = [(a, b) for a in entity_1_chain_nm_lst for b in entity_2_chain_nm_lst]
            # ## chain_pair_names_list = [(a, b) for a in chain_names_list for b in chain_names_list if a != b]

            # Create a sub-dictionary with processed data
            sub_dict = {
                'entity_1_chain_nm_lst': entity_1_chain_nm_lst,
                'entity_2_chain_nm_lst': entity_2_chain_nm_lst,
                'chain_names_list': chain_names_list,
                'chain_pair_names_list': chain_pair_names_list
            }
            print(f"protein_name: {protein_name}  :: sub_dict: {sub_dict}")
            tot_no_chain_pairs += len(chain_pair_names_list)

            # Add an entry to the main dictionary with protein_name as the key
            # and the sub-dictionary as the value
            processed_data[protein_name] = sub_dict
    # end of for loop: for entry in complex_column:
    print(f"tot_no_chain_pairs: {tot_no_chain_pairs}")
    # Save the output dictionary as a pkl file
    PPIPUtils.createFolder(output_location)
    output_pkl_file = os.path.join(output_location, f"dock_bm_parsed_dict.pkl")
    joblib.dump(processed_data, output_pkl_file)
    print('\n#### inside the process_dock_bm_csv_file() method - End\n')


def fetch_pdb_data_for_chain_seq(root_path='./', dock_benchmark_ver = 'BM_5_5'):
    """
    Downloads a PDB file for a given protein name, parses it to extract chain sequences,
    and saves the data as a pickle (pkl) file.

    Args:
        root_path (str): root_path

    Returns:
        None
    """
    print('\n#### inside the fetch_pdb_data_for_chain_seq() method - Start')

    # The directory to save the downloaded PDB file.
    pdb_file_location = os.path.join(root_path, f"dataset/preproc_data_docking_{dock_benchmark_ver}/pdb_files")
    PPIPUtils.createFolder(pdb_file_location)
    # The directory to save the output pickle (pkl) file.
    output_location = os.path.join(root_path, f"dataset/preproc_data_docking_{dock_benchmark_ver}/prot_chain_seq")
    PPIPUtils.createFolder(output_location)

    # load the dock_bm_parsed_dict.pkl file saved in process_dock_bm_csv_file() method
    dock_bm_parsed_pkl_loc = os.path.join(root_path, f"dataset/preproc_data_docking_{dock_benchmark_ver}", f"dock_bm_parsed_dict.pkl")
    dock_bm_parsed_dict = joblib.load(dock_bm_parsed_pkl_loc)

    # iterate through the dock_bm_parsed_dict
    key_lst_in_dock_bm_parsed_dict = list(dock_bm_parsed_dict.keys())
    no_of_keys_in_dock_bm_parsed_dict = len(key_lst_in_dock_bm_parsed_dict)
    for itr in range(no_of_keys_in_dock_bm_parsed_dict):
    # for itr in range(210, no_of_keys_in_dock_bm_parsed_dict):  # ###################### FOR DEBUGGING
        print(f"\n################# starting {itr}-th iteration out of {no_of_keys_in_dock_bm_parsed_dict-1} \n")
        protein_name = key_lst_in_dock_bm_parsed_dict[itr]
        sub_dict = dock_bm_parsed_dict[protein_name]
        # retrieve the chain_names_list from sub_dict
        chain_names_list = sub_dict['chain_names_list']

        print(f"Fetching PDB file for protein: {protein_name}")
        # Generate the PDB URL based on the protein name
        pdb_url = f"https://files.rcsb.org/download/{protein_name}.pdb"

        # Define the path for saving the PDB file
        pdb_file_path = os.path.join(pdb_file_location, f"{protein_name}.pdb")

        # Download the PDB file from the URL
        response = requests.get(pdb_url)
        response.raise_for_status()

        # Save the PDB file to the specified location
        with open(pdb_file_path, 'wb') as pdb_file:
            pdb_file.write(response.content)

        print(f"PDB file saved as {pdb_file_path}")

        # Create a PDBParser object
        parser = PDB.PDBParser(QUIET=False)

        # Parse the downloaded PDB file
        structure = parser.get_structure(protein_name, pdb_file_path)

        # Initialize an empty dictionary to store chain names and sequences
        chain_sequences = {}

        # extract the model from the structure (usually there's only one model)
        model = structure[0]

        # Iterate through the chain_names_list and retrieve each individual chain sequence from the model
        for chain_name in chain_names_list:
            print(f"chain name: {chain_name}")
            # Retrieve the specific chain for the given chain_name
            chain = model[chain_name]

            # Initialize an empty string to store the sequence
            sequence = ''

            # Iterate through the residues in the chain
            for residue in chain:
                # filter out non-standard amino acids
                if PDB.is_aa(residue, standard=True):
                    # Append the one-letter code to the sequence
                    sequence += PDB.Polypeptide.index_to_one(PDB.Polypeptide.three_to_index(residue.get_resname()))
                # end of if block: 
            # end of for loop: for residue in chain:

            # Add an entry to the output dictionary with chain name as the key and sequence as the value
            chain_sequences[chain_name] = sequence
        # end of for loop: for chain_name in chain_names_list:
        print(f"Extracted chain sequences: {chain_sequences}")

        # Save the output dictionary as a pkl file
        output_pkl_file = os.path.join(output_location, f"chain_seq_{protein_name}.pkl")
        joblib.dump(chain_sequences, output_pkl_file)
        print(f"Saved chain sequences as {output_pkl_file}")
    # end of for loop: for itr in range(no_of_keys_in_dock_bm_parsed_dict):
    print('\n#### inside the fetch_pdb_data_for_chain_seq() method - End\n')


def create_seq_and_test_csv(root_path='./', dock_benchmark_ver = 'BM_5_5'):
    print('\n#### inside the create_seq_and_test_csv() method - Start')
    prot_chain_seq_pkl_folder = os.path.join(root_path, f"dataset/preproc_data_docking_{dock_benchmark_ver}/prot_chain_seq")
    # load the dock_bm_parsed_dict.pkl file saved in process_dock_bm_csv_file() method
    print('loading the dock_bm_parsed_dict.pkl file saved in process_dock_bm_csv_file() method')
    dock_bm_parsed_pkl_loc = os.path.join(root_path, f"dataset/preproc_data_docking_{dock_benchmark_ver}", f"dock_bm_parsed_dict.pkl")
    dock_bm_parsed_dict = joblib.load(dock_bm_parsed_pkl_loc)

    # iterate through the dock_bm_parsed_dict
    print('iterating through the dock_bm_parsed_dict')
    # initialize a few lists to be populated during the iteration through dock_bm_parsed_dict
    prot_id_lst, seq_lst = [], []  # required to create sequence CSV file
    prot_1_id_lst, prot_2_id_lst = [], []  # required to create test CSV file

    key_lst_in_dock_bm_parsed_dict = list(dock_bm_parsed_dict.keys())
    no_of_keys_in_dock_bm_parsed_dict = len(key_lst_in_dock_bm_parsed_dict)
    for itr in range(no_of_keys_in_dock_bm_parsed_dict):
    # for itr in range(21, no_of_keys_in_dock_bm_parsed_dict):  # ################### FOR DEBUGGING
        print(f"\n################# starting {itr}-th iteration out of {no_of_keys_in_dock_bm_parsed_dict-1} \n")
        protein_name = key_lst_in_dock_bm_parsed_dict[itr]
        sub_dict = dock_bm_parsed_dict[protein_name]
        print(f"protein_name: {protein_name}")

        # load chain_seq_{protein_name}.pkl file correponding to protein_name
        print(f"loading chain_seq_{protein_name}.pkl file correponding to protein_name")
        prot_chain_seq_pkl_location = os.path.join(prot_chain_seq_pkl_folder, f"chain_seq_{protein_name}.pkl")
        chain_sequences_dict = joblib.load(prot_chain_seq_pkl_location)

        # retrieve the chain_names_list from sub_dict
        chain_names_list = sub_dict['chain_names_list']
        # iterate through the chain_names_list
        print('iterating through the chain_names_list')
        for chain_name in chain_names_list:
            print(f"chain name: {chain_name}")
            # create protein id as [protein_name]_[chain_name] 
            prot_id = f"{protein_name}_{chain_name}"
            print(f"prot_id: {prot_id}")
            # retrieve the sequence correponding to the chain_name from chain_sequences_dict
            chain_seq = chain_sequences_dict[chain_name]
            # populate prot_id_lst, seq_lst
            prot_id_lst.append(prot_id)
            seq_lst.append(chain_seq)
        #end of for loop: for chain_name in chain_names_list:

        # retrieve the chain_pair_names_list from sub_dict
        chain_pair_names_list = sub_dict['chain_pair_names_list']
        # iterate through the chain_pair_names_list
        print('iterating through the chain_pair_names_list')
        for chain_pair_names in chain_pair_names_list:
            print(f"chain_pair_names: {chain_pair_names}")
            chain_1_nm, chain_2_nm = chain_pair_names
            # create protein id as [protein_name]_[chain_name]
            prot_1_id = f"{protein_name}_{chain_1_nm}"
            prot_2_id = f"{protein_name}_{chain_2_nm}"
            # populate prot_1_id_lst, prot_2_id_lst
            prot_1_id_lst.append(prot_1_id)
            prot_2_id_lst.append(prot_2_id)
        # end of for loop: for chain_pair_names in chain_pair_names_list:
    # end of for loop: for itr in range(no_of_keys_in_dock_bm_parsed_dict):

    # create and save dock_seq_df
    print('\n creating and saving dock_seq_df')
    dock_seq_df = pd.DataFrame(data = {'prot_id': prot_id_lst, 'seq': seq_lst})
    dock_seq_df.to_csv(os.path.join(root_path, f'dataset/preproc_data_docking_{dock_benchmark_ver}', 'dock_seq.csv'), index=False)
    
    # create and save dock_test_df
    print('\n creating and saving dock_test_df')
    # populate test labels
    test_labels_lst = [1.0] * len(prot_1_id_lst)
    # Create and save dock_test_df
    dock_test_df = pd.DataFrame(data = {'prot_1_id': prot_1_id_lst, 'prot_2_id': prot_2_id_lst, 'test_labels': test_labels_lst})
    dock_test_tsv_location = os.path.join(root_path, f'dataset/preproc_data_docking_{dock_benchmark_ver}/derived_feat/dock')
    PPIPUtils.createFolder(dock_test_tsv_location)
    dock_test_df.to_csv(os.path.join(dock_test_tsv_location, 'dock_test.tsv'), index=False, sep='\t', header=False)
    print('\n#### inside the create_seq_and_test_csv() method - End')


def create_len_limited_seq_and_test_csv(root_path='./', dock_benchmark_ver = 'BM_5_5', limiting_len=400):
    print('\n#### inside the create_len_limited_seq_and_test_csv() method - Start')
    print("### creating modified_dock_seq_df -Start")
    # read the dock_seq.csv file in a pandas dataframe
    dock_seq_fl_nm_loc = os.path.join(root_path, f'dataset/preproc_data_docking_{dock_benchmark_ver}', 'dock_seq.csv')
    dock_seq_df = pd.read_csv(dock_seq_fl_nm_loc)
    mod_prot_id_lst, mod_seq_lst = [], []
    # iterate through dock_seq_df and based on limiting_len, populate mod_prot_id_lst and mod_seq_lst
    for index, row in dock_seq_df.iterrows():
        if(len(row['seq']) <= limiting_len):
           mod_prot_id_lst.append(row['prot_id'])
           mod_seq_lst.append(row['seq'])
    # end of for loop: for index, row in dock_seq_df.iterrows():
    # create and save modified_dock_seq_df
    modified_dock_seq_df = pd.DataFrame(data={'prot_id': mod_prot_id_lst, 'seq': mod_seq_lst})
    modified_dock_seq_df.to_csv(os.path.join(root_path, f'dataset/preproc_data_docking_{dock_benchmark_ver}', f'dock_seq_lenLimit_{limiting_len}.csv'), index=False)
    print("### creating modified_dock_seq_df -End")

    print("### creating modified_dock_test_df -Start")
    # read the dock_test.tsv file in a pandas dataframe
    doc_test_tsv_fl_nm_loc = os.path.join(root_path, f'dataset/preproc_data_docking_{dock_benchmark_ver}/derived_feat/dock', 'dock_test.tsv')
    dock_test_df = pd.read_csv(doc_test_tsv_fl_nm_loc, sep='\t', header=None, names=['prot_1_id', 'prot_2_id', 'test_labels'])
    mod_prot_1_id_lst, mod_prot_2_id_lst, mod_test_labels_lst = [], [], []
    # iterate over the dock_test_df and for each pair of participating proteins in each row, check whether both the proteins 
    # are in mod_prot_id_lst and if they are, then populate mod_prot_1_id_lst, mod_prot_2_id_lst, mod_test_labels_lst.
    for index, row in dock_test_df.iterrows():
        if(row['prot_1_id'] in mod_prot_id_lst and row['prot_2_id'] in mod_prot_id_lst):
            mod_prot_1_id_lst.append(row['prot_1_id'])
            mod_prot_2_id_lst.append(row['prot_2_id'])
            mod_test_labels_lst.append(row['test_labels'])
    # end of for loop: for index, row in dock_test_df.iterrows():
    # create and save modified_dock_test_df
    modified_dock_test_df = pd.DataFrame(data = {'prot_1_id': mod_prot_1_id_lst, 'prot_2_id': mod_prot_2_id_lst, 'test_labels': mod_test_labels_lst})
    modified_dock_test_df.to_csv(os.path.join(root_path, f'dataset/preproc_data_docking_{dock_benchmark_ver}/derived_feat/dock', f'dock_test_lenLimit_{limiting_len}.tsv'), index=False, sep='\t', header=False)
    print(f'total number of PPI entries: {len(mod_prot_1_id_lst)}')
    print("### creating modified_dock_test_df -End")
    print('\n#### inside the create_len_limited_seq_and_test_csv() method - End')


if __name__ == "__main__":
    root_path = os.path.join('/project/root/directory/path/here')
    root_path = os.path.join('/scratch/pralaycs/Shubh_Working_Remote/PPI_Wkspc/PPI_Code/mat_p2ip_prj_working')
    # root_path = os.path.join('/home/Shubh_Working_Ubuntu/Workspaces/PPI_Wkspc/PPI_Code/mat_p2ip_prj_working')

    dock_benchmark_ver = 'BM_4_0'  # BM_5_5, BM_4_0
    dock_bm_csv_file_location = os.path.join(root_path, f"dataset/orig_data_docking/Table_{dock_benchmark_ver}.csv")
    output_location = os.path.join(root_path, f"dataset/preproc_data_docking_{dock_benchmark_ver}")
    process_dock_bm_csv_file(dock_bm_csv_file_location, output_location)
    fetch_pdb_data_for_chain_seq(root_path=root_path, dock_benchmark_ver=dock_benchmark_ver)
    create_seq_and_test_csv(root_path=root_path, dock_benchmark_ver=dock_benchmark_ver)

    limiting_len = 400
    create_len_limited_seq_and_test_csv(root_path=root_path, dock_benchmark_ver=dock_benchmark_ver, limiting_len=limiting_len)
