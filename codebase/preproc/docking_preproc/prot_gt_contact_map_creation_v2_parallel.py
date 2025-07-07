import os
import sys
from pathlib import Path

path_root = Path(__file__).parents[2]  # upto 'codebase' folder
sys.path.insert(0, str(path_root))
# print(sys.path)

from concurrent.futures import ProcessPoolExecutor
import joblib
import numpy as np
import pandas as pd
from Bio.PDB import PDBParser, NeighborSearch, Selection
from Bio.PDB.Polypeptide import is_aa

from utils import PPIPUtils


def calculate_and_save_contact_map(**kwargs):
    """
    Calculate the interaction map between two chains in a protein complex, considering only standard amino acids.

    Returns:
        np.ndarray: A binary interaction map (2D numpy array).
    """
    root_path = kwargs['root_path']; distance_cutoff = kwargs['distance_cutoff'];  
    pdb_file_location = kwargs['pdb_file_location']; output_location = kwargs['output_location']
    protein_name = kwargs['protein_name']; chain_1_name = kwargs['chain_1_name']; chain_2_name = kwargs['chain_2_name'];  

    try:
        # Define interaction map location
        contact_map_location = os.path.join(output_location, f"contact_map_{protein_name}_{chain_1_name}_{chain_2_name}.pkl")

        # Check whether the interaction map already exists at contact_map_location
        if(os.path.exists(contact_map_location)):
            # if interaction map already exists at contact_map_location, then return immediately
            print(f'\n[INFO] #### contact_map_{protein_name}_{chain_1_name}_{chain_2_name}.pkl already exists at {output_location}\n')
            return f"contact_map_{protein_name}_{chain_1_name}_{chain_2_name}"

        # Read the PDB file corresponding to the protein_name
        pdb_file_path = os.path.join(pdb_file_location, f"{protein_name}.pdb")
        # create a PDBParser object
        parser = PDBParser(QUIET=True)
        # parse already saved PDB file
        structure = parser.get_structure(protein_name, pdb_file_path)
        # extract the model from the structure (usually there's only one model)
        model = structure[0]
        # Extract the specified chains
        chain1 = model[chain_1_name]
        chain2 = model[chain_2_name]
        
        # Filter residues to include only standard amino acids
        residues_chain1 = [res for res in chain1.get_residues() if is_aa(res, standard=True)]
        residues_chain2 = [res for res in chain2.get_residues() if is_aa(res, standard=True)]
        
        # Extract atoms from the filtered residues
        atoms_chain1 = Selection.unfold_entities(residues_chain1, "A")  # Get all atoms in chain 1
        atoms_chain2 = Selection.unfold_entities(residues_chain2, "A")  # Get all atoms in chain 2
        
        # Use BioPython's NeighborSearch to identify atom pairs within the distance threshold
        all_atoms = atoms_chain1 + atoms_chain2
        neighbor_search = NeighborSearch(all_atoms)
        
        # Initialize interaction map
        contact_map = np.zeros((len(residues_chain1), len(residues_chain2)), dtype=int)
        
        # Populate the interaction map
        for i, res1 in enumerate(residues_chain1):
            for j, res2 in enumerate(residues_chain2):
                if(j % 40 == 0): print(f'\ncontact_map_{protein_name}_{chain_1_name}_{chain_2_name}::  i = {i} / {len(residues_chain1)} and j = {j} / {len(residues_chain2)}')
                # Find if any atom in res1 is within the threshold distance to any atom in res2
                atoms1 = Selection.unfold_entities([res1], "A")
                atoms2 = Selection.unfold_entities([res2], "A")
                close_contacts = neighbor_search.search_all(distance_cutoff, level="A")
                
                # Check if any atom pair from the two residues is in the contact set
                if any((a1 in atoms1 and a2 in atoms2) or (a1 in atoms2 and a2 in atoms1) for a1, a2 in close_contacts):
                    contact_map[i, j] = 1
                # End of if block
            # End of inner for loop: for j, res2 in enumerate(residues_chain2):
        # End of outer for loop: for i, res1 in enumerate(residues_chain1):

        # Save the interaction map
        joblib.dump(contact_map, contact_map_location)
        print(f"[INFO] interaction map saved: contact_map_{protein_name}_{chain_1_name}_{chain_2_name}.pkl")
        return f"contact_map_{protein_name}_{chain_1_name}_{chain_2_name}"
    except Exception as e:
        print(f"[ERROR] Failed to process {protein_name}_{chain_1_name}_{chain_2_name}:: {e}")
        return None  # Return None for failed processing
# End of calculate_and_save_contact_map() method


def calculate_contact_maps_parallel(root_path='./', docking_version = '4_0', distance_cutoff = 5.0, max_process = 5):
    print('\n#### inside the calculate_contact_maps_parallel() method - Start\n')
    # The directory for the saved PDB files
    pdb_file_location = os.path.join(root_path, f'dataset/preproc_data_docking_BM_{docking_version}/pdb_files')
    # The directory to save the output interaction maps as pickle (pkl) file.
    output_location = os.path.join(root_path, f'dataset/preproc_data_docking_BM_{docking_version}/prot_gt_contact_map')
    PPIPUtils.createFolder(output_location)

    # read the dock_test.tsv file in a pandas dataframe
    doc_test_tsv_fl_nm_loc = os.path.join(root_path, f'dataset/preproc_data_docking_BM_{docking_version}/derived_feat/dock', 'dock_test_lenLimit_400.tsv')
    dock_test_df = pd.read_csv(doc_test_tsv_fl_nm_loc, sep='\t', header=None, names=['prot_1_id', 'prot_2_id', 'test_labels'])
    
    total_entries_dock_test_df = dock_test_df.shape[0]  # number of rows
    print(f"[INFO] Starting processing of {total_entries_dock_test_df} entries with {max_process} workers...")
    processed_count = 0

    with ProcessPoolExecutor(max_workers=max_process) as executor:
        # Prepare arguments for parallel processing by iterating over the dock_test_df 
        args_dict_list = []  # list of dictionaries

        for index, row in dock_test_df.iterrows():  # iterate full dock_test_df
        # for index, row in dock_test_df.iloc[: total_entries_dock_test_df // 2].iterrows():  # iterate first half of dock_test_df 
        # for index, row in dock_test_df.iloc[total_entries_dock_test_df // 2 :].iterrows():  # iterate the last half of dock_test_df
            prot_1_id, prot_2_id = row['prot_1_id'], row['prot_2_id']
            # protein id has the format of [protein_name]_[chain_name]
            protein_name, chain_1_name = prot_1_id.split('_')
            protein_name, chain_2_name = prot_2_id.split('_')
            # print(f"protein_name: {protein_name} :: chain_1_name: {chain_1_name} :: chain_2_name: {chain_2_name}")
            indiv_arg_dict = {'root_path': root_path, 'distance_cutoff': distance_cutoff
                              , 'pdb_file_location': pdb_file_location, 'output_location': output_location
                              , 'protein_name': protein_name, 'chain_1_name': chain_1_name, 'chain_2_name': chain_2_name}
            args_dict_list.append(indiv_arg_dict)
        # End of for loop: for index, row in dock_test_df.iterrows():
        
        # Submit tasks to the thread pool
        # 'futures' is a dictionary, where the keys are the Future objects (objects returned by executor.submit()), 
        # and the values are the corresponding {protein_name}_{chain_1_name}_{chain_2_name}.
        futures = {executor.submit(calculate_and_save_contact_map, **indiv_arg_dict): f"{indiv_arg_dict['protein_name']}_{indiv_arg_dict['chain_1_name']}_{indiv_arg_dict['chain_2_name']}" for indiv_arg_dict in args_dict_list}
        
        # Collect results as they complete
        for future in futures:
            try:
                res = future.result()
                if(res is not None):
                    processed_count += 1
                    print(f"[INFO] Processed {processed_count}/{total_entries_dock_test_df}: {res}")
                else:
                    print(f"[WARNING] Skipped due to errors: {protein_name}_{chain_1_name}_{chain_2_name}")
            except Exception as e:
                print(f"[ERROR] Failed to process a protein: {e}")
        # End of for loop: for future in futures:
    # End of with block: with ProcessPoolExecutor(max_workers=max_process) as executor:
    print(f"[INFO] Completed processing {processed_count}/{total_entries_dock_test_df} entries.")
    print('\n#### inside the calculate_contact_maps_parallel() method - End\n')
# End of calculate_contact_maps_parallel() method



if __name__ == "__main__":
    root_path = os.path.join('/project/root/directory/path/here')
    
    # root_path = os.path.join('/home/Shubh_Working_Ubuntu/Workspaces/PPI_Wkspc/PPI_Code/mat_p2ip_prj_working')

    docking_version = '4_0'  # '4_0', '5_5'
    distance_cutoff = 5.0  # Distance threshold in Å
    max_process = 40 # Number of parallel workers
    calculate_contact_maps_parallel(root_path = root_path, docking_version = docking_version, distance_cutoff = distance_cutoff, max_process = max_process)
