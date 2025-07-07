import os
import sys
from pathlib import Path
path_root = Path(__file__).parents[2]  
sys.path.insert(0, str(path_root))
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
        contact_map_location = os.path.join(output_location, f"contact_map_{protein_name}_{chain_1_name}_{chain_2_name}.pkl")
        if(os.path.exists(contact_map_location)):
            return f"contact_map_{protein_name}_{chain_1_name}_{chain_2_name}"
        pdb_file_path = os.path.join(pdb_file_location, f"{protein_name}.pdb")
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure(protein_name, pdb_file_path)
        model = structure[0]
        chain1 = model[chain_1_name]
        chain2 = model[chain_2_name]
        residues_chain1 = [res for res in chain1.get_residues() if is_aa(res, standard=True)]
        residues_chain2 = [res for res in chain2.get_residues() if is_aa(res, standard=True)]
        atoms_chain1 = Selection.unfold_entities(residues_chain1, "A")  
        atoms_chain2 = Selection.unfold_entities(residues_chain2, "A")  
        all_atoms = atoms_chain1 + atoms_chain2
        neighbor_search = NeighborSearch(all_atoms)
        contact_map = np.zeros((len(residues_chain1), len(residues_chain2)), dtype=int)
        for i, res1 in enumerate(residues_chain1):
            for j, res2 in enumerate(residues_chain2):
                atoms1 = Selection.unfold_entities([res1], "A")
                atoms2 = Selection.unfold_entities([res2], "A")
                close_contacts = neighbor_search.search_all(distance_cutoff, level="A")
                if any((a1 in atoms1 and a2 in atoms2) or (a1 in atoms2 and a2 in atoms1) for a1, a2 in close_contacts):
                    contact_map[i, j] = 1
        joblib.dump(contact_map, contact_map_location)
        return f"contact_map_{protein_name}_{chain_1_name}_{chain_2_name}"
    except Exception as e:
        return None  


def calculate_contact_maps_parallel(root_path='./', docking_version = '5_5', distance_cutoff = 5.0, max_process = 5):
    pdb_file_location = os.path.join(root_path, f'dataset/preproc_data_docking_BM_{docking_version}/pdb_files')
    output_location = os.path.join(root_path, f'dataset/preproc_data_docking_BM_{docking_version}/prot_gt_contact_map')
    PPIPUtils.createFolder(output_location)
    doc_test_tsv_fl_nm_loc = os.path.join(root_path, f'dataset/preproc_data_docking_BM_{docking_version}/derived_feat/dock', 'dock_test_lenLimit_400.tsv')
    dock_test_df = pd.read_csv(doc_test_tsv_fl_nm_loc, sep='\t', header=None, names=['prot_1_id', 'prot_2_id', 'test_labels'])
    total_entries_dock_test_df = dock_test_df.shape[0]  
    print(f"[INFO] Starting processing of {total_entries_dock_test_df} entries with {max_process} workers...")
    processed_count = 0
    with ProcessPoolExecutor(max_workers=max_process) as executor:
        args_dict_list = []  
        for index, row in dock_test_df.iterrows():  
            prot_1_id, prot_2_id = row['prot_1_id'], row['prot_2_id']
            protein_name, chain_1_name = prot_1_id.split('_')
            protein_name, chain_2_name = prot_2_id.split('_')
            indiv_arg_dict = {'root_path': root_path, 'distance_cutoff': distance_cutoff
                              , 'pdb_file_location': pdb_file_location, 'output_location': output_location
                              , 'protein_name': protein_name, 'chain_1_name': chain_1_name, 'chain_2_name': chain_2_name}
            args_dict_list.append(indiv_arg_dict)
        futures = {executor.submit(calculate_and_save_contact_map, **indiv_arg_dict): f"{indiv_arg_dict['protein_name']}_{indiv_arg_dict['chain_1_name']}_{indiv_arg_dict['chain_2_name']}" for indiv_arg_dict in args_dict_list}
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


if __name__ == "__main__":
    root_path = os.path.join('/project/root/directory/path/here')
    docking_version = '5_5'  
    distance_cutoff = 8.0
    max_process = 40 
    calculate_contact_maps_parallel(root_path = root_path, docking_version = docking_version, distance_cutoff = distance_cutoff, max_process = max_process)
