import os, sys
from pathlib import Path
path_root = Path(__file__).parents[1]  
sys.path.insert(0, str(path_root))
import numpy as np
import cv2  
from sklearn.cluster import DBSCAN
from Bio import PDB

residue_to_index = {
    'A': 0, 'R': 1, 'N': 2, 'D': 3, 'C': 4,
    'Q': 5, 'E': 6, 'G': 7, 'H': 8, 'I': 9,
    'L': 10, 'K': 11, 'M': 12, 'F': 13, 'P': 14,
    'S': 15, 'T': 16, 'W': 17, 'Y': 18, 'V': 19
}


lower_triangle = [
    [-0.08],
    [0.18, 0.03],
    [0.07, -0.08, -0.26],
    [0.10, -0.51, -0.27, -0.09],
    [0.01, 0.32, 0.21, 0.25, -0.55],
    [0.09, -0.14, -0.19, -0.10, 0.10, -0.10],
    [0.19, -0.51, -0.19, 0.04, 0.31, -0.09, 0.04],
    [-0.04, 0.03, -0.09, -0.06, 0.03, 0.00, 0.13, -0.25],
    [0.11, -0.01, -0.08, -0.27, -0.02, 0.02, -0.19, 0.04, -0.36],
    [-0.07, 0.20, 0.32, 0.31, -0.04, 0.12, 0.24, 0.17, 0.17, -0.18],
    [-0.04, 0.20, 0.26, 0.40, -0.01, 0.13, 0.28, 0.18, 0.16, -0.21, -0.20],
    [0.13, 0.28, -0.18, -0.50, 0.35, -0.17, -0.57, 0.01, 0.10, 0.22, 0.24, 0.16],
    [0.00, 0.20, 0.19, 0.36, -0.04, 0.04, 0.16, 0.10, -0.02, -0.13, -0.13, 0.24, -0.20],
    [-0.01, 0.20, 0.22, 0.32, -0.02, 0.07, 0.26, 0.16, 0.01, -0.12, -0.15, 0.22, -0.25, -0.22],
    [0.06, -0.02, -0.03, 0.03, 0.03, -0.09, 0.05, -0.07, -0.08, 0.12, 0.09, 0.06, 0.01, 0.03, -0.11],
    [0.01, -0.03, -0.12, -0.20, 0.08, -0.04, -0.14, -0.10, -0.05, 0.20, 0.19, -0.05, 0.19, 0.10, -0.02, -0.17],
    [0.01, -0.02, -0.11, -0.13, 0.12, -0.08, -0.10, -0.07, -0.06, 0.08, 0.12, -0.02, 0.09, 0.13, -0.03, -0.12, -0.07],
    [0.02, -0.02, 0.07, 0.15, -0.07, 0.10, 0.06, 0.04, -0.08, -0.05, -0.03, 0.06, -0.21, -0.08, -0.21, 0.15, 0.20, -0.10],
    [0.05, -0.10, 0.02, -0.03, 0.16, -0.06, -0.06, 0.03, -0.05, 0.02, 0.00, -0.12, -0.08, -0.02, -0.13, 0.04, 0.09, 0.01, 0.01],
    [-0.07, 0.23, 0.25, 0.40, -0.04, 0.16, 0.28, 0.10, 0.19, -0.16, -0.19, 0.22, -0.03, -0.11, 0.07, 0.16, 0.11, -0.01, 0.08, -0.19]
]


class ProteinContactMapProcessor:
    def __init__(self, pdb_file_location=None, protein_name=None, chain_1_name=None, chain_2_name=None, pred_contact_map=None, seq_order='SCGB', lambda_weight=0.5):
        self.aaindex_potential = self.compute_aaindex_potential(pdb_file_location, protein_name, chain_1_name, chain_2_name)
        self.pred_contact_map = pred_contact_map
        self.seq_order_lst = list(seq_order)
        self.lambda_weight = lambda_weight
    
    
    def apply_statistical_potential(self):
        self.pred_contact_map = (self.lambda_weight * self.aaindex_potential) + ((1 - self.lambda_weight) * self.pred_contact_map)
        self.pred_contact_map = (self.pred_contact_map - np.min(self.pred_contact_map)) / (np.max(self.pred_contact_map) - np.min(self.pred_contact_map))  
        return self.pred_contact_map
    
    
    def apply_clustering(self):
        coords = np.argwhere(self.pred_contact_map > 0.3)  
        if len(coords) == 0:
            return self.pred_contact_map  
        clustering = DBSCAN(eps=3, min_samples=2, algorithm='auto').fit(coords)
        cluster_labels = clustering.labels_
        clustered_map = np.zeros_like(self.pred_contact_map)
        for idx, label in enumerate(cluster_labels):
            if label != -1:
                clustered_map[tuple(coords[idx])] = self.pred_contact_map[tuple(coords[idx])]
        self.pred_contact_map = clustered_map
        return self.pred_contact_map
    
    
    def apply_graph_smoothing(self):
        m, n = self.pred_contact_map.shape
        adjacency_matrix_A = np.copy(self.pred_contact_map)  
        row_degrees = np.sum(adjacency_matrix_A, axis=1)  
        col_degrees = np.sum(adjacency_matrix_A, axis=0)  
        D_r = np.diag(row_degrees)  
        D_c = np.diag(col_degrees)  
        D_r_inv_sqrt = np.linalg.pinv(np.sqrt(D_r + np.eye(m) * 1e-6))  
        D_c_inv_sqrt = np.linalg.pinv(np.sqrt(D_c + np.eye(n) * 1e-6))  
        laplacian_smoothed = D_r_inv_sqrt @ adjacency_matrix_A @ D_c_inv_sqrt
        self.pred_contact_map = laplacian_smoothed
        self.pred_contact_map = (self.pred_contact_map - np.min(self.pred_contact_map)) / (np.max(self.pred_contact_map) - np.min(self.pred_contact_map))
        return self.pred_contact_map
    
    
    def apply_adaptive_binarization(self):
        threshold_value = 0.95
        self.pred_contact_map = np.where(self.pred_contact_map >= threshold_value, self.pred_contact_map, 0)
        _, binarized_map = cv2.threshold((self.pred_contact_map * 255).astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        self.pred_contact_map = binarized_map.astype(int) / 255 
        return self.pred_contact_map
    
    
    def compute_aaindex_potential(self, pdb_file_location, protein_name, chain_1_name, chain_2_name):
        stat_potential_mat = np.zeros((20, 20))
        for i in range(20):
            for j in range(i + 1):
                stat_potential_mat[i, j] = lower_triangle[i][j]  
                stat_potential_mat[j, i] = lower_triangle[i][j]  
        stat_potential_mat = -stat_potential_mat  
        stat_potential_mat = (stat_potential_mat - np.min(stat_potential_mat)) / (np.max(stat_potential_mat) - np.min(stat_potential_mat))  
        pdb_file_path = os.path.join(pdb_file_location, f"{protein_name}.pdb")
        parser = PDB.PDBParser(QUIET=True)
        structure = parser.get_structure(protein_name, pdb_file_path)
        model = structure[0]
        chain1 = model[chain_1_name]
        chain2 = model[chain_2_name]
        chain_1_lst, chain_2_lst = [], []
        for residue in chain1:
            three_letter_res_name = residue.get_resname()
            is_standard_aa = PDB.Polypeptide.is_aa(three_letter_res_name, standard = True)
            if(is_standard_aa):
                chain_1_lst.append(PDB.Polypeptide.index_to_one(PDB.Polypeptide.three_to_index(three_letter_res_name)))
        for residue in chain2:
            three_letter_res_name = residue.get_resname()
            is_standard_aa = PDB.Polypeptide.is_aa(three_letter_res_name, standard = True)
            if(is_standard_aa):
                chain_2_lst.append(PDB.Polypeptide.index_to_one(PDB.Polypeptide.three_to_index(three_letter_res_name)))
        len1, len2 = len(chain_1_lst), len(chain_2_lst)
        aaindex_potential = np.zeros((len1, len2))
        for i in range(len1):
            for j in range(len2):
                res1, res2 = chain_1_lst[i], chain_2_lst[j]
                if res1 in residue_to_index and res2 in residue_to_index:
                    idx1 = residue_to_index[res1]
                    idx2 = residue_to_index[res2]
                    aaindex_potential[i, j] = stat_potential_mat[idx1, idx2]
                else:
                    aaindex_potential[i, j] = 0.0  
        return aaindex_potential
    
    
    def process(self):
        steps = {
            'S': self.apply_statistical_potential,
            'C': self.apply_clustering,
            'G': self.apply_graph_smoothing,
            'B': self.apply_adaptive_binarization,
        }
        for step in self.seq_order_lst:
            if step in steps:
                steps[step]()
        return self.pred_contact_map
