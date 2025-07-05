import os, sys
from pathlib import Path

path_root = Path(__file__).parents[1]  # upto 'codebase' folder
sys.path.insert(0, str(path_root))

import numpy as np
import cv2  # For Otsu's thresholding
from sklearn.cluster import DBSCAN
import hdbscan
from Bio import PDB


# Mapping of amino acid residues to indices (example for standard 20 amino acids)
residue_to_index = {
    'A': 0, 'R': 1, 'N': 2, 'D': 3, 'C': 4,
    'Q': 5, 'E': 6, 'G': 7, 'H': 8, 'I': 9,
    'L': 10, 'K': 11, 'M': 12, 'F': 13, 'P': 14,
    'S': 15, 'T': 16, 'W': 17, 'Y': 18, 'V': 19
}

# AAindex3 dataset: Miyazawa and Jernigan: MIYS990107
# Define the lower triangular part of the MJ-matrix
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
        """ Incorporate AAindex3 statistical potentials to refine pred_contact_map. 
            This reduces biologically implausible contacts early, making later steps more effective.
            We can refine pred_contact_map by boosting contacts that are more biophysically plausible based on these potential values.
            If a residue pair has a low statistical potential score (indicating unfavorable interaction), we can suppress it in pred_contact_map.
            Conversely, pairs with high statistical potential can be given a confidence boost.
        """
        # Fuse MJ potentials with pred_contact_map using a weighted sum
        # final_map = λ⋅normalized(−MJ) + (1−λ)pred_contact_map  where λ is a hyperparameter controlling the weight of statistical potential incorporation.
        self.pred_contact_map = (self.lambda_weight * self.aaindex_potential) + ((1 - self.lambda_weight) * self.pred_contact_map)
        # self.pred_contact_map = np.clip(self.pred_contact_map, 0, 1)  # Ensure values remain between 0-1
        self.pred_contact_map = (self.pred_contact_map - np.min(self.pred_contact_map)) / (np.max(self.pred_contact_map) - np.min(self.pred_contact_map))  # Ensure values remain between 0-1
        print(f'Step-S => Statistical Potential is done')
        return self.pred_contact_map


    def apply_clustering(self):
        """ Use optimized DBSCAN clustering (with KD-tree) to refine pred_contact_map. 
            Since true contacts tend to form localized clusters in gt_map, we can apply clustering-based filtering.
            Detect clusters of strong contacts and remove isolated false positives.
        """
        coords = np.argwhere(self.pred_contact_map > 0.3)  # Extract contact points
        if len(coords) == 0:
            return self.pred_contact_map  # No need to cluster if no contacts detected
        
        clustering = DBSCAN(eps=3, min_samples=2, algorithm='auto').fit(coords)
        cluster_labels = clustering.labels_
        clustered_map = np.zeros_like(self.pred_contact_map)
        for idx, label in enumerate(cluster_labels):
            if label != -1:
                clustered_map[tuple(coords[idx])] = self.pred_contact_map[tuple(coords[idx])]
        
        self.pred_contact_map = clustered_map
        print(f'Step-C => Clustering is done')
        return self.pred_contact_map


    # def apply_clustering(self):
    #     """
    #     Use optimized HDBSCAN clustering to refine pred_contact_map. 
    #     Since true contacts tend to form localized clusters in gt_map, we can apply clustering-based filtering.
    #     Detect clusters of strong contacts and remove isolated false positives.

    #     Parameters:
    #     - clustering_threshold: Minimum probability threshold for a point to be considered for clustering.
    #     - min_cluster_size: Minimum size of clusters (HDBSCAN parameter).
    #     - min_samples: Minimum samples per cluster (HDBSCAN parameter).

    #     Returns:
    #     - clustered_map: 2D numpy array (same size as pred_contact_map) with clustered regions enhanced.
    #     """
    #     clustering_threshold=0.3; min_cluster_size=5; min_samples=2

    #     # Get indices where pred_contact_map values exceed the clustering threshold
    #     contact_points = np.column_stack(np.where(self.pred_contact_map >= clustering_threshold))

    #     if len(contact_points) == 0:
    #         return np.zeros_like(self.pred_contact_map)  # Return an empty map if no contacts are detected

    #     # Apply HDBSCAN clustering
    #     clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples)
    #     labels = clusterer.fit_predict(contact_points)

    #     # Create a new interaction map with clustered contacts
    #     clustered_map = np.zeros_like(self.pred_contact_map)

    #     # Assign clustered contacts back to the map (ignore noise points with label -1)
    #     for idx, label in enumerate(labels):
    #         if label != -1:  # Ignore noise points
    #             x, y = contact_points[idx]
    #             clustered_map[x, y] = self.pred_contact_map[x, y]  # Retain original probabilities for clustered points
    #     self.pred_contact_map = clustered_map
    #     print(f'Step-C => Clustering is done')
    #     return self.pred_contact_map


    def apply_graph_smoothing(self):
        """ Perform Laplacian smoothing on pred_contact_map using sparse matrix operations.
            Protein structures are inherently graph-like; thus, graph-based techniques can be useful.
            Graph Laplacian Smoothing: We can define a graph where residues are nodes, and edges represent predicted interactions. Using Laplacian smoothing, we can reduce false positives and improve clustering of true contacts.
            This step ensures that local contact neighborhoods remain consistent and connected.
            Example: If two residues interact, their nearby residues should also show interaction, so graph-based smoothing enforces this.
        """
        # ############################ Special notes on Laplacian for the rectangula graphs - Start #############
        # The normalized Laplacian for rectangular graphs is defined as L = D_r^(-1/2) * A * D_c^(-1/2) where
        # D_r is a row-degree matrix (sum across columns, shape m × m) 
        # and D_c is a column-degree matrix (sum across rows, shape n × n).
        # However, this deviates from the standard normalized Laplacian definition, which is primarily applicable 
        # to square matrices. For rectangular matrices, the concept of a normalized Laplacian isn't as straightforward.
        # The approach suggested here aims to normalize the adjacency matrix by the degrees of its rows and columns, 
        # but it doesn't incorporate the identity matrix subtraction as in the standard definition because defining 
        # an identity matrix I for a non-square matrix A is ambiguous, as I is inherently square.
        # ############################ Special notes on Laplacian for the rectangula graphs - End #############
        
        # m, n = self.pred_contact_map.shape
        # # Construct an adjacency matrix (A) for the interaction graph
        # adjacency_matrix_A = np.copy(self.pred_contact_map)  # A copy of the interaction map as a graph
        # # Compute row-wise and column-wise degree matrices
        # row_degrees = np.sum(adjacency_matrix_A, axis=1)  # Sum over columns (size m)
        # col_degrees = np.sum(adjacency_matrix_A, axis=0)  # Sum over rows (size n)
        # # Convert to diagonal matrices
        # D_r = np.diag(row_degrees)  # Shape: (m, m)
        # D_c = np.diag(col_degrees)  # Shape: (n, n)
        
        # # ###### Compute inverse square root using pseudo-inverse - Start ####
        # # Avoid division by zero by adding small epsilon (Compute inverse square root using pseudo-inverse)
        # D_r_inv_sqrt = np.linalg.pinv(np.sqrt(D_r + np.eye(m) * 1e-6))  # (m, m)
        # D_c_inv_sqrt = np.linalg.pinv(np.sqrt(D_c + np.eye(n) * 1e-6))  # (n, n)
        # # ###### Compute inverse square root using pseudo-inverse - End ####

        # # # ###### Compute inverse square root directly instead of pseudo-inverse (alternative to pseudo-inverse)- Start ####
        # # # Prevent division by zero using a small epsilon
        # # row_degrees_safe = row_degrees + 1e-6
        # # col_degrees_safe = col_degrees + 1e-6
        # # # Compute inverse square root directly instead of pseudo-inverse
        # # D_r_inv_sqrt = np.diag(1.0 / np.sqrt(row_degrees_safe))  # Shape: (m, m)
        # # D_c_inv_sqrt = np.diag(1.0 / np.sqrt(col_degrees_safe))  # Shape: (n, n)
        # # # ###### Compute inverse square root directly instead of pseudo-inverse (alternative to pseudo-inverse)- End ####
        
        # # Compute generalized Laplacian smoothing: L = D_r^(-1/2) * A * D_c^(-1/2)
        # laplacian_smoothed = D_r_inv_sqrt @ adjacency_matrix_A @ D_c_inv_sqrt
        # # The Laplacian now acts as a direct transformation 
        # self.pred_contact_map = laplacian_smoothed
        # # Normalize to keep values within [0,1]
        # self.pred_contact_map = (self.pred_contact_map - np.min(self.pred_contact_map)) / (np.max(self.pred_contact_map) - np.min(self.pred_contact_map))
        print(f'Step-G => Graph smoothing is done')
        return self.pred_contact_map
    

    def apply_adaptive_binarization(self):
        """ Use Otsu's thresholding for binarization.
            Since pred_contact_map consists of values between 0 and 1, a simple way to post-process it is to apply a threshold to convert it into a binary interaction map:
            Instead of using a fixed threshold (e.g., 0.5), we could use an adaptive threshold based on the distribution of values in pred_contact_map.
            Otsu’s method: A well-known method for adaptive binarization in image processing.
        """
        # Apply Otsu's thresholding on the top X% of values only
        # Compute the threshold for the top X% values
        # ### top_percentile = 5
        # ### threshold_value = np.percentile(self.pred_contact_map, 100 - top_percentile)
        threshold_value = 0.95
        
        # Mask values below this threshold
        self.pred_contact_map = np.where(self.pred_contact_map >= threshold_value, self.pred_contact_map, 0)

        # Apply Otsu's thresholding only on the remaining high-value regions
        _, binarized_map = cv2.threshold((self.pred_contact_map * 255).astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        self.pred_contact_map = binarized_map.astype(int) / 255 # Convert back to binary (0 or 1)

        # top_percentile = 100; n_keep = 100
        
        # flat_values = self.pred_contact_map.flatten()
    
        # # Compute the threshold for the top X% values
        # top_k_threshold = np.percentile(flat_values, 100 - top_percentile)

        # # Mask out values below the top X% thresholdbinarized_map
        # mask = self.pred_contact_map >= top_k_threshold
        # filtered_values = self.pred_contact_map[mask]

        # if len(filtered_values) == 0:
        #     return np.zeros_like(self.pred_contact_map)  # Return empty map if no values qualify

        # # Convert filtered values to uint8 (scale from 0-1 to 0-255)
        # filtered_values_uint8 = (filtered_values * 255).astype(np.uint8)

        # # Apply Otsu’s thresholding
        # otsu_thresh, _ = cv2.threshold(filtered_values_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # # Convert Otsu threshold back to [0,1] range
        # otsu_thresh = otsu_thresh / 255.0
        
        # # Create binary map after Otsu’s thresholding
        # binarized_map = np.zeros_like(self.pred_contact_map)
        # binarized_map[mask] = (self.pred_contact_map[mask] >= otsu_thresh).astype(np.uint8)

        # # Keep only the top `n_keep` highest values
        # positive_indices = np.column_stack(np.where(binarized_map == 1))
        
        # if len(positive_indices) > n_keep:
        #     # Sort positive indices by the original probability values (descending order)
        #     sorted_indices = sorted(positive_indices, key=lambda idx: self.pred_contact_map[idx[0], idx[1]], reverse=True)
            
        #     # Select the top `n_keep` highest probability contacts
        #     selected_indices = sorted_indices[:n_keep]

        #     # Reset the entire binarized map
        #     binarized_map[:] = 0

        #     # Keep only the selected top `n_keep` positive contacts
        #     for x, y in selected_indices:
        #         binarized_map[x, y] = 1
        # # End of if block: if len(positive_indices) > n_keep:

        # self.pred_contact_map = binarized_map
        # Percentile thresholding - End
        print(f'Step-B => Binarization is done')
        return self.pred_contact_map
    

    def compute_aaindex_potential(self, pdb_file_location, protein_name, chain_1_name, chain_2_name):
        """
        Compute the aaindex_potential array for the given sequences.
        
        Returns:
        np.ndarray: 2D array of contact potentials.
        """
        # Initialize stat_potential_mat -Start
        stat_potential_mat = np.zeros((20, 20))
        for i in range(20):
            for j in range(i + 1):
                stat_potential_mat[i, j] = lower_triangle[i][j]  # Fill the lower triangular part
                stat_potential_mat[j, i] = lower_triangle[i][j]  # Fill the symmetric upper triangular part
        # print(stat_potential_mat.shape)
        # In the context of the Miyazawa-Jernigan(MJ) matrix (MIYS990107) from the AAindex3 dataset, 
        # lower (more negative) values indicate stronger interactions (higher likelihood of contact) between residues. 
        # This is because the MJ matrix represents inter-residue contact energies, where more negative values correspond to more stable 
        # and favorable interactions.
        # Since MJ values are negatively correlated with contact probability, invert and normalize MJ values.
        stat_potential_mat = -stat_potential_mat  # Invert
        stat_potential_mat = (stat_potential_mat - np.min(stat_potential_mat)) / (np.max(stat_potential_mat) - np.min(stat_potential_mat))  # Normalize between [0.1]
        # Initialize stat_potential_mat -End

        # Read the PDB file corresponding to the protein_name
        pdb_file_path = os.path.join(pdb_file_location, f"{protein_name}.pdb")
        # create a PDBParser object
        parser = PDB.PDBParser(QUIET=True)
        # parse already saved PDB file
        structure = parser.get_structure(protein_name, pdb_file_path)
        # extract the model from the structure (usually there's only one model)
        model = structure[0]
        # Extract the specified chains
        chain1 = model[chain_1_name]
        chain2 = model[chain_2_name]
        
        # use only standard residue
        chain_1_lst, chain_2_lst = [], []
        for residue in chain1:
            three_letter_res_name = residue.get_resname()
            is_standard_aa = PDB.Polypeptide.is_aa(three_letter_res_name, standard = True)
            # print(f"residue: {three_letter_res_name} : standard: {is_standard_aa}")
            if(is_standard_aa):
                chain_1_lst.append(PDB.Polypeptide.index_to_one(PDB.Polypeptide.three_to_index(three_letter_res_name)))
        for residue in chain2:
            three_letter_res_name = residue.get_resname()
            is_standard_aa = PDB.Polypeptide.is_aa(three_letter_res_name, standard = True)
            # print(f"residue: {three_letter_res_name} : standard: {is_standard_aa}")
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
                    # Handle non-standard amino acids if necessary
                    aaindex_potential[i, j] = 0.0  # or some default value
        # End of for loop: for i in range(len1):
        return aaindex_potential
    

    def process(self):
        """ Execute processing steps in the order specified in seq_order_lst. """
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
    # End of process() method
# End of ProteinContactMapProcessor class

# Example Usage
# pred_contact_map = np.random.rand(400, 400)  # Simulated normalized contact scores
# processor = ProteinContactMapProcessor(pdb_file_location=None, protein_name=None, chain_1_name=None, chain_2_name=None, pred_contact_map=pred_contact_map, seq_order='SCGB')
# final_map = processor.process()

