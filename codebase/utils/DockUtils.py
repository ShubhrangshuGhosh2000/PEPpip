import sys
from pathlib import Path
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.metrics import precision_recall_curve, roc_auc_score, auc

path_root = Path(__file__).parents[1]  # upto 'codebase' folder
sys.path.insert(0, str(path_root))


def calculateEMD(arr_2d_1, arr_2d_2):
    # Approach:
    # Uses the Hungarian algorithm with linear sum assignment on raw 2D matrices.
    # Computes pairwise costs between all elements.
    # EMD = Total assignment cost (sum of matched costs).
    # Strengths:
    # Preserves spatial relationships between residues in 2D interaction maps.
    # Matches biological intuition (residue positions matter).
    # Weakness:
    # Computationally expensive (O(N³) complexity for N×N matrices).

    cost_matrix = cdist(arr_2d_1, arr_2d_2)
    assignment = linear_sum_assignment(cost_matrix)
    emd = cost_matrix[assignment].sum() / cost_matrix.shape[0]  # Average cost per element
    # emd = cost_matrix[assignment].sum()  # Total Assignment Cost
    return emd


def calcMatrixAbsDiff(matrix1, matrix2):
    """
    Calculate the absolute difference between corresponding elements of two 2D numpy matrices.

    Parameters:
    - matrix1 (numpy.ndarray): First 2D matrix.
    - matrix2 (numpy.ndarray): Second 2D matrix.

    Returns:
    - float: Sum of absolute differences between corresponding elements.

    Example:
    >>> matrix1 = np.array([[0.5, 0.3], [0.2, 0.7]])
    >>> matrix2 = np.array([[0.4, 0.6], [0.1, 0.8]])
    >>> matrix_difference(matrix1, matrix2)
    0.6
    """
    # Ensure that input matrices are numpy arrays
    matrix1 = np.array(matrix1)
    matrix2 = np.array(matrix2)

    # Check if the matrices have the same shape
    if matrix1.shape != matrix2.shape:
        raise ValueError("Input matrices must have the same shape")

    # Calculate the absolute differences and sum them up
    diff_sum = np.sum(np.abs(matrix1 - matrix2))

    return diff_sum


def shuffle_predContactMap_and_calc_emd(pred_contact_map, gt_contact_map, axis=1, no_random_shuffle=500, seed=456
                                        , consider_full=False, pct=99.0):
    """
    Randomly shuffle pred_contact_map (a 2D numpy array) along a specified axis multiple times and
    for each shuffling, calculate the EMD between shuffled version and gt_contact_map

    Parameters:
    - pred_contact_map (numpy.ndarray): Input 2D array of floats.
    - gt_contact_map (numpy.ndarray): Input 2D array of floats.
    - axis (int): Axis along which shuffling will be done (0 for rows, 1 for columns).
    - no_random_shuffle (int): Number of times to shuffle the matrix.
    - seed (int): Random seed for reproducibility (default is 42).
    - consider_full (boolean): whether to consider full-version or 99 percentile version of the interaction maps

    Returns:
    - list: A list containing the EMD(s) between shuffled version and gt_contact_map for all shufflings

    Example:
    >>> arr = np.array([[1.0, 2.0, 3.0],
    ...                 [4.0, 5.0, 6.0],
    ...                 [7.0, 8.0, 9.0]])
    >>> 
    Shuffled Matrix 1:
    [[3. 1. 2.]
     [6. 4. 5.]
     [9. 7. 8.]]

    Shuffled Matrix 2:
    [[2. 1. 3.]
     [5. 4. 6.]
     [8. 7. 9.]]

    Shuffled Matrix 3:
    [[2. 3. 1.]
     [5. 6. 4.]
     [8. 9. 7.]]
    """

    print('\n #############################\n inside the shuffle_predContactMap_and_calc_emd() method - Start\n')
    if(not consider_full):
        print('considering the 99 percentile version of the interaction map')
        # Calculate the 9X-th percentile value for pred_contact_map
        percentile = np.percentile(pred_contact_map, pct)
        # Set values below 9X-th percentile to zero
        pred_contact_map[pred_contact_map <= percentile] = 0
        
        # Calculate the 9X-th percentile value for gt_contact_map
        # percentile = np.percentile(gt_contact_map, pct)
        # Set values below 9X-th percentile to zero
        # gt_contact_map[gt_contact_map <= percentile] = 0
    # set random generator to a specific seed
    rng = np.random.default_rng(seed)
    emd_for_shuffled_lst = []

    for i in range(1, no_random_shuffle + 1):
        # Use numpy.random.Generator.permuted for shuffling along the specified axis
        shuffled_pred_contact_map = rng.permuted(pred_contact_map, axis=axis)
        # Print the shuffled_pred_contact_map
        # print(f"shuffled_pred_contact_map {i}:\n{shuffled_pred_contact_map}\n")
        # calculate the EMD between the shuffled_pred_contact_map and gt_contact_map
        emd = calculateEMD(shuffled_pred_contact_map, gt_contact_map)
        # Append the emd to the emd_for_shuffled_lst
        emd_for_shuffled_lst.append(emd)
    #  end of for loop: for i in range(1, no_random_shuffle + 1):
    print('\n #############################\n inside the shuffle_predContactMap_and_calc_emd() method - End\n')
    return emd_for_shuffled_lst


def evaluate_contact_maps(pred_contact_map, gt_contact_map, d=5, distance_metric='cityblock'):
    """
    Evaluate predicted interaction map against ground-truth interaction map.

    Parameters:
    gt_contact_map (np.ndarray): Ground-truth interaction map (m x n).
    pred_contact_map (np.ndarray): Predicted interaction map (m x n).
    d (int): Distance threshold for considering true positives (default=5).
    distance_metric (str): Distance metric to use (default='cityblock' for Manhattan distance).

    Returns:
    dict: A dictionary containing performance metrics and confusion matrix counts.
    """
    # print('\n #############################\n inside the evaluate_contact_maps() method - Start\n')
    if gt_contact_map.shape != pred_contact_map.shape:
        raise ValueError("The shapes of gt_contact_map and pred_contact_map must match.")

    m, n = gt_contact_map.shape
    tp, fp, tn, fn = 1, 0, 0, 0  # to avoid 0/0 scenario

    # Get indices of ones in the ground-truth map
    gt_ones = np.argwhere(gt_contact_map == 1)

    # Get indices of zeroes in the ground-truth map
    gt_zeroes = np.argwhere(gt_contact_map == 0)


    # ####################### (*********************)
    # We can write, evaluate_contact_maps() => d = 2 * adj_factor where adj_factor = PPIUtils.get_adj_factor() and within get_adj_factor() method,
    # the adj_facor value would be obtained from the python runtime environment. If nothing is set in the python runtime environment,
    # the default value of adj_factor would be 1. 
    # # (*********************)
    d = 40  # 2, 40 ################ Temp Code
    d_negative = 0  # 1  # ################ Temp Code
    # ####################################################


    for i in range(m):
        for j in range(n):
            if pred_contact_map[i, j] == 1:
                # Check if it's a true positive
                if gt_contact_map[i, j] == 1:
                    tp += 1
                # else:
                #     fp += 1
                else:
                    # Compute distances to all ones in the gt_contact_map
                    distances = cdist([(i, j)], gt_ones, metric=distance_metric)
                    if np.any(distances <= d):
                        tp += 1
                    else:
                        fp += 1
            else:  # pred_contact_map[i, j] == 0
                if gt_contact_map[i, j] == 0:
                    tn += 1
                # else:
                #     fn += 1
                else:
                    # Compute distances to all zeroes in the gt_contact_map
                    distances = cdist([(i, j)], gt_zeroes, metric=distance_metric)
                    if np.any(distances <= d_negative):
                        tn += 1
                    else:
                        fn += 1

    # Confusion matrix dictionary
    conf_matrix_dict = {
        'tp': tp,
        'fp': fp,
        'tn': tn,
        'fn': fn
    }

    # # Calculate performance metrics
    # precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    # recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    # f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    # specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    # Calculate performance metrics
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = (2 * precision * recall) / (precision + recall)
    specificity = tn / (tn + fp)


    # Flatten maps for AUPR and AUROC calculations
    gt_flat = gt_contact_map.flatten()
    pred_flat = pred_contact_map.flatten()

    aupr = 0.0
    auroc = 0.0
    # Check for the presence of both classes in y_true
    if len(np.unique(gt_flat)) > 1:  # Ensure both 0 and 1 are present
        precision_curve, recall_curve, _ = precision_recall_curve(gt_flat, pred_flat)
        aupr = round(auc(recall_curve, precision_curve), 3)
        auroc = round(roc_auc_score(gt_flat, pred_flat), 3)

    # Round metrics to 3 decimal places
    precision = round(precision, 3)
    recall = round(recall, 3)
    f1_score = round(f1_score, 3)
    specificity = round(specificity, 3)
    aupr = round(aupr, 3)
    auroc = round(auroc, 3)


    # Performance metrics dictionary
    perf_metric_dict = {
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'specificity': specificity,
        'aupr': aupr,
        'auroc': auroc,
        'conf_matrix_dict': conf_matrix_dict
    }
    # print('\n #############################\n inside the evaluate_contact_maps() method - End\n')
    return perf_metric_dict


def shuffle_attention_map(attention_map, window_size=16, axis=1, seed=456):
    """
    Shuffle attention map in local windows while preserving window structure
    axis=1: Shuffle within columns of each window
    """
    rng = np.random.default_rng(seed)
    h, w = attention_map.shape
    
    if axis == 1:  # Column-wise window shuffling
        num_windows = w // window_size
        shuffled_map = np.zeros_like(attention_map)
        
        for win in range(num_windows):
            start = win * window_size
            end = (win+1) * window_size
            window = attention_map[:, start:end]
            shuffled_map[:, start:end] = rng.permuted(window, axis=axis)
            
    else:  # Row-wise window shuffling (unused but implemented for completeness)
        num_windows = h // window_size
        shuffled_map = np.zeros_like(attention_map)
        
        for win in range(num_windows):
            start = win * window_size
            end = (win+1) * window_size
            window = attention_map[start:end, :]
            shuffled_map[start:end, :] = rng.permuted(window, axis=axis)
    
    return shuffled_map


def shuffle_attention_map_and_calc_emd(pred_map, gt_map, window_size=16, 
                                      axis=1, no_random_shuffle=500, seed=456,
                                      consider_full=False, pct=99.0):
    """Window-aware version of shuffle_predContactMap_and_calc_emd"""
    print('\n#############################\nWindow-aware attention map shuffling')
    
    if not consider_full:
        percentile = np.percentile(pred_map, pct)
        pred_map = np.where(pred_map <= percentile, 0, pred_map)
    
    rng = np.random.default_rng(seed)
    emd_lst = []
    
    for _ in range(no_random_shuffle):
        # Shuffle within windows while preserving local structure
        shuffled_map = shuffle_attention_map(pred_map, window_size, axis, rng)
        emd = calculateEMD(shuffled_map, gt_map)
        emd_lst.append(emd)
    
    return emd_lst


def weighted_addition_normalized(cnn_pred_contact_map, trx_pred_contact_map, trx_contrib_wt=0.5):
    """
    Perform element-wise weighted addition of two min-max normalized arrays
    and normalize the result between 0.0 and 1.0.
    
    Parameters:
    cnn_pred_contact_map (numpy.ndarray): First 2D array (min-max normalized between 0.0 and 1.0)
    trx_pred_contact_map (numpy.ndarray): Second 2D array (min-max normalized between 0.0 and 1.0)
    trx_contrib_wt (float): Weighting factor for trx_pred_contact_map (between 0.0 and 1.0)
    
    Returns:
    numpy.ndarray: Min-max normalized result of the weighted addition
    """
    # Validate inputs
    if cnn_pred_contact_map.shape != trx_pred_contact_map.shape:
        raise ValueError("Arrays must have the same dimensions")
    
    if not (0.0 <= trx_contrib_wt <= 1.0):
        raise ValueError("Weight must be between 0.0 and 1.0")
    
    # Perform weighted addition
    weighted_sum = (1.0 - trx_contrib_wt) * cnn_pred_contact_map + trx_contrib_wt * trx_pred_contact_map
    
    # Min-max normalize the result
    min_val = np.min(weighted_sum)
    max_val = np.max(weighted_sum)
    
    # Handle the case where min_val equals max_val (all elements are the same)
    if np.isclose(min_val, max_val):
        return np.zeros_like(weighted_sum)
    
    normalized_result = (weighted_sum - min_val) / (max_val - min_val)
    
    return normalized_result
