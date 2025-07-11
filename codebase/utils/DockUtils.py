import os
import sys
from pathlib import Path
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.metrics import precision_recall_curve, roc_auc_score, auc
path_root = Path(__file__).parents[1]  
sys.path.insert(0, str(path_root))


def calculateEMD(arr_2d_1, arr_2d_2):
    cost_matrix = cdist(arr_2d_1, arr_2d_2)
    assignment = linear_sum_assignment(cost_matrix)
    emd = cost_matrix[assignment].sum() / cost_matrix.shape[0]  
    return emd


def calcMatrixAbsDiff(matrix1, matrix2):
    matrix1 = np.array(matrix1)
    matrix2 = np.array(matrix2)
    if matrix1.shape != matrix2.shape:
        raise ValueError("Input matrices must have the same shape")
    diff_sum = np.sum(np.abs(matrix1 - matrix2))
    return diff_sum


def shuffle_predContactMap_and_calc_emd(pred_contact_map, gt_contact_map, axis=1, no_random_shuffle=500, seed=456
                                        , consider_full=False, pct=99.0):
    if(not consider_full):
        percentile = np.percentile(pred_contact_map, pct)
        pred_contact_map[pred_contact_map <= percentile] = 0
    rng = np.random.default_rng(seed)
    emd_for_shuffled_lst = []
    for i in range(1, no_random_shuffle + 1):
        shuffled_pred_contact_map = rng.permuted(pred_contact_map, axis=axis)
        emd = calculateEMD(shuffled_pred_contact_map, gt_contact_map)
        emd_for_shuffled_lst.append(emd)
    return emd_for_shuffled_lst


def get_adj_factor():
    try:
        return float(os.environ.get("adj_factor", 1.0))
    except ValueError:
        return 1


def evaluate_contact_maps(pred_contact_map, gt_contact_map):
    if gt_contact_map.shape != pred_contact_map.shape:
        raise ValueError("The shapes of gt_contact_map and pred_contact_map must match.")
    top_Lby5_prec = find_topL_prec_score(gt_contact_map, pred_contact_map, metric_type="top_Lby5_prec")
    top_Lby10_prec = find_topL_prec_score(gt_contact_map, pred_contact_map, metric_type="top_Lby10_prec")
    top_50_prec = find_topL_prec_score(gt_contact_map, pred_contact_map, metric_type="top_50_prec")
    perf_metric_dict = {
        'top_Lby5_prec': top_Lby5_prec,
        'top_Lby10_prec': top_Lby10_prec,
        'top_50_prec': top_50_prec
    }
    return perf_metric_dict


def find_topL_prec_score(gt_contact_map, pred_contact_map, metric_type="top_50_prec"):
    if gt_contact_map.shape != pred_contact_map.shape:
        raise ValueError("The shapes of gt_contact_map and pred_contact_map must match.")
    m, n = gt_contact_map.shape
    if metric_type == "top_Lby5_prec":
        l = min(m, n) // 5
    elif metric_type == "top_Lby10_prec":
        l = min(m, n) // 10
    elif metric_type == "top_50_prec":
        l = 50
    else:
        raise ValueError("metric_type must be one of 'top_Lby5_prec', 'top_Lby10_prec', or 'top_50_prec'")
    l = min(l, m * n)
    if l == 0:
        l = 1  
    flat_pred = pred_contact_map.flatten()
    sorted_indices = np.argsort(flat_pred)[::-1]  
    top_l_flat_indices = sorted_indices[:l]
    top_l_row_indices, top_l_col_indices = np.unravel_index(top_l_flat_indices, (m, n))
    top_l_positions = list(zip(top_l_row_indices, top_l_col_indices))
    gt_ones = np.argwhere(gt_contact_map == 1)
    d = 2 * get_adj_factor()
    tp = 0
    for i, j in top_l_positions:
        if gt_contact_map[i, j] == 1:
            tp += 1
        else:
            if len(gt_ones) > 0:
                distances = cdist([(i, j)], gt_ones, metric='cityblock')
                if np.any(distances <= d):
                    tp += 1
    top_l_precision = tp / l if l > 0 else 0.0
    return top_l_precision


def shuffle_attention_map(attention_map, window_size=16, axis=1, seed=456):
    rng = np.random.default_rng(seed)
    h, w = attention_map.shape
    if axis == 1:  
        num_windows = w // window_size
        shuffled_map = np.zeros_like(attention_map)
        for win in range(num_windows):
            start = win * window_size
            end = (win+1) * window_size
            window = attention_map[:, start:end]
            shuffled_map[:, start:end] = rng.permuted(window, axis=axis)
    else:  
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
    if not consider_full:
        percentile = np.percentile(pred_map, pct)
        pred_map = np.where(pred_map <= percentile, 0, pred_map)
    rng = np.random.default_rng(seed)
    emd_lst = []
    for _ in range(no_random_shuffle):
        shuffled_map = shuffle_attention_map(pred_map, window_size, axis, rng)
        emd = calculateEMD(shuffled_map, gt_map)
        emd_lst.append(emd)
    return emd_lst


def weighted_addition_normalized(cnn_pred_contact_map, trx_pred_contact_map, trx_contrib_wt=0.5):
    if cnn_pred_contact_map.shape != trx_pred_contact_map.shape:
        raise ValueError("Arrays must have the same dimensions")
    if not (0.0 <= trx_contrib_wt <= 1.0):
        raise ValueError("Weight must be between 0.0 and 1.0")
    weighted_sum = (1.0 - trx_contrib_wt) * cnn_pred_contact_map + trx_contrib_wt * trx_pred_contact_map
    min_val = np.min(weighted_sum)
    max_val = np.max(weighted_sum)
    if np.isclose(min_val, max_val):
        return np.zeros_like(weighted_sum)
    normalized_result = (weighted_sum - min_val) / (max_val - min_val)
    return normalized_result
