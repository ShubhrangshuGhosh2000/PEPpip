import os
import sys
from pathlib import Path

path_root = Path(__file__).parents[3]  # upto 'codebase' folder
sys.path.insert(0, str(path_root))
# print(sys.path)

from utils import dl_reproducible_result_util
import glob
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
# from matplotlib.colors import Normalize

from utils import DockUtils
from utils import PPIPUtils


def create_predicted_contact_map(root_path='./', model_path='./', docking_version='5_5'):
    print('\n #############################\n inside the create_predicted_contact_map() method - Start\n')
    print('\n########## docking_version: ' + str(docking_version))

    # retrieve the docking_version specific concatenated prediction result
    print('retrieving the docking_version specific concatenated prediction result')
    test_tag = model_path.split('/')[-1]
    xai_result_dir = os.path.join(root_path, f'dataset/proc_data_tl_feat_to_img/img_p2ip_cnn/xai_dock_{docking_version}/{test_tag}')
    concat_pred_res_file_nm_with_loc = os.path.join(xai_result_dir, f'concat_pred_res_dock_{docking_version}.csv')
    concat_pred_res_df = pd.read_csv(concat_pred_res_file_nm_with_loc)
    # separate true-positive (tp) and false-negative (fn) result
    print('separating true-positive (tp) and false-negative (fn) result')
    tp_df = concat_pred_res_df[ concat_pred_res_df['actual_res'] == concat_pred_res_df['pred_res']]
    tp_df = tp_df.reset_index(drop=True)
    fn_df = concat_pred_res_df[ concat_pred_res_df['actual_res'] != concat_pred_res_df['pred_res']]
    fn_df = fn_df.reset_index(drop=True)

    # read the dock_test_lenLimit_{limiting_len}.tsv in a pandas dataframe
    limiting_len = 400
    print(f'read the dock_test_lenLimit_{limiting_len}.tsv in a pandas dataframe')
    dock_test_lenLimit_file_nm_with_loc = os.path.join(root_path, f'dataset/preproc_data_docking_BM_{docking_version}/derived_feat/dock', f'dock_test_lenLimit_{limiting_len}.tsv')
    dock_test_lenLimit_df = pd.read_csv(dock_test_lenLimit_file_nm_with_loc, delimiter='\t', header=None, names=['prot1_id', 'prot2_id', 'label'])
    # lenLimit_prot1_id_lst and lenLimit_prot2_id_lst will be used to add 'upto400' and 'moreThan400' categories in the predicted interaction map names
    lenLimit_prot1_id_lst = dock_test_lenLimit_df['prot1_id'].tolist()
    lenLimit_prot2_id_lst = dock_test_lenLimit_df['prot2_id'].tolist()

    # create folders for the predicted interaction maps for the tp and fn result
    print('creating folders for the predicted interaction maps for the tp and fn result')
    tp_pred_contact_map_loc = os.path.join(xai_result_dir, 'tp_pred_contact_map')
    PPIPUtils.createFolder(tp_pred_contact_map_loc)
    fn_pred_contact_map_loc = os.path.join(xai_result_dir, 'fn_pred_contact_map')
    PPIPUtils.createFolder(fn_pred_contact_map_loc)

    postproc_attr_result_dir = os.path.join(xai_result_dir, 'attr_postproc')
    # create predicted interaction map for the tp cases
    print('\n creating predicted interaction map for the tp cases')
    # iterate over the tp_df and create predicted interaction map
    for index, row in tp_df.iterrows():
        print('tp: starting ' + str(index) + '-th entry out of ' + str(tp_df.shape[0]-1))
        idx, prot1_id, prot2_id = row['idx'], row['prot1_id'], row['prot2_id']
        # fetch the attribution map corresponding to idx
        postproc_res_dict_file_nm_with_loc = os.path.join(postproc_attr_result_dir, 'idx_' + str(idx) + '.pkl')
        postproc_res_dict = joblib.load(postproc_res_dict_file_nm_with_loc)
        # retrieve 'non_zero_pxl_indx_lst_lst' from postproc_res_dict
        non_zero_pxl_indx_lst_lst = postproc_res_dict['non_zero_pxl_indx_lst_lst']
        # retrieve 'non_zero_pxl_tot_attr_lst' from postproc_res_dict
        non_zero_pxl_tot_attr_lst = postproc_res_dict['non_zero_pxl_tot_attr_lst']
        # create predicted 2d attribution map
        print('tp: creating predicted 2d attribution map')
        pred_attr_map_2d = create_pred_attr_map_2d(non_zero_pxl_indx_lst_lst, non_zero_pxl_tot_attr_lst)
        # create predicted interaction map for tp cases by performing an usual min-max normalization
        print('tp: creating predicted interaction map for tp cases by performing an usual min-max normalization')
        min_val, max_val = pred_attr_map_2d.min(), pred_attr_map_2d.max()
        pred_contact_map = (pred_attr_map_2d - min_val) / (max_val - min_val)
        # save pred_contact_map
        print('tp: saving pred_contact_map')
        len_tag = 'moreThan400'
        if((prot1_id in lenLimit_prot1_id_lst) and (prot2_id in lenLimit_prot2_id_lst)):
            # both the proteins are having length <=400
            len_tag = 'upto400'
        protein_name, chain_1_name = prot1_id.split('_')
        protein_name, chain_2_name = prot2_id.split('_')
        print(f'protein_name: {protein_name} :: chain_1_name: {chain_1_name} :: chain_2_name: {chain_2_name} : len_tag: {len_tag}')
        pred_contact_map_location = os.path.join(tp_pred_contact_map_loc, f"pred_contact_map_{protein_name}_{chain_1_name}_{chain_2_name}_{len_tag}.pkl")
        joblib.dump(pred_contact_map, pred_contact_map_location)
        print(f'tp: pred_contact_map is saved as {pred_contact_map_location}')
    # end of for loop: for index, row in tp_df.iterrows():
    # create predicted interaction map for the fn cases
    print('\n creating predicted interaction map for the fn cases')
    # iterate over the fn_df and create predicted interaction map
    for index, row in fn_df.iterrows():
        print('fn: starting ' + str(index) + '-th entry out of ' + str(fn_df.shape[0]-1))
        idx, prot1_id, prot2_id = row['idx'], row['prot1_id'], row['prot2_id']
        # fetch the attribution map corresponding to idx
        postproc_res_dict_file_nm_with_loc = os.path.join(postproc_attr_result_dir, 'idx_' + str(idx) + '.pkl')
        postproc_res_dict = joblib.load(postproc_res_dict_file_nm_with_loc)
        # retrieve 'non_zero_pxl_indx_lst_lst' from postproc_res_dict
        non_zero_pxl_indx_lst_lst = postproc_res_dict['non_zero_pxl_indx_lst_lst']
        # retrieve 'non_zero_pxl_tot_attr_lst' from postproc_res_dict
        non_zero_pxl_tot_attr_lst = postproc_res_dict['non_zero_pxl_tot_attr_lst']
        # create predicted 2d attribution map
        print('fn: creating predicted 2d attribution map')
        pred_attr_map_2d = create_pred_attr_map_2d(non_zero_pxl_indx_lst_lst, non_zero_pxl_tot_attr_lst)
        # create predicted interaction map for fn cases by performing an usual min-max normalization
        print('fn: creating predicted interaction map for fn cases by performing an usual min-max normalization')
        min_val, max_val = pred_attr_map_2d.min(), pred_attr_map_2d.max()
        # pred_contact_map = (pred_attr_map_2d - min_val) / (max_val - min_val)  # usual min-max normalization
        # ## A note on justification for using min-max normalization in a reverse way: In the hybrid approach MaTPIP is combined on 
        # ImgPIP and MaTPIP has high recall value. So, in real-time, the cases where MaTPIP would predict as negative but ImgPIP predictis as positive
        # can be treated as false positive and will be treated as negative and in that case finding the interaction map does not make much sense. But if MaTPIP predicts
        # non-negative i.e. positive with high confidence and ImgPIP predicts negative then it can be treated as false negative and in this case (false 
        # negative), min-max normalization in a reverse way can be used.  
        pred_contact_map = (max_val - pred_attr_map_2d) / (max_val - min_val)  # ################ min-max normalization in a reverse way
        # save pred_contact_map
        print('fn: saving pred_contact_map')
        len_tag = 'moreThan400'
        if((prot1_id in lenLimit_prot1_id_lst) and (prot2_id in lenLimit_prot2_id_lst)):
            # both the proteins are having length <=400
            len_tag = 'upto400'
        protein_name, chain_1_name = prot1_id.split('_')
        protein_name, chain_2_name = prot2_id.split('_')
        print(f'protein_name: {protein_name} :: chain_1_name: {chain_1_name} :: chain_2_name: {chain_2_name} : len_tag: {len_tag}')
        pred_contact_map_location = os.path.join(fn_pred_contact_map_loc, f"pred_contact_map_{protein_name}_{chain_1_name}_{chain_2_name}_{len_tag}.pkl")
        joblib.dump(pred_contact_map, pred_contact_map_location)
        print(f'fn: pred_contact_map is saved as {pred_contact_map_location}')
    # end of for loop: for index, row in fn_df.iterrows():
    print('\n #############################\n inside the create_predicted_contact_map() method - End\n')


def create_pred_attr_map_2d(non_zero_pxl_indx_lst_lst, non_zero_pxl_tot_attr_lst):
    """
    Create a 2D numpy array of attributions based on the given lists.

    Parameters:
    - non_zero_pxl_indx_lst_lst (list): A list of lists where each inner list contains the non-zero pixel index (H,W) i.e. 2 integers
    - non_zero_pxl_tot_attr_lst (list): A list of floats of the same length as 1st argument list, containing the total attribution for all the 3 channels together for the non-zero pixel.

    Returns:
    - numpy.ndarray: A 2D numpy array initiated with zeros and populated based on input argument lists.

    Example:
    >>> lst1 = [[3, 1], [4, 2], [2, 0], [5, 2]]
    >>> lst2 = [0.1, 0.2, 0.3, 0.4]
    >>> create_pred_attr_map_2d(lst1, lst2)
    array([[0. , 0. , 0. ],
           [0. , 0. , 0. ],
           [0.3, 0. , 0. ],
           [0. , 0.1, 0. ],
           [0. , 0., 0.2 ],
           [0. , 0., 0.4 ]])
    """
    # Find 'h_max' and 'w_max' from the input non_zero_pxl_indx_lst_lst
    h_max = max(non_zero_pxl_indx_lst_lst, key=lambda x: x[0])[0]
    w_max = max(non_zero_pxl_indx_lst_lst, key=lambda x: x[1])[1]

    # Create a numpy array initialized with zeros of dimensions (h_max + 1) by (w_max + 1)
    pred_attr_map_2d = np.zeros((h_max+1, w_max+1))

    # Iterate over non_zero_pxl_indx_lst_lst and populate pred_attr_map_2d based on non_zero_pxl_tot_attr_lst
    for i in range(len(non_zero_pxl_indx_lst_lst)):
        x, y = non_zero_pxl_indx_lst_lst[i]
        pred_attr_map_2d[x, y] = non_zero_pxl_tot_attr_lst[i]
    return pred_attr_map_2d


def calculate_emd_betwn_gt_and_pred_contact_maps(root_path='./', model_path='./', docking_version='5_5', consider_full=False):
    print('\n #############################\n inside the calculate_emd_betwn_gt_and_pred_contact_maps() method - Start\n')
    print('\n########## docking_version: ' + str(docking_version))
    gt_contact_map_dir = os.path.join(root_path, f"dataset/preproc_data_docking_BM_{docking_version}/prot_gt_contact_map")
    test_tag = model_path.split('/')[-1]
    xai_result_dir = os.path.join(root_path, f'dataset/proc_data_tl_feat_to_img/img_p2ip_cnn/xai_dock_{docking_version}/{test_tag}')
    tp_pred_contact_map_loc = os.path.join(xai_result_dir, 'tp_pred_contact_map')
    fn_pred_contact_map_loc = os.path.join(xai_result_dir, 'fn_pred_contact_map')
    emd_result_dir = os.path.join(xai_result_dir, 'emd_result')
    PPIPUtils.createFolder(emd_result_dir)
    contact_heatmap_plt_dir = None
    if(consider_full):
        # consider full-version version of the interaction maps
        contact_heatmap_plt_dir = os.path.join(emd_result_dir, 'contact_heatmap_plt_full')
    else:
        # consider 99 percentile version of the interaction maps
        contact_heatmap_plt_dir = os.path.join(emd_result_dir, 'contact_heatmap_plt_9Xp')
    PPIPUtils.createFolder(contact_heatmap_plt_dir)

    # read the dock_test.tsv file in a pandas dataframe
    # doc_test_tsv_fl_nm_loc = os.path.join(root_path, f'dataset/preproc_data_docking_BM_{docking_version}/derived_feat/dock', 'dock_test.tsv')
    doc_test_tsv_fl_nm_loc = os.path.join(root_path, f'dataset/preproc_data_docking_BM_{docking_version}/derived_feat/dock', 'dock_test_lenLimit_400.tsv')
    dock_test_df = pd.read_csv(doc_test_tsv_fl_nm_loc, sep='\t', header=None, names=['prot_1_id', 'prot_2_id', 'test_labels'])
    # iterate over the dock_test_df and for each pair of participating proteins in each row calculate the EMD between the ground-truth (gt) contact
    # map and the predicted interaction map
    dock_test_row_idx_lst, prot_1_id_lst, prot_2_id_lst, len_tag_lst, tp_fn_lst, emd_lst, abs_diff_lst = [], [], [], [], [], [], []  # lists to create a dataframe later
    for index, row in dock_test_df.iterrows():
        print(f"\n ################# starting {index}-th row out of {dock_test_df.shape[0]-1}\n")
        prot_1_id, prot_2_id = row['prot_1_id'], row['prot_2_id']
        # protein id has the format of [protein_name]_[chain_name]
        protein_name, chain_1_name = prot_1_id.split('_')
        protein_name, chain_2_name = prot_2_id.split('_')
        print(f"protein_name: {protein_name} :: chain_1_name: {chain_1_name} :: chain_2_name: {chain_2_name}")
        # retrieve the pred_contact_map
        print('retrieving the pred_contact_map...')
        # first check whether it belongs to tp_pred_contact_map_loc
        tp_pred_contact_map_path = os.path.join(tp_pred_contact_map_loc, f"pred_contact_map_{protein_name}_{chain_1_name}_{chain_2_name}*.pkl")
        pred_contact_map_lst = glob.glob(tp_pred_contact_map_path, recursive=False)
        if(len(pred_contact_map_lst) == 0):
            print('No pred_contact_map found in tp_pred_contact_map folder and next searching in fn_pred_contact_map folder...')
            # if the desired pred_contact_map does not exist inside tp_pred_contact_map_path, then
            # search inside fn_pred_contact_map_path for the same
            fn_pred_contact_map_path = os.path.join(fn_pred_contact_map_loc, f"pred_contact_map_{protein_name}_{chain_1_name}_{chain_2_name}*.pkl")
            pred_contact_map_lst = glob.glob(fn_pred_contact_map_path, recursive=False)
            if(len(pred_contact_map_lst) == 0):
                raise Exception(f"No pred_contact_map found in tp_pred_contact_map and fn_pred_contact_map folders for \
                                protein_name: {protein_name} :: chain_1_name: {chain_1_name} :: chain_2_name: {chain_2_name}")
            else:  
                # desired pred_contact_map exists inside fn_pred_contact_map folder
                print('desired pred_contact_map exists inside fn_pred_contact_map folder')
                tp_fn_lst.append('fn')
            # end of else block
        else:
            print('desired pred_contact_map exists inside tp_pred_contact_map folder')
            tp_fn_lst.append('tp')
        # end of if-else block: if(len(pred_contact_map_lst) == 0):
        pred_contact_map_name = pred_contact_map_lst[0].split('/')[-1]
        len_tag = pred_contact_map_name.split('.')[0].split('_')[-1]  # f"pred_contact_map_{protein_name}_{chain_1_name}_{chain_2_name}_{len_tag}.pkl"
        print(f'len_tag: {len_tag}')
        pred_contact_map = joblib.load(pred_contact_map_lst[0])

        # retrieve the gt_contact_map
        print('retrieving the gt_contact_map...')
        gt_contact_map_location = os.path.join(gt_contact_map_dir, f"gt_contact_map_{protein_name}_{chain_1_name}_{chain_2_name}.pkl")
        gt_contact_map = joblib.load(gt_contact_map_location)

        # check whether dimensionally both the interaction maps are same
        if(pred_contact_map.shape == gt_contact_map.shape):
           print('dimensionally both the interaction maps (pred_contact_map and gt_contact_map) are same')
        else:
            print(f'!!!!! dimensionally both the interaction maps (pred_contact_map and gt_contact_map) are not the same')
            print(f"pred_contact_map.shape: {pred_contact_map.shape} :: gt_contact_map.shape: {gt_contact_map.shape}")
            raise Exception(f"Dimensionally both the interaction maps (pred_contact_map and gt_contact_map) are not the same for \
                                protein_name: {protein_name} :: chain_1_name: {chain_1_name} :: chain_2_name: {chain_2_name} \
                                pred_contact_map.shape: {pred_contact_map.shape} :: gt_contact_map.shape: {gt_contact_map.shape}")
        
        # calculate the EMD between pred_contact_map and gt_contact_map
        print('Calculating the EMD between pred_contact_map and gt_contact_map')
        if(not consider_full):
            # Calculate the 9X-th percentile value for pred_contact_map
            percentile = np.percentile(pred_contact_map, 99)
            # Set values below 9X-th percentile to zero
            pred_contact_map[pred_contact_map <= percentile] = 0
            # Calculate the 9X-th percentile value for gt_contact_map
            percentile = np.percentile(gt_contact_map, 99)
            # Set values below 9X-th percentile to zero
            gt_contact_map[gt_contact_map <= percentile] = 0

        emd = DockUtils.calculateEMD(pred_contact_map, gt_contact_map)
        print(f'emd: {emd}')

        # generate contact heatmap plot for pred_contact_map
        print('generating contact heatmap plot for pred_contact_map')
        pred_contact_map_plot_path = os.path.join(contact_heatmap_plt_dir, f'prot_{prot_1_id}_prot_{prot_2_id}_{tp_fn_lst[-1]}_pred_contact_map.png')
        generate_pred_contact_heatmap_plot(pred_contact_map, title='Predicted interaction map', row = prot_1_id, col=prot_2_id, save_path=pred_contact_map_plot_path)

        # generate contact heatmap plot for gt_contact_map
        print('generating contact heatmap plot for gt_contact_map')
        gt_contact_map_plot_path = os.path.join(contact_heatmap_plt_dir, f'prot_{prot_1_id}_prot_{prot_2_id}_gt_contact_map.png')
        generate_gt_contact_heatmap_plot(gt_contact_map, title='gt interaction map', row = prot_1_id, col=prot_2_id, save_path=gt_contact_map_plot_path)

        # calculate the absolute difference between pred_contact_map and gt_contact_map
        abs_diff = DockUtils.calcMatrixAbsDiff(pred_contact_map, gt_contact_map)

        # populate the lists required to create a df later
        dock_test_row_idx_lst.append(index); prot_1_id_lst.append(prot_1_id); prot_2_id_lst.append(prot_2_id)
        len_tag_lst.append(len_tag); emd_lst.append(emd); abs_diff_lst.append(abs_diff)
    # end of for loop: for index, row in dock_test_df.iterrows():
    # create the dataframe and save it
    emd_col_nm = 'emd_full' if(consider_full) else 'emd_9Xp'
    emd_df = pd.DataFrame({'dock_test_row_idx': dock_test_row_idx_lst, 'prot_1_id': prot_1_id_lst, 'prot_2_id': prot_2_id_lst, 'len_tag_lst': len_tag_lst
                           , 'tp_fn': tp_fn_lst, emd_col_nm: emd_lst, 'abs_diff': abs_diff_lst})
    # save emd_df
    if(consider_full):
        # consider full-version version of the interaction maps
        emd_df.to_csv(os.path.join(emd_result_dir, f'gt_vs_pred_emd_full.csv'))
    else:
        # consider 99 percentile version of the interaction maps
        emd_df.to_csv(os.path.join(emd_result_dir, f'gt_vs_pred_emd_9Xp.csv'))
    print('\n #################\n inside the calculate_emd_betwn_gt_and_pred_contact_maps() method - End\n')


def generate_pred_contact_heatmap_plot(data_array_2d, title='Heatmap Plot with Colormap', row='prot_1_id', col='prot_2_id', save_path='heatmap_plot.png'):
    """
    Generate a heatmap plot using Matplotlib for the given numpy array.

    Parameters:
    - data_array_2d (numpy.ndarray): Input 2D array of floats with values between 0.0 and 1.0.
    - save_path (str): Path to save the generated heatmap plot image (default: 'heatmap_plot.png').

    Returns:
    - None

    Example:
    >>> data_array_2d = np.random.rand(5, 8)  # Example 5x8 array of random floats
    >>> generate_pred_contact_heatmap_plot(data_array_2d, save_path='my_heatmap_plot.png')
    """
    # # Ensure that input array is a numpy array
    data_array_2d = np.array(data_array_2d)
    # Check if the array has the correct dimension (2D)
    if data_array_2d.ndim != 2:
        raise ValueError("Input array must be a 2D numpy array")
    # Calculate the 90th percentile value
    # percentile = np.percentile(data_array_2d, 99)
    # print(f'percentile: {percentile}')

    # Set values below 90th percentile to zero
    # data_array_2d[data_array_2d <= percentile] = 0

    # In this program, the colormap is set to gray_r (reverse of the grayscale colormap), and 
    # the normalization is adjusted such that values greater than the 90th percentile appear darker in the colormap.
    # Create a colormap with increasing density of black for values greater than 90th percentile
    cmap = plt.cm.gray_r
    # norm = Normalize(vmin=0, vmax=percentile)

    # Create a custom heatmap plot
    plt.imshow(data_array_2d, cmap=cmap, norm=None, interpolation=None, origin='upper', aspect=None)
    # Set colorbar
    cbar = plt.colorbar()
    # cbar.set_label('Color Scale (0.0 to 1.0)')
    # Invert Y-axis to start from the top
    # plt.gca().invert_yaxis()
    # Set labels and title
    plt.xlabel(f'{col} (Col Idx : Y Axis)')
    plt.ylabel(f'{row} (Row Indices : X Axis)')
    plt.title(f'{title}')
    # Save the plot
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    # Show the plot (optional)
    # plt.show()
    # Close the plot
    plt.close()


def generate_gt_contact_heatmap_plot(data_array_2d, title='Heatmap Plot with Colormap', row='prot_1_id', col='prot_2_id', save_path='heatmap_plot.png'):
    """
    Generate a heatmap plot using Matplotlib for the given numpy array.

    Parameters:
    - data_array_2d (numpy.ndarray): Input 2D array of floats with values between 0.0 and 1.0.
    - save_path (str): Path to save the generated heatmap plot image (default: 'heatmap_plot.png').

    Returns:
    - None

    Example:
    >>> data_array_2d = np.random.rand(5, 8)  # Example 5x8 array of random floats
    >>> generate_gt_contact_heatmap_plot(data_array_2d, save_path='my_heatmap_plot.png')
    """
    # # Ensure that input array is a numpy array
    data_array_2d = np.array(data_array_2d)
    # Check if the array has the correct dimension (2D)
    if data_array_2d.ndim != 2:
        raise ValueError("Input array must be a 2D numpy array")
    # Calculate the 90th percentile value
    # percentile = np.percentile(data_array_2d, 99)
    # print(f'percentile: {percentile}')

    # Set values below 90th percentile to zero
    # data_array_2d[data_array_2d < percentile] = 0

    # In this program, the colormap is set to gray_r (reverse of the grayscale colormap), and 
    # the normalization is adjusted such that values greater than the 90th percentile appear darker in the colormap.
    # Create a colormap with increasing density of black for values greater than 90th percentile
    cmap = plt.cm.gray_r
    # norm = Normalize(vmin=0, vmax=percentile)

    # Create a custom heatmap plot
    plt.imshow(data_array_2d, cmap=cmap, norm=None, interpolation=None, origin='upper', aspect=None)
    # Set colorbar
    cbar = plt.colorbar()
    # cbar.set_label('Color Scale (0.0 to 1.0)')
    # Invert Y-axis to start from the top
    # plt.gca().invert_yaxis()
    # Set labels and title
    plt.xlabel(f'{col} (Col Idx : Y Axis)')
    plt.ylabel(f'{row} (Row Indices : X Axis)')
    plt.title(title)
    # Save the plot
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    # Show the plot (optional)
    # plt.show()
    # Close the plot
    plt.close()


if __name__ == '__main__':
    root_path = os.path.join('/scratch/pralaycs/Shubh_Working_Remote/PPI_Wkspc/PPI_Code/mat_p2ip_prj_working/')
    model_path = os.path.join(root_path, 'dataset/proc_data_tl_feat_to_img/img_p2ip_cnn/train/ResNet_manTlStruct_r400n18D')
    # partial_model_name = 'ImgP2ipCnn'

    consider_full_lst = [True, False]  # True, False  # consider full-version or 99 percentile version of the interaction maps
    docking_version_lst = ['5_5', '5_5']  # '5_5', '5_5'
    for docking_version in docking_version_lst:
        print('\n########## docking_version: ' + str(docking_version))
        create_predicted_contact_map(root_path=root_path, model_path=model_path, docking_version=docking_version)
        for consider_full in consider_full_lst:
            print('\n########## consider_full: ' + str(consider_full))
            calculate_emd_betwn_gt_and_pred_contact_maps(root_path=root_path, model_path=model_path, docking_version=docking_version, consider_full=consider_full)
        # end of for loop: for consider_full in consider_full_lst:
    # end of for loop: for docking_version in docking_version_lst:
