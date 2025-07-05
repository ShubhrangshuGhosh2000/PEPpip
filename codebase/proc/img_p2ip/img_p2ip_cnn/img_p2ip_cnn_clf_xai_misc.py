import os
import sys
from pathlib import Path

path_root = Path(__file__).parents[3]  # upto 'codebase' folder
sys.path.insert(0, str(path_root))
# print(sys.path)

from utils import dl_reproducible_result_util
import glob
import joblib
import pandas as pd


def compare_attr_rg_vs_b(root_path='./', model_path='./', spec_type='ecoli'):
    print('\n #############################\n inside the compare_attr_rg_vs_b() method - Start\n')
    print('\n########## spec_type: ' + str(spec_type)) 
    # search and load (one by one) all the pkl files containing 
    # attribution post-process dictionaries for the given species type
    print("searching and loading (one by one) all the pkl files containing attribution post-process dictionaries")
    test_tag = model_path.split('/')[-1]
    xai_result_dir = os.path.join(root_path, 'dataset/proc_data_tl_feat_to_img/img_p2ip_cnn/xai/dscript_other_spec_' + test_tag, spec_type)
    attr_postproc_res_dict_loc = os.path.join(xai_result_dir, spec_type + '_attr_postproc', 'idx_*.pkl')
    pkl_file_nm_lst = glob.glob(attr_postproc_res_dict_loc)
    len_pkl_file_nm_lst = len(pkl_file_nm_lst)
    # iterate over the pkl_file_nm_lst and load the respective pkl file and process it
    overall_attr_rg_vs_b_lst = []
    for itr in range(len_pkl_file_nm_lst):
        if((itr+1) % 20 == 0):  print('spec_type: ' + spec_type + ' : outer loop: starting ' + str(itr+1) + 'th iteration out of ' + str(len_pkl_file_nm_lst))
        indiv_pkl_fl_nm = pkl_file_nm_lst[itr]
        attr_postproc_res_dict = joblib.load(indiv_pkl_fl_nm)
        attr_compare_dict = {dict_key: attr_postproc_res_dict[dict_key] for dict_key in attr_postproc_res_dict if dict_key \
                              not in ['non_zero_pxl_indx_lst_lst', 'non_zero_pxl_attr_lst_lst', 'non_zero_pxl_tot_attr_lst']}
        # initialize the comparison counts
        attr_rg_greater_cnt = attr_b_greater_cnt = 0
        # non_zero_pxl_attr_lst_lst is a list of lists where each inner list contains the non-zero pixel attributions i.e. 3 floats for 3 channels (R,G,B)
        non_zero_pxl_attr_lst_lst = attr_postproc_res_dict['non_zero_pxl_attr_lst_lst']
        # iterate over non_zero_pxl_attr_lst_lst and update the comparison counts
        for non_zero_pxl_attr_lst in non_zero_pxl_attr_lst_lst:
            attr_r_channel, attr_g_channel, attr_b_channel = non_zero_pxl_attr_lst
            if((attr_r_channel + attr_g_channel) > attr_b_channel):
                attr_rg_greater_cnt += 1
            else:
                attr_b_greater_cnt += 1
        # end of for loop: for non_zero_pxl_attr_lst in non_zero_pxl_attr_lst_lst:
        attr_compare_dict['attr_rg_greater_cnt'] = attr_rg_greater_cnt
        attr_compare_dict['attr_b_greater_cnt'] = attr_b_greater_cnt
        # append attr_compare_dict in overall_attr_rg_vs_b_lst
        overall_attr_rg_vs_b_lst.append(attr_compare_dict)
    # end of for loop: for itr in range(len_pkl_file_nm_lst):
    # create a df from overall_attr_rg_vs_b_lst and save it
    print('creating and saving overall_attr_rg_vs_b_df ...')
    overall_attr_rg_vs_b_df = pd.DataFrame(overall_attr_rg_vs_b_lst)
    overall_attr_df_nm_loc = os.path.join(xai_result_dir, spec_type + '_attr_rg_vs_b.csv')
    overall_attr_rg_vs_b_df.to_csv(overall_attr_df_nm_loc, index=False)
    print('\n #############################\n inside the compare_attr_rg_vs_b() method - End\n')


def calc_avg_attr_rg_vs_b(root_path='./', model_path='./'):
    print('\n #############################\n inside the calc_avg_attr_rg_vs_b() method - Start\n')
    spec_type_lst = ['ecoli', 'fly', 'mouse', 'worm', 'yeast']  # human, ecoli, fly, mouse, worm, yeast
    avg_attr_rg_vs_b_per_spec_dict_lst = []
    for spec_type in spec_type_lst:
        print('\n########## spec_type: ' + str(spec_type))
        avg_attr_rg_vs_b_per_spec_dict = {'species': spec_type}
        # load species specific attr_rg_vs_b.csv file
        test_tag = model_path.split('/')[-1]
        xai_result_dir = os.path.join(root_path, 'dataset/proc_data_tl_feat_to_img/img_p2ip_cnn/xai/dscript_other_spec_' + test_tag, spec_type)
        attr_df_nm_loc = os.path.join(xai_result_dir, spec_type + '_attr_rg_vs_b.csv')
        attr_rg_vs_b_df = pd.read_csv(attr_df_nm_loc)
        # take the average/mean of 'attr_rg_greater_cnt' column and 'attr_b_greater_cnt' column
        avg_ser = attr_rg_vs_b_df.loc[:, ['attr_rg_greater_cnt', 'attr_b_greater_cnt']].mean(axis=0)
        avg_arr = avg_ser.to_numpy()
        avg_attr_rg_vs_b_per_spec_dict['attr_rg_greater_avg_cnt'] = avg_arr[0]
        avg_attr_rg_vs_b_per_spec_dict['attr_b_greater_avg_cnt'] = avg_arr[1]
        avg_attr_rg_vs_b_per_spec_dict_lst.append(avg_attr_rg_vs_b_per_spec_dict)
    # end of for loop: for spec_type in spec_type_lst:
    # create a df from avg_attr_rg_vs_b_per_spec_dict_lst and save it
    print('creating and saving avg_attr_rg_vs_b_per_spec_df ...')
    avg_attr_rg_vs_b_per_spec_df = pd.DataFrame(avg_attr_rg_vs_b_per_spec_dict_lst)
    avg_attr_rg_vs_b_per_spec_df_nm_loc = os.path.join(root_path, 'dataset/proc_data_tl_feat_to_img/img_p2ip_cnn/xai/dscript_other_spec_' + \
                                                       test_tag, 'avg_attr_rg_vs_b.csv')
    avg_attr_rg_vs_b_per_spec_df.to_csv(avg_attr_rg_vs_b_per_spec_df_nm_loc, index=False)
    print('\n #############################\n inside the calc_avg_attr_rg_vs_b() method - End\n')
    


if __name__ == '__main__':
    root_path = os.path.join('/scratch/pralaycs/Shubh_Working_Remote/PPI_Wkspc/PPI_Code/mat_p2ip_prj_working/')
    model_path = os.path.join(root_path, 'dataset/proc_data_tl_feat_to_img/img_p2ip_cnn/train/ResNet_expt_r400n18D')

    # spec_type_lst = ['ecoli', 'fly', 'mouse', 'worm', 'yeast']  # human, ecoli, fly, mouse, worm, yeast
    # # spec_type_lst = ['yeast']  # human, ecoli, fly, mouse, worm, yeast
    # for spec_type in spec_type_lst:
    #     print('\n########## spec_type: ' + str(spec_type))
    #     compare_attr_rg_vs_b(root_path=root_path, model_path=model_path, spec_type=spec_type)
    # # end of for loop: for spec_type in spec_type_lst:

    calc_avg_attr_rg_vs_b(root_path=root_path, model_path=model_path)
