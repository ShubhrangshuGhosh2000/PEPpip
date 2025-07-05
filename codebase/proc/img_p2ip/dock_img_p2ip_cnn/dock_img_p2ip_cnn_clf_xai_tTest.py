import os
import sys
from pathlib import Path

path_root = Path(__file__).parents[3]  # upto 'codebase' folder
sys.path.insert(0, str(path_root))
# print(sys.path)

from utils import dl_reproducible_result_util
import glob
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import scipy.stats as stats

from utils import DockUtils
from utils import PPIPUtils


def calc_candidatePPI_p_val(root_path='./', model_path='./', docking_version='4_0', no_random_shuffle=500, consider_fn=False, consider_full=False):
    print('\n #############################\n inside calc_candidatePPI_p_val() method - Start\n')
    print('\n########## docking_version: ' + str(docking_version))
    gt_contact_map_dir = os.path.join(root_path, f"dataset/preproc_data_docking_BM_{docking_version}/prot_gt_contact_map")
    test_tag = model_path.split('/')[-1]
    xai_result_dir = os.path.join(root_path, f'dataset/proc_data_tl_feat_to_img/img_p2ip_cnn/xai_dock_{docking_version}/{test_tag}')
    tp_pred_contact_map_loc = os.path.join(xai_result_dir, 'tp_pred_contact_map')
    fn_pred_contact_map_loc = os.path.join(xai_result_dir, 'fn_pred_contact_map')
    emd_result_dir = os.path.join(xai_result_dir, 'emd_result')

    # read gt_vs_pred_emd.csv file in a pandas dataframe
    print('reading gt_vs_pred_emd.csv')
    gt_vs_pred_emd_df = None
    if(consider_full):
        # consider full-version version of the interaction maps
        gt_vs_pred_emd_df = pd.read_csv(os.path.join(emd_result_dir, f'gt_vs_pred_emd_full.csv'))
    else:
        # consider 99 percentile version of the interaction maps
        gt_vs_pred_emd_df = pd.read_csv(os.path.join(emd_result_dir, f'gt_vs_pred_emd_9Xp.csv'))

    # separate true-positive (tp) and false-negative (fn) result
    print('separating true-positive (tp) and false-negative (fn) result')
    tp_df = gt_vs_pred_emd_df[ gt_vs_pred_emd_df['tp_fn'] == 'tp']
    tp_df = tp_df.reset_index(drop=True)
    fn_df = gt_vs_pred_emd_df[ gt_vs_pred_emd_df['tp_fn'] == 'fn']
    fn_df = fn_df.reset_index(drop=True)

    candidatePPI_p_val_lst = []
    # iterate over the tp_df and for each iteration, calculate candidate PPI p-value
    for index, row in tp_df.iterrows():
        print(f"\n ################# tp : starting {index}-th row out of {tp_df.shape[0]-1}\n")
        prot_1_id, prot_2_id = row['prot_1_id'], row['prot_2_id']
        # protein id has the format of [protein_name]_[chain_name]
        protein_name, chain_1_name = prot_1_id.split('_')
        protein_name, chain_2_name = prot_2_id.split('_')
        print(f"protein_name: {protein_name} :: chain_1_name: {chain_1_name} :: chain_2_name: {chain_2_name}")
        
        # retrieve the gt_contact_map
        print('retrieving the gt_contact_map...')
        gt_contact_map_location = os.path.join(gt_contact_map_dir, f"gt_contact_map_{protein_name}_{chain_1_name}_{chain_2_name}.pkl")
        gt_contact_map = joblib.load(gt_contact_map_location)

        # retrieve the pred_contact_map
        print('retrieving the pred_contact_map...')
        tp_pred_contact_map_path = os.path.join(tp_pred_contact_map_loc, f"pred_contact_map_{protein_name}_{chain_1_name}_{chain_2_name}*.pkl")
        pred_contact_map_lst = glob.glob(tp_pred_contact_map_path, recursive=False)
        pred_contact_map = joblib.load(pred_contact_map_lst[0])

        print(f'consider_full: {consider_full}')
        # retrieve the EMD between the pred_contact_map and gt_contact_map and name it as 'emd_observed'
        if(consider_full):
            # consider the full version of the interaction map
            print('considering the full version of the interaction map')
            # observed_emd = DockUtils.calculateEMD(pred_contact_map, gt_contact_map)
            observed_emd = row['emd_full']
        else:
            # consider the 99 percentile version of the interaction map
            print('considering the 99 percentile version of the interaction map')
            observed_emd = row['emd_9Xp']
        print(f'observed_emd = EMD between the pred_contact_map and gt_contact_map = {observed_emd}')
        
        # randomly shuffle pred_contact_map (a 2D numpy array) along a specified axis multiple times and
        # for each shuffling, calculate the EMD between shuffled version and gt_contact_map
        print('randomly shuffle pred_contact_map (a 2D numpy array) along a specified axis multiple times and\n \
                for each shuffling, calculate the EMD between shuffled version and gt_contact_map')
        emd_for_shuffled_lst = DockUtils.shuffle_predContactMap_and_calc_emd(pred_contact_map, gt_contact_map
                                                                             , axis=1, no_random_shuffle=no_random_shuffle, seed=456
                                                                             , consider_full=consider_full)
        # favourable case is counted when sampled_emd <= observed_emd
        no_of_favourable_cases = 0
        for sampled_emd in emd_for_shuffled_lst:
            if(sampled_emd <= observed_emd):
                no_of_favourable_cases += 1
        # end of for loop: for sampled_emd in emd_for_shuffled_lst:
        print(f'no_of_favourable_cases = {no_of_favourable_cases} :: no_random_shuffle = {no_random_shuffle}')
        # calculate indiv_candidatePPI_p_val
        print('calculating indiv_candidatePPI_p_val')
        indiv_candidatePPI_p_val = float(no_of_favourable_cases)/float(no_random_shuffle)
        # append indiv_candidatePPI_p_val at the candidatePPI_p_val_lst
        candidatePPI_p_val_lst.append(indiv_candidatePPI_p_val)
    # end of for loop: for index, row in tp_df.iterrows():

    # conditionally consider fn
    print(f'\n consider_fn: {consider_fn}')
    if(consider_fn):
        print('Also considering fn ...')
        # iterate over the fn_df and for each iteration, calculate candidate PPI p-value
        for index, row in fn_df.iterrows():
            print(f"\n ################# fn : starting {index}-th row out of {fn_df.shape[0]-1}\n")
            prot_1_id, prot_2_id = row['prot_1_id'], row['prot_2_id']
            # protein id has the format of [protein_name]_[chain_name]
            protein_name, chain_1_name = prot_1_id.split('_')
            protein_name, chain_2_name = prot_2_id.split('_')
            print(f"protein_name: {protein_name} :: chain_1_name: {chain_1_name} :: chain_2_name: {chain_2_name}")
            
            # retrieve the gt_contact_map
            print('retrieving the gt_contact_map...')
            gt_contact_map_location = os.path.join(gt_contact_map_dir, f"gt_contact_map_{protein_name}_{chain_1_name}_{chain_2_name}.pkl")
            gt_contact_map = joblib.load(gt_contact_map_location)

            # retrieve the pred_contact_map
            print('retrieving the pred_contact_map...')
            fn_pred_contact_map_path = os.path.join(fn_pred_contact_map_loc, f"pred_contact_map_{protein_name}_{chain_1_name}_{chain_2_name}*.pkl")
            pred_contact_map_lst = glob.glob(fn_pred_contact_map_path, recursive=False)
            pred_contact_map = joblib.load(pred_contact_map_lst[0])

            print(f'consider_full: {consider_full}')
            # retrieve the EMD between the pred_contact_map and gt_contact_map and name it as 'emd_observed'
            if(consider_full):
                # consider the full version of the interaction map
                print('considering the full version of the interaction map')
                # observed_emd = DockUtils.calculateEMD(pred_contact_map, gt_contact_map)
                observed_emd = row['emd_full']
            else:
                # consider the 99 percentile version of the interaction map
                print('considering the 99 percentile version of the interaction map')
                observed_emd = row['emd_9Xp']
            print(f'observed_emd = EMD between the pred_contact_map and gt_contact_map = {observed_emd}')
            
            # randomly shuffle pred_contact_map (a 2D numpy array) along a specified axis multiple times and
            # for each shuffling, calculate the EMD between shuffled version and gt_contact_map
            print('randomly shuffle pred_contact_map (a 2D numpy array) along a specified axis multiple times and\n \
                    for each shuffling, calculate the EMD between shuffled version and gt_contact_map')
            emd_for_shuffled_lst = DockUtils.shuffle_predContactMap_and_calc_emd(pred_contact_map, gt_contact_map
                                                                                 , axis=1, no_random_shuffle=no_random_shuffle, seed=456
                                                                                 , consider_full=consider_full)
            # favourable case is counted when sampled_emd <= observed_emd
            no_of_favourable_cases = 0
            for sampled_emd in emd_for_shuffled_lst:
                if(sampled_emd <= observed_emd):
                    no_of_favourable_cases += 1
            # end of for loop: for sampled_emd in emd_for_shuffled_lst:
            print(f'no_of_favourable_cases = {no_of_favourable_cases} :: no_random_shuffle = {no_random_shuffle}')
            # calculate indiv_candidatePPI_p_val
            print('calculating indiv_candidatePPI_p_val')
            indiv_candidatePPI_p_val = float(no_of_favourable_cases)/float(no_random_shuffle)
            # append indiv_candidatePPI_p_val at the candidatePPI_p_val_lst
            candidatePPI_p_val_lst.append(indiv_candidatePPI_p_val)
        # end of for loop: for index, row in fn_df.iterrows():
    # end of if block: if(consider_fn):
    
    # save candidatePPI_p_val_lst as a pkl file
    print('saving candidatePPI_p_val_lst as a pkl file')
    tTest_result_dir = os.path.join(xai_result_dir, 'tTest_result')
    PPIPUtils.createFolder(tTest_result_dir)
    candidatePPI_p_val_lst_loc = os.path.join(tTest_result_dir, f"candPPI_pVal_lst_considerFN_{consider_fn}_considerFull_{consider_full}.pkl")
    joblib.dump(value=candidatePPI_p_val_lst, filename=candidatePPI_p_val_lst_loc, compress=3)
    print('\n #############################\n inside calc_candidatePPI_p_val() method - End\n')


def perform_one_tailed_t_test(root_path='./', model_path='./', docking_version='4_0', consider_fn=False, consider_full=False):
    print('\n #############################\n inside perform_one_tailed_t_test() method - Start\n')
    print('\n########## docking_version: ' + str(docking_version))
    test_tag = model_path.split('/')[-1]
    xai_result_dir = os.path.join(root_path, f'dataset/proc_data_tl_feat_to_img/img_p2ip_cnn/xai_dock_{docking_version}/{test_tag}')
    tTest_result_dir = os.path.join(xai_result_dir, 'tTest_result')
    candidatePPI_p_val_lst_loc = os.path.join(tTest_result_dir, f"candPPI_pVal_lst_considerFN_{consider_fn}_considerFull_{consider_full}.pkl")
    # load candidatePPI_p_val_lst from pkl file
    candidatePPI_p_val_lst = joblib.load(candidatePPI_p_val_lst_loc)
    # perform one-sample one-tailed (less than) t-test
    result = stats.ttest_1samp(candidatePPI_p_val_lst, popmean=0.5, alternative='less')
    print(f'result:: t-statistic = {result.statistic} ; p-value = {result.pvalue}')
    # interpret the p-value
    print('interpreting the p-value')
    alpha = 0.05  # significance level
    print(f'alpha: {alpha}')
    p = result.pvalue
    conclusion = 'None'
    if p > alpha:
        print('Same proportions of errors (fail to reject H0)')
        conclusion = 'fail to reject H0'
    else:
        print('Different proportions of errors (reject H0)')
        conclusion = 'reject H0'
    # create a result dictionary and save it
    one_tailed_t_test_res_dict = {'consider_fn': consider_fn, 'consider_full': consider_full, 't-statistic': result.statistic, 'p-value': result.pvalue
                                  , 'alpha': alpha, 'conclusion': conclusion}
    one_tailed_t_test_res_dict_loc = os.path.join(tTest_result_dir, f"t_test_res_dict_considerFN_{consider_fn}_considerFull_{consider_full}.pkl")
    joblib.dump(value=one_tailed_t_test_res_dict, filename=one_tailed_t_test_res_dict_loc, compress=3)
    print('\n #############################\n inside perform_one_tailed_t_test() method - End\n')
    return one_tailed_t_test_res_dict


def generate_single_violin_plot(data_frame=None, save_plot_fl_nm_loc='./'):
    print('\n #############################\n inside generate_single_violin_plot() method - Start\n')
    # Create a figure and axis using seaborn
    fig, ax = plt.subplots(figsize=(8, 5))

    # Plot the violin plot using seaborn with different colors
    sns.violinplot(x='Category', y='Values', data=data_frame, ax=ax
                , palette=sns.color_palette("Paired"),  # print(sns.color_palette("pastel6").as_hex())
                inner='box', legend='full')

    # Set Y-axis range
    ax.set_ylim(0.0, 1.0)

    # Add a straight line along y=0.5
    ax.axhline(y=0.5, color='black', linestyle='--', label='y=0.5')
    # Add text to specify the value on the Y-axis
    ax.text(x=-0.70, y=0.49, s='0.5', color='black', fontsize=10)

    # Set plot labels and title
    ax.set_title("Violin Plot of p-values for 'Positive' and 'Negative'")
    ax.set_xlabel("")
    ax.set_ylabel("p-values")
    # ax.legend()

    # Save the plot as PNG
    plt.savefig(save_plot_fl_nm_loc, dpi=300, bbox_inches='tight')
    print('\n #############################\n inside generate_single_violin_plot() method - End\n')


def gen_violin_plt_of_pVal_for_manTl_manTlStruct(root_path='./', docking_version='4_0'):
    print('\n #############################\n inside gen_violin_plt_of_pVal_for_manTl_manTlStruct() method - Start\n')
    print('\n########## docking_version: ' + str(docking_version))
    consider_full = True  # consider full-version of the interaction maps for all plots
    common_plots_dir = os.path.join(root_path, f'dataset/proc_data_tl_feat_to_img/img_p2ip_cnn/xai_dock_{docking_version}/common_plots')
    PPIPUtils.createFolder(common_plots_dir)

    # generate violin plot for manTl features
    print('generating violin plot for manTl features')
    test_tag = 'ResNet_manTl_r400n18D'
    xai_result_dir = os.path.join(root_path, f'dataset/proc_data_tl_feat_to_img/img_p2ip_cnn/xai_dock_{docking_version}/{test_tag}')
    tTest_result_dir = os.path.join(xai_result_dir, 'tTest_result')
    # first fetch the p-value list for only true-positive (i.e. consider_fn=False) cases
    consider_fn=False
    candidatePPI_p_val_lst_fn_False_loc = os.path.join(tTest_result_dir, f"candPPI_pVal_lst_considerFN_{consider_fn}_considerFull_{consider_full}.pkl")
    # load candidatePPI_p_val_lst from pkl file
    candidatePPI_p_val_lst_fn_False = joblib.load(candidatePPI_p_val_lst_fn_False_loc)
    # next fetch the p-value list for both true-positive and false negative (i.e. consider_fn=True) cases
    consider_fn=True
    candidatePPI_p_val_lst_fn_True_loc = os.path.join(tTest_result_dir, f"candPPI_pVal_lst_considerFN_{consider_fn}_considerFull_{consider_full}.pkl")
    # load candidatePPI_p_val_lst from pkl file
    candidatePPI_p_val_lst_fn_True = joblib.load(candidatePPI_p_val_lst_fn_True_loc)
    # next derive p_val_lst_tp and p_val_lst_fn
    p_val_lst_tp = candidatePPI_p_val_lst_fn_False
    p_val_lst_fn = [p_val for p_val in candidatePPI_p_val_lst_fn_True if p_val not in p_val_lst_tp]
    # # create dataframe required for the violin plot generation
    # manTl_violin_plt_df = pd.DataFrame({'Category': ['Positive'] * len(p_val_lst_tp) + ['Negative'] * len(p_val_lst_fn)
    #                     , 'Values': p_val_lst_tp + p_val_lst_fn})
    # # specify location to save the violin plot
    # manTl_violin_plt_save_loc = os.path.join(common_plots_dir, f'violin_plt_manTl.png')
    # generate_single_violin_plot(data_frame=manTl_violin_plt_df, save_plot_fl_nm_loc=manTl_violin_plt_save_loc)

    # generate violin plot for manTlStruct features
    print('generating violin plot for manTlStruct features')
    test_tag = 'ResNet_manTlStruct_r400n18D'
    xai_result_dir = os.path.join(root_path, f'dataset/proc_data_tl_feat_to_img/img_p2ip_cnn/xai_dock_{docking_version}/{test_tag}')
    tTest_result_dir = os.path.join(xai_result_dir, 'tTest_result')
    # first fetch the p-value list for only true-positive (i.e. consider_fn=False) cases
    consider_fn=False
    candidatePPI_p_val_lst_fn_False_loc = os.path.join(tTest_result_dir, f"candPPI_pVal_lst_considerFN_{consider_fn}_considerFull_{consider_full}.pkl")
    # load candidatePPI_p_val_lst from pkl file
    candidatePPI_p_val_lst_fn_False = joblib.load(candidatePPI_p_val_lst_fn_False_loc)
    # next fetch the p-value list for both true-positive and false negative (i.e. consider_fn=True) cases
    consider_fn=True
    candidatePPI_p_val_lst_fn_True_loc = os.path.join(tTest_result_dir, f"candPPI_pVal_lst_considerFN_{consider_fn}_considerFull_{consider_full}.pkl")
    # load candidatePPI_p_val_lst from pkl file
    candidatePPI_p_val_lst_fn_True = joblib.load(candidatePPI_p_val_lst_fn_True_loc)
    # next derive p_val_lst_tp_struct and p_val_lst_fn_struct
    p_val_lst_tp_struct = candidatePPI_p_val_lst_fn_False
    p_val_lst_fn_struct = [p_val for p_val in candidatePPI_p_val_lst_fn_True if p_val not in p_val_lst_tp_struct]
    # # create dataframe required for the violin plot generation
    # manTlStruct_violin_plt_df = pd.DataFrame({'Category': ['Positive'] * len(p_val_lst_tp_struct) + ['Negative'] * len(p_val_lst_fn_struct)
    #                     , 'Values': p_val_lst_tp_struct + p_val_lst_fn_struct})
    # # specify location to save the violin plot
    # manTlStruct_violin_plt_save_loc = os.path.join(common_plots_dir, f'violin_plt_manTlStruct.png')
    # generate_single_violin_plot(data_frame=manTlStruct_violin_plt_df, save_plot_fl_nm_loc=manTlStruct_violin_plt_save_loc)

    # finally, generate consolidated violin plot for manTl and manTlStruct features together
    print('generating consolidated violin plot for manTl and manTlStruct features together')
    # create dataframe required for the violin plot generation
    manTl_manTlStruct_violin_plt_df = pd.DataFrame({'Category': ['Positive'] * len(p_val_lst_tp) + ['Positive(Struct)'] * len(p_val_lst_tp_struct)
                                         + ['Negative'] * len(p_val_lst_fn) + ['Negative(Struct)'] * len(p_val_lst_fn_struct)
                                        , 'Values': p_val_lst_tp + p_val_lst_tp_struct + p_val_lst_fn + p_val_lst_fn_struct})
    # specify location to save the violin plot
    manTl_manTlStruct_violin_plt_save_loc = os.path.join(common_plots_dir, f'violin_plt_pVal_manTl_manTlStruct.png')
    generate_single_violin_plot(data_frame=manTl_manTlStruct_violin_plt_df, save_plot_fl_nm_loc=manTl_manTlStruct_violin_plt_save_loc)
    print('\n #############################\n inside gen_violin_plt_of_pVal_for_manTl_manTlStruct() method - End\n')



if __name__ == '__main__':
    root_path = os.path.join('/scratch/pralaycs/Shubh_Working_Remote/PPI_Wkspc/PPI_Code/mat_p2ip_prj_working/')
    model_path = os.path.join(root_path, 'dataset/proc_data_tl_feat_to_img/img_p2ip_cnn/train/ResNet_manTl_r400n18D')
    
    no_random_shuffle = 500
    consider_full_lst = [True, False]  # True, False  # consider full-version or 99 percentile version of the interaction maps
    consider_fn_lst = [False, True]  # True, False  # include false-negative in the calculation
    docking_version_lst = ['4_0', '5_5']  # '4_0', '5_5'
    for docking_version in docking_version_lst:
        print('\n########## docking_version: ' + str(docking_version))
        one_tailed_t_test_res_dict_lst = [] 
        for consider_full in consider_full_lst:
            print('\n########## consider_full: ' + str(consider_full))
            for consider_fn in consider_fn_lst:
                print('\n########## consider_fn: ' + str(consider_fn))
                calc_candidatePPI_p_val(root_path=root_path, model_path=model_path, docking_version=docking_version
                                        , no_random_shuffle=no_random_shuffle, consider_fn=consider_fn, consider_full=consider_full)
                one_tailed_t_test_res_dict = perform_one_tailed_t_test(root_path=root_path, model_path=model_path, docking_version=docking_version
                                                                        , consider_fn=consider_fn, consider_full=consider_full)
                one_tailed_t_test_res_dict_lst.append(one_tailed_t_test_res_dict)
            # end of for loop
        # end of for loop: for consider_full in consider_full_lst:
        # create a pandas dataframe from one_tailed_t_test_res_dict_lst and save it
        one_tailed_t_test_res_df = pd.DataFrame(one_tailed_t_test_res_dict_lst)
        # save one_tailed_t_test_res_df
        test_tag = model_path.split('/')[-1]
        xai_result_dir = os.path.join(root_path, f'dataset/proc_data_tl_feat_to_img/img_p2ip_cnn/xai_dock_{docking_version}/{test_tag}')
        one_tailed_t_test_res_df.to_csv(os.path.join(xai_result_dir, 'tTest_result', 'one_tailed_t_test_res.csv'), index=False)
    # end of for loop: for docking_version in docking_version_lst:
    
    # ('################ generate violin plot of p-values for manTl and manTlStruct
    print('################ generating violin plot of p-values for manTl and manTlStruct features...')
    docking_version_lst = ['4_0', '5_5']  # '4_0', '5_5'
    for docking_version in docking_version_lst:
        print('\n########## docking_version: ' + str(docking_version))
        gen_violin_plt_of_pVal_for_manTl_manTlStruct(root_path=root_path, docking_version=docking_version)
