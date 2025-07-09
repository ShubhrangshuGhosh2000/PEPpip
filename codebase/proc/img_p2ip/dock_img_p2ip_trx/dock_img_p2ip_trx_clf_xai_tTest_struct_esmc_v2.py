import os
import sys
from pathlib import Path
path_root = Path(__file__).parents[3]  
sys.path.insert(0, str(path_root))
from utils import dl_reproducible_result_util
import glob
import joblib
import pandas as pd
import scipy.stats as stats
from proc.img_p2ip.img_p2ip_trx.img_p2ip_trx_clf_train_struct_esmc_v2 import ImgP2ipTrx
from utils import DockUtils
from utils import PPIPUtils


def load_final_ckpt_model(root_path='./', model_path='./', partial_model_name = 'ImgP2ipTrx'):
    final_chkpt_path = os.path.join(model_path, partial_model_name + '*.ckpt' )
    final_ckpt_file_name = glob.glob(final_chkpt_path, recursive=False)[0]
    model = ImgP2ipTrx.load_from_checkpoint(final_ckpt_file_name)
    return model


def calc_candidatePPI_p_val(root_path='./', model_path='./', docking_version='5_5', attn_mode='total', min_max_mode='R', no_random_shuffle=500
                            , consider_fn=False, consider_full=False, pct=99.0, pcmp_mode='SCGB'):
    gt_contact_map_dir = os.path.join(root_path, f"dataset/preproc_data_docking_BM_{docking_version}/prot_gt_contact_map")
    test_tag = model_path.split('/')[-1]
    xai_result_dir = os.path.join(root_path, f'dataset/proc_data_tl_feat_to_img/img_p2ip_trx/xai_dock_{docking_version}/{test_tag}')
    tp_pred_contact_map_proc_loc = os.path.join(xai_result_dir, 'tp_pred_contact_map_proc', f'attnMode_{attn_mode}', f'minMaxMode_{min_max_mode}', f'pcmp_mode_{pcmp_mode}')
    fn_pred_contact_map_proc_loc = os.path.join(xai_result_dir, 'fn_pred_contact_map_proc', f'attnMode_{attn_mode}', f'minMaxMode_{min_max_mode}', f'pcmp_mode_{pcmp_mode}')
    emd_result_dir = os.path.join(xai_result_dir, 'emd_result')
    model = load_final_ckpt_model(root_path=root_path, model_path=model_path, partial_model_name = 'ImgP2ipTrx')
    patch_size = model.hparams.config['patch_size']  
    gt_vs_pred_emd_df = None
    if(consider_full):
        gt_vs_pred_emd_df = pd.read_csv(os.path.join(emd_result_dir, f'gt_vs_pred_emd_full.csv'))
    else:
        gt_vs_pred_emd_df = pd.read_csv(os.path.join(emd_result_dir, f'gt_vs_pred_emd_9Xp.csv'))
    tp_df = gt_vs_pred_emd_df[ (gt_vs_pred_emd_df['tp_fn'] == 'tp') & (gt_vs_pred_emd_df['attn_mode'] == attn_mode) & (gt_vs_pred_emd_df['min_max_mode'] == min_max_mode) & (gt_vs_pred_emd_df['pcmp_mode'] == pcmp_mode)]
    tp_df = tp_df.reset_index(drop=True)
    fn_df = gt_vs_pred_emd_df[ (gt_vs_pred_emd_df['tp_fn'] == 'fn') & (gt_vs_pred_emd_df['attn_mode'] == attn_mode) & (gt_vs_pred_emd_df['min_max_mode'] == min_max_mode) & (gt_vs_pred_emd_df['pcmp_mode'] == pcmp_mode)]
    fn_df = fn_df.reset_index(drop=True)
    candidatePPI_p_val_lst = []
    for index, row in tp_df.iterrows():
        prot_1_id, prot_2_id = row['prot_1_id'], row['prot_2_id']
        protein_name, chain_1_name = prot_1_id.split('_')
        protein_name, chain_2_name = prot_2_id.split('_')
        gt_contact_map_location = os.path.join(gt_contact_map_dir, f"contact_map_{protein_name}_{chain_1_name}_{chain_2_name}.pkl")
        gt_contact_map = joblib.load(gt_contact_map_location)
        tp_pred_contact_map_path = os.path.join(tp_pred_contact_map_proc_loc, f"pred_contact_map_{protein_name}_{chain_1_name}_{chain_2_name}_attnMode_{attn_mode}_minMaxMode_{min_max_mode}_pcmp_mode_{pcmp_mode}.pkl")
        pred_contact_map_lst = glob.glob(tp_pred_contact_map_path, recursive=False)
        pred_contact_map = joblib.load(pred_contact_map_lst[0])
        observed_emd = row['emd']
        emd_for_shuffled_lst = DockUtils.shuffle_attention_map_and_calc_emd(pred_contact_map, gt_contact_map
                                                                            , window_size=patch_size  
                                                                            , axis=1, no_random_shuffle=no_random_shuffle, seed=456
                                                                            , consider_full=consider_full, pct=pct)
        no_of_favourable_cases = 0
        for sampled_emd in emd_for_shuffled_lst:
            if(sampled_emd <= observed_emd):
                no_of_favourable_cases += 1
        indiv_candidatePPI_p_val = float(no_of_favourable_cases)/float(no_random_shuffle)
        candidatePPI_p_val_lst.append(indiv_candidatePPI_p_val)
    if(consider_fn):
        for index, row in fn_df.iterrows():
            prot_1_id, prot_2_id = row['prot_1_id'], row['prot_2_id']
            protein_name, chain_1_name = prot_1_id.split('_')
            protein_name, chain_2_name = prot_2_id.split('_')
            gt_contact_map_location = os.path.join(gt_contact_map_dir, f"contact_map_{protein_name}_{chain_1_name}_{chain_2_name}.pkl")
            gt_contact_map = joblib.load(gt_contact_map_location)
            fn_pred_contact_map_path = os.path.join(fn_pred_contact_map_proc_loc, f"pred_contact_map_{protein_name}_{chain_1_name}_{chain_2_name}_attnMode_{attn_mode}_minMaxMode_{min_max_mode}_pcmp_mode_{pcmp_mode}.pkl")
            pred_contact_map_lst = glob.glob(fn_pred_contact_map_path, recursive=False)
            pred_contact_map = joblib.load(pred_contact_map_lst[0])
            observed_emd = row['emd']
            emd_for_shuffled_lst = DockUtils.shuffle_attention_map_and_calc_emd(pred_contact_map, gt_contact_map, axis=1, no_random_shuffle=no_random_shuffle, seed=456
                                                                                 , consider_full=consider_full, pct=pct)
            no_of_favourable_cases = 0
            for sampled_emd in emd_for_shuffled_lst:
                if(sampled_emd <= observed_emd):
                    no_of_favourable_cases += 1
            indiv_candidatePPI_p_val = float(no_of_favourable_cases)/float(no_random_shuffle)
            candidatePPI_p_val_lst.append(indiv_candidatePPI_p_val)
    tTest_result_dir = os.path.join(xai_result_dir, 'tTest_result')
    PPIPUtils.createFolder(tTest_result_dir)
    candidatePPI_p_val_lst_loc = os.path.join(tTest_result_dir, f"candPPI_pVal_lst_considerFN_{consider_fn}_considerFull_{consider_full}_attnMode_{attn_mode}_minMaxMode_{min_max_mode}_pcmp_mode_{pcmp_mode}.pkl")
    joblib.dump(value=candidatePPI_p_val_lst, filename=candidatePPI_p_val_lst_loc, compress=3)


def perform_one_tailed_t_test(root_path='./', model_path='./', docking_version='5_5', consider_fn=False, consider_full=False, attn_mode='total', min_max_mode='R', pcmp_mode='SCGB'):
    test_tag = model_path.split('/')[-1]
    xai_result_dir = os.path.join(root_path, f'dataset/proc_data_tl_feat_to_img/img_p2ip_trx/xai_dock_{docking_version}/{test_tag}')
    tTest_result_dir = os.path.join(xai_result_dir, 'tTest_result')
    candidatePPI_p_val_lst_loc = os.path.join(tTest_result_dir, f"candPPI_pVal_lst_considerFN_{consider_fn}_considerFull_{consider_full}_attnMode_{attn_mode}_minMaxMode_{min_max_mode}_pcmp_mode_{pcmp_mode}.pkl")
    candidatePPI_p_val_lst = joblib.load(candidatePPI_p_val_lst_loc)
    result = stats.ttest_1samp(candidatePPI_p_val_lst, popmean=0.5, alternative='less')
    alpha = 0.05  
    p = result.pvalue
    conclusion = 'None'
    if p > alpha:
        conclusion = 'fail to reject H0'
    else:
        conclusion = 'reject H0'
    one_tailed_t_test_res_dict = {'consider_fn': consider_fn, 'consider_full': consider_full, 'attrMode': attn_mode, 'minMaxMode': min_max_mode, 'pcmp_mode': pcmp_mode
                                  , 't-statistic': result.statistic, 'p-value': result.pvalue, 'alpha': alpha, 'conclusion': conclusion}
    one_tailed_t_test_res_dict_loc = os.path.join(tTest_result_dir, f"t_test_res_dict_considerFN_{consider_fn}_considerFull_{consider_full}_attnMode_{attn_mode}_minMaxMode_{min_max_mode}_pcmp_mode_{pcmp_mode}.pkl")
    joblib.dump(value=one_tailed_t_test_res_dict, filename=one_tailed_t_test_res_dict_loc, compress=3)
    return one_tailed_t_test_res_dict


if __name__ == '__main__':
    root_path = os.path.join('/project/root/directory/path/here')
    model_path = os.path.join(root_path, 'dataset/proc_data_tl_feat_to_img/img_p2ip_trx/train/CDAMViT_tlStructEsmc_r400p16')
    no_random_shuffle = 500  
    consider_full_lst = [True]  
    consider_fn_lst = [False, True]  
    docking_version_lst = ['5_5']  
    attn_mode_lst = ['total']  
    min_max_mode_lst = ['N']  
    pcmp_mode_lst = ['SCGB']  
    pct = 99.0  
    for docking_version in docking_version_lst:
        one_tailed_t_test_res_dict_lst = [] 
        for consider_full in consider_full_lst:
            for consider_fn in consider_fn_lst:
                for attn_mode in attn_mode_lst:
                    for min_max_mode in min_max_mode_lst:
                        for pcmp_mode in pcmp_mode_lst:
                            calc_candidatePPI_p_val(root_path=root_path, model_path=model_path, docking_version=docking_version, attn_mode=attn_mode, min_max_mode=min_max_mode
                                                , no_random_shuffle=no_random_shuffle, consider_fn=consider_fn, consider_full=consider_full, pct=pct, pcmp_mode=pcmp_mode)
                            one_tailed_t_test_res_dict = perform_one_tailed_t_test(root_path=root_path, model_path=model_path, docking_version=docking_version
                                                                               , attn_mode=attn_mode, min_max_mode=min_max_mode, consider_fn=consider_fn, consider_full=consider_full, pcmp_mode=pcmp_mode)
                            one_tailed_t_test_res_dict_lst.append(one_tailed_t_test_res_dict)
        one_tailed_t_test_res_df = pd.DataFrame(one_tailed_t_test_res_dict_lst)
        test_tag = model_path.split('/')[-1]
        xai_result_dir = os.path.join(root_path, f'dataset/proc_data_tl_feat_to_img/img_p2ip_trx/xai_dock_{docking_version}/{test_tag}')
        one_tailed_t_test_res_df.to_csv(os.path.join(xai_result_dir, 'tTest_result', 'one_tailed_t_test_res.csv'), index=False)
