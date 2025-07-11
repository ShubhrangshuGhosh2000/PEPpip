import os, sys
from pathlib import Path
path_root = Path(__file__).parents[2]  
sys.path.insert(0, str(path_root))
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.calibration import CalibrationDisplay
from utils import PPIPUtils


def create_scaling_hybrid_score(spec_type = 'ecoli', img_pip_cnn_data_path = './', img_pip_trx_data_path = './', img_pip_hybrid_data_path = './', spec_prec_npv_dict=None):
    r = 'img_pip_cnn'; v = 'img_pip_trx'
    indiv_spec_prec_npv_dict = spec_prec_npv_dict[spec_type]
    prec_r = indiv_spec_prec_npv_dict[f'{r}_prec']; npv_r = indiv_spec_prec_npv_dict[f'{r}_npv']
    prec_v = indiv_spec_prec_npv_dict[f'{v}_prec']; npv_v = indiv_spec_prec_npv_dict[f'{v}_npv']
    prec_h = r if(prec_r >= prec_v) else v
    npv_h = r if(npv_r >= npv_v) else v
    S_prec_1 = prec_r / prec_v; S_prec_2 = prec_v / prec_r
    S_npv_1 = npv_r / npv_v; S_npv_2 = npv_v / npv_r
    spec_data_path = os.path.join(img_pip_hybrid_data_path, spec_type)
    PPIPUtils.makeDir(spec_data_path)
    hybrid_pred_prob_1_lst, hybrid_pred_label_lst = [], []
    img_pip_cnn_res_df = pd.read_csv(os.path.join(img_pip_cnn_data_path, spec_type, spec_type + '_pred_res.csv')
                                            , usecols = ['prot1_id', 'prot2_id', 'actual_res', 'pred_prob_1'])
    img_pip_cnn_res_df.rename(columns={'pred_prob_1': 'img_pip_cnn_pred_prob_1'}, inplace=True)
    img_pip_trx_res_df = pd.read_csv(os.path.join(img_pip_trx_data_path, spec_type, spec_type + '_pred_res.csv')
                                            , usecols = ['prot1_id', 'prot2_id', 'actual_res', 'pred_prob_1'])
    img_pip_trx_res_df.rename(columns={'actual_res': 'actual_res_trx', 'pred_prob_1': 'img_pip_trx_pred_prob_1'}, inplace=True)
    combined_df = pd.concat([img_pip_cnn_res_df, img_pip_trx_res_df], axis=1)
    for index, row in combined_df.iterrows():
        r_pred_prob_1= row[f'{r}_pred_prob_1']
        v_pred_prob_1 = row[f'{v}_pred_prob_1']
        p_r = r_pred_prob_1 
        p_v = v_pred_prob_1
        if(prec_h == r and npv_h == v):
            if(p_r >= 0.5 and p_v < 0.5):
                p_r = min(S_prec_1 * p_r, 1.0)  
                p_v = S_npv_1 * p_v  
            elif(p_r < 0.5 and p_v >= 0.5):
                p_r = min(S_npv_2 * p_r, 1.0)  
                p_v = S_prec_2 * p_v  
        elif(prec_h == v and npv_h == r):
            if(p_r >= 0.5 and p_v < 0.5):
                p_r = S_prec_1 * p_r  
                p_v = min(S_npv_1 * p_v, 1.0)  
            elif(p_r < 0.5 and p_v >= 0.5):
                p_r = S_npv_2 * p_r  
                p_v = min(S_prec_2 * p_v, 1.0)  
        elif(prec_h == r and npv_h == r):
            if(p_r >= 0.5 and p_v < 0.5):
                p_r = min(S_prec_1 * p_r, 1.0)  
                p_v = min(S_npv_1 * p_v, 1.0)  
            elif(p_r < 0.5 and p_v >= 0.5):
                p_r = S_npv_2 * p_r  
                p_v = S_prec_2 * p_v  
        elif(prec_h == v and npv_h == v):
            if(p_r >= 0.5 and p_v < 0.5):
                p_r = S_prec_1 * p_r  
                p_v = S_npv_1 * p_v 
            elif(p_r < 0.5 and p_v >= 0.5):
                p_r = min(S_npv_2 * p_r, 1.0)  
                p_v = min(S_prec_2 * p_v , 1.0) 
        p_f = p_r  
        if(p_r >= 0.5 and p_v >= 0.5):  
            p_f = p_r if(p_r >= p_v) else p_v
        elif(p_r < 0.5 and p_v < 0.5):  
            p_f = p_r if(p_r <= p_v) else p_v
        elif(p_r > 0.5 and p_v < 0.5):  
            exceedance = p_r - 0.5
            p_f = p_v + exceedance
        elif(p_r < 0.5 and p_v > 0.5):  
            exceedance = p_v - 0.5
            p_f = p_r + exceedance
        hybrid_pred_prob_1_lst.append(p_f)
        if(p_f >= 0.5):
            hybrid_pred_label_lst.append(1)
        else:
            hybrid_pred_label_lst.append(0)
    combined_df['hybrid_pred_prob_1'] = hybrid_pred_prob_1_lst
    combined_df['hybrid_pred_label'] = hybrid_pred_label_lst
    combined_df.to_csv(os.path.join(spec_data_path, 'combined_df_' + spec_type + '.csv'), index=False) 
    results = PPIPUtils.calcScores_DS(combined_df['actual_res'].to_numpy(), combined_df['hybrid_pred_prob_1'].to_numpy().reshape((-1, 1)))
    score_df = pd.DataFrame({'Species': [spec_type]
                            , 'AUPR': [results['AUPR']], 'Precision': [results['Precision']]
                            , 'Recall': [results['Recall']], 'AUROC': [results['AUROC']]
                            , 'NPV': [results['NPV']]
                            })
    score_df.to_csv(os.path.join(spec_data_path, 'hybrid_score_' + spec_type + '.csv'), index=False) 


def plot_prob_calibration_graph(spec_type = 'ecoli', img_pip_hybrid_data_path = './'):
    spec_data_path = os.path.join(img_pip_hybrid_data_path, spec_type)
    test_res_file_nm_with_loc = os.path.join(spec_data_path, 'combined_df_' + spec_type + '.csv')
    pred_result_df = pd.read_csv(test_res_file_nm_with_loc)
    y_true = pred_result_df['actual_res'].to_numpy()
    y_prob = pred_result_df['hybrid_pred_prob_1'].to_numpy()
    calib_disp = CalibrationDisplay.from_predictions(y_true, y_prob, name=f'PIC')
    calib_disp.plot()
    calib_graph_nm = f'{spec_type}_prob_calib_graph_pic.png'
    plt.savefig(os.path.join(spec_data_path, calib_graph_nm), dpi=300)


if __name__ == '__main__':
    root_path = os.path.join('/project/root/directory/path/here')
    
    img_pip_cnn_test_tag = 'ResNet_tlStructEsmc_r400n18DnoWS_epoch37_noWS'
    img_pip_trx_test_tag = 'CDAMViT_tlStructEsmc_r400p16_lossBasedChkpt'
    img_pip_cnn_data_path = os.path.join(root_path, f'dataset/proc_data_tl_feat_to_img/img_p2ip_cnn/test/dscript_other_spec_{img_pip_cnn_test_tag}')
    img_pip_trx_data_path = os.path.join(root_path, f'dataset/proc_data_tl_feat_to_img/img_p2ip_trx/test/dscript_other_spec_{img_pip_trx_test_tag}')
    img_pip_hybrid_data_path = os.path.join(root_path, f'dataset/proc_data_tl_feat_to_img/img_pip_hybrid/img_pip_cnn_trx_scaling_hybrid')
    PPIPUtils.createFolder(img_pip_hybrid_data_path, recreate_if_exists=False)
    spec_prec_npv_dict = {}
    spec_prec_npv_dict['ecoli'] = {'img_pip_cnn_prec': 0.7, 'img_pip_cnn_npv': 0.94, 'img_pip_trx_prec': 0.73, 'img_pip_trx_npv': 0.93}
    spec_prec_npv_dict['fly'] = {'img_pip_cnn_prec': 0.9, 'img_pip_cnn_npv': 0.94, 'img_pip_trx_prec': 0.87, 'img_pip_trx_npv': 0.93}
    spec_prec_npv_dict['mouse'] = {'img_pip_cnn_prec': 0.86, 'img_pip_cnn_npv': 0.96, 'img_pip_trx_prec': 0.85, 'img_pip_trx_npv': 0.94}
    spec_prec_npv_dict['worm'] = {'img_pip_cnn_prec': 0.9, 'img_pip_cnn_npv': 0.95, 'img_pip_trx_prec': 0.87, 'img_pip_trx_npv': 0.94}
    spec_prec_npv_dict['yeast'] = {'img_pip_cnn_prec': 0.79, 'img_pip_cnn_npv': 0.95, 'img_pip_trx_prec': 0.79, 'img_pip_trx_npv': 0.94}
    spec_prec_npv_dict['human'] = {'img_pip_cnn_prec': 0.88, 'img_pip_cnn_npv': 0.96, 'img_pip_trx_prec': 0.87, 'img_pip_trx_npv': 0.95}
    spec_type_lst = ['ecoli', 'fly', 'mouse', 'worm', 'yeast', 'human']
    for spec_type in spec_type_lst:
        create_scaling_hybrid_score(spec_type=spec_type, img_pip_cnn_data_path=img_pip_cnn_data_path, img_pip_trx_data_path=img_pip_trx_data_path
                                    , img_pip_hybrid_data_path=img_pip_hybrid_data_path, spec_prec_npv_dict=spec_prec_npv_dict)
