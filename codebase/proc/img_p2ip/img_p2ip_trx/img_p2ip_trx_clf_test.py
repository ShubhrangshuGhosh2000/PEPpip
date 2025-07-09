import os
import sys
from pathlib import Path
path_root = Path(__file__).parents[3]  
sys.path.insert(0, str(path_root))
from utils import dl_reproducible_result_util
import torch
import torch.nn.functional as F
import lightning as L
import gc
import glob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibrationDisplay
from codebase.proc.img_p2ip.img_p2ip_trx_datamodule import ImgP2ipCustomDataModule
from proc.img_p2ip.img_p2ip_trx.img_p2ip_trx_clf_train import ImgP2ipTrx
from utils import PPIPUtils


def load_final_ckpt_model(root_path='./', model_path='./', partial_model_name = 'ImgP2ipTrx'):
    final_chkpt_path = os.path.join(model_path, partial_model_name + '*.ckpt' )
    final_ckpt_file_name = glob.glob(final_chkpt_path, recursive=False)[0]
    model = ImgP2ipTrx.load_from_checkpoint(final_ckpt_file_name, map_location='cuda:0')
    model.test_step_outputs.clear()  
    return model


def prepare_test_data(root_path='./', model=None, spec_type='ecoli'):
    test_data_module = None
    test_data_module = ImgP2ipCustomDataModule(root_path=root_path
                                               , batch_size=1  
                                               , workers=os.cpu_count() - 2  
                                               , img_resoln=model.hparams.config['img_resoln'], spec_type=spec_type
                                               , dbl_combi_flg=model.hparams.config['dbl_combi_flg'])
    return test_data_module


def test_model(root_path='./', model_path='./', partial_model_name = 'ImgP2ipTrx', spec_type='ecoli'):
    model = load_final_ckpt_model(root_path, model_path, partial_model_name)
    test_data_module = prepare_test_data(root_path, model, spec_type)
    trainer = L.Trainer(deterministic=True
                    , logger = False
                    , callbacks=[]  
                    , accelerator="gpu", devices=1, num_nodes=1  
                    , precision = '16-mixed'
                    , enable_progress_bar = True
                    , enable_model_summary = False
                    , inference_mode = False  
                    )
    trainer.test(model, test_data_module)
    pred_logits_2d_tensor_lst, pred_attn_2d_tensor_lst, test_label_1d_tensor_lst = [], [], []
    for itr, indiv_test_step_outputs in enumerate(model.test_step_outputs):
        pred_logits_2d_tensor_lst.append(indiv_test_step_outputs['logits'])
        pred_attn_2d_tensor_lst.append(indiv_test_step_outputs['attn'])
        test_label_1d_tensor_lst.append(indiv_test_step_outputs['y'])
    con_pred_logits_2d_tensor = torch.cat(pred_logits_2d_tensor_lst, dim=0)
    con_pred_attn_2d_tensor = torch.cat(pred_attn_2d_tensor_lst, dim=0)
    con_test_label_1d_tensor = torch.cat(test_label_1d_tensor_lst, dim=0)
    con_pred_prob_2d_tensor = F.softmax(con_pred_logits_2d_tensor, dim=1)
    tot_no_predictions = con_pred_prob_2d_tensor.shape[0]
    pred_prob_1_lst, pred_lst, pred_prob_arr_lst = [], [], []
    for pred_idx in range(tot_no_predictions):
        out =  con_pred_prob_2d_tensor[pred_idx,:]
        prediction = torch.argmax(out)
        pred_lst.append(prediction.item())
        full_pred_prob_arr = out.numpy()  
        pred_prob_1_lst.append(full_pred_prob_arr[1])
        pred_prob_arr_lst.append(full_pred_prob_arr)
    pred_prob_dict = {}
    for i in range(pred_prob_arr_lst[0].size):
        pred_prob_dict[i] = []
    for ind in range(len(pred_prob_arr_lst)):
        full_pred_prob_arr = pred_prob_arr_lst[ind]
        for i in range(full_pred_prob_arr.size):
            pred_prob_dict[i].append(full_pred_prob_arr[i])
    y_test_lst = con_test_label_1d_tensor.cpu().numpy().tolist()
    pred_result_df = pd.DataFrame({ 'actual_res': y_test_lst, 'pred_res': pred_lst})
    for key in pred_prob_dict.keys():
        pred_result_df['pred_prob_' + str(key)] = pred_prob_dict[key]
    test_pair_list_path = os.path.join(root_path, 'dataset/orig_data_DS/pairs', spec_type + '_test.tsv' )
    spec_test_pairs_df = pd.read_csv(test_pair_list_path, delimiter='\t', header=None, names=['prot1_id', 'prot2_id', 'label'])
    pred_result_df.insert(0, 'prot2_id', spec_test_pairs_df['prot2_id'])
    pred_result_df.insert(0, 'prot1_id', spec_test_pairs_df['prot1_id'])
    test_tag = model_path.split('/')[-1]
    test_result_dir = os.path.join(root_path, 'dataset/proc_data_tl_feat_to_img/img_p2ip_trx/test/dscript_other_spec_' + test_tag, spec_type)
    try:
        if not os.path.exists(test_result_dir):
            os.makedirs(test_result_dir)
    except OSError as ex:
        errorMessage = "Creation of the directory " + test_result_dir + " failed. Exception is: " + str(ex)
        raise Exception(errorMessage)
    else:
        print("Successfully created the directory %s " % test_result_dir)
    test_res_file_nm_with_loc = os.path.join(test_result_dir, spec_type + '_pred_res.csv')
    pred_result_df.to_csv(test_res_file_nm_with_loc, index=False)
    score_dict = PPIPUtils.calcScores_DS(np.array(y_test_lst), np.array(pred_prob_1_lst))
    score_df = pd.DataFrame({'Species': [spec_type]
                    , 'AUPR': [score_dict['AUPR']], 'Precision': [score_dict['Precision']]
                    , 'Recall': [score_dict['Recall']], 'AUROC': [score_dict['AUROC']]
                    , 'NPV': [score_dict['NPV']]
                    })
    score_df_name_with_loc = os.path.join(test_result_dir, spec_type + '_pred_score.csv')
    score_df.to_csv(score_df_name_with_loc, index=False)
    del model  
    gc.collect()  
    torch.cuda.empty_cache()  


def start(root_path='./', model_path='./', partial_model_name = 'ImgP2ipTrx', spec_type='ecoli'):
    test_model(root_path=root_path, model_path=model_path, partial_model_name=partial_model_name, spec_type=spec_type)


def plot_prob_calibration_graph(root_path='./', model_path='./', spec_type='ecoli'):
    test_tag = model_path.split('/')[-1]
    test_result_dir = os.path.join(root_path, 'dataset/proc_data_tl_feat_to_img/img_p2ip_trx/test/dscript_other_spec_' + test_tag, spec_type)
    test_res_file_nm_with_loc = os.path.join(test_result_dir, spec_type + '_pred_res.csv')
    pred_result_df = pd.read_csv(test_res_file_nm_with_loc)
    y_true = pred_result_df['actual_res'].to_numpy()
    y_prob = pred_result_df['pred_prob_1'].to_numpy()
    calib_disp = CalibrationDisplay.from_predictions(y_true, y_prob, name=f'VC')
    calib_disp.plot()
    calib_graph_nm = f'{spec_type}_prob_calib_graph_vc.png'
    plt.savefig(os.path.join(test_result_dir, calib_graph_nm), dpi=300)


if __name__ == '__main__':
    root_path = os.path.join('/project/root/directory/path/here')
    model_path = os.path.join(root_path, 'dataset/proc_data_tl_feat_to_img/img_p2ip_trx/train/CDAMViT_tlStructEsmc_r400p16_lossBasedChkpt')
    partial_model_name = 'ImgP2ipTrx'
    spec_type_lst = ['human', 'ecoli', 'fly', 'mouse', 'worm', 'yeast']  
    for spec_type in spec_type_lst:
        start(root_path=root_path, model_path=model_path, partial_model_name=partial_model_name, spec_type=spec_type)
