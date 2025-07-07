import os
import sys
from pathlib import Path

path_root = Path(__file__).parents[3]  # upto 'codebase' folder
sys.path.insert(0, str(path_root))
# print(sys.path)

from utils import dl_reproducible_result_util
import torch
import torch.nn.functional as F
import lightning as L
import glob
import numpy as np
import pandas as pd

from proc.img_p2ip.dock_img_p2ip_trx_datamodule_struct_esmc_v2 import DockImgP2ipCustomDataModule
from proc.img_p2ip.img_p2ip_trx.img_p2ip_trx_clf_train_struct_esmc_v2 import ImgP2ipTrx
from utils import PPIPUtils


def load_final_ckpt_model(root_path='./', model_path='./', partial_model_name = 'ImgP2ipTrx'):
    print('#### inside the load_final_ckpt_model() method - Start')
    # create the final checkpoint file name with path
    final_chkpt_path = os.path.join(model_path, partial_model_name + '*.ckpt' )
    final_ckpt_file_name = glob.glob(final_chkpt_path, recursive=False)[0]
    
    ################################# TEMPORARY -START #################################
    # final_ckpt_file_name = os.path.join(model_path, 'ImgP2ipTrx-epoch03-val_aupr0.82-last.ckpt')
    ################################# TEMPORARY -END #################################

    print('final_ckpt_file_name: ' + str(final_ckpt_file_name))
    # load the model
    model = ImgP2ipTrx.load_from_checkpoint(final_ckpt_file_name)
    # # automatically restores model, epoch, step, LR schedulers, apex, etc...
    # model = ImgP2ipTrx()
    # trainer = pl.Trainer()
    # trainer.fit(model, ckpt_path=final_ckpt_file_name)
   
    # Please note that, all model hyper-params like batch_size, block_name, etc. can be retrieved using model.hparams.config
    model.test_step_outputs.clear()  # reset the list

    print('#### inside the load_final_ckpt_model() method - End')
    return model


def prepare_test_data(root_path='./', model=None, docking_version='5_5'):
    print('#### inside the prepare_test_data() method - Start')
    test_data_module = DockImgP2ipCustomDataModule(root_path=root_path, batch_size=model.hparams.config['batch_size']
                                               , workers= 2  # os.cpu_count() - 5  # model.hparams.config['num_workers']
                                               , img_resoln=model.hparams.config['img_resoln']
                                               , spec_type='human'
                                               , docking_version=docking_version
                                               , dbl_combi_flg=model.hparams.config['dbl_combi_flg'])
    print('#### inside the prepare_test_data() method - End')
    return test_data_module


def test_model(root_path='./', model_path='./', partial_model_name = 'ImgP2ipTrx', docking_version='5_5'):
    print('\n #############################\n inside the test_model() method - Start\n')
    print('# #############################\n')
    print('\n########## docking_version: ' + str(docking_version))
    # load the final checkpointed model
    print('loading the final checkpointed model')
    model = load_final_ckpt_model(root_path, model_path, partial_model_name)
    # prepare the test data
    print('\n preparing the test data')
    test_data_module = prepare_test_data(root_path, model, docking_version)
    # perform the prediction
    print('\n performing the prediction')
    # perform testing using pytorch lightning Trainer 
    trainer = L.Trainer(deterministic=True
                    , logger = False
                    , callbacks=[]  # can also use checkpoint_callback, early_stop_callback
                    , accelerator="gpu", devices=1, num_nodes=1  # for single-gpu, single-node training
                    , precision = '16-mixed'
                    , enable_progress_bar = True
                    , enable_model_summary = False
                    , inference_mode = False  # To enable gradient tracking for CDAM generation during evaluation 
                    )
    trainer.test(model, test_data_module)
    # processing the prediction
    print('\n processing the prediction')
    pred_logits_2d_tensor_lst, test_label_1d_tensor_lst = [], []
    # pred_attn_2d_tensor_lst = []
    for itr, indiv_test_step_outputs in enumerate(model.test_step_outputs):
        # if(itr % 500 == 0): print('#### completed ' + str(itr) + ', out of ' + str(len(model.test_step_outputs)))
        pred_logits_2d_tensor_lst.append(indiv_test_step_outputs['logits'])
        # pred_attn_2d_tensor_lst.append(indiv_test_step_outputs['attn'])
        test_label_1d_tensor_lst.append(indiv_test_step_outputs['y'])
    # end of for loop
    # create consolidated prediction result 2d tensor, predicted attn map 2d tensor and the respective test label 1d tensor
    con_pred_logits_2d_tensor = torch.cat(pred_logits_2d_tensor_lst, dim=0)
    # con_pred_attn_2d_tensor = torch.cat(pred_attn_2d_tensor_lst, dim=0)
    con_test_label_1d_tensor = torch.cat(test_label_1d_tensor_lst, dim=0)
    print('con_pred_logits_2d_tensor.shape: ' + str(con_pred_logits_2d_tensor.shape))
    # print('con_pred_attn_2d_tensor.shape: ' + str(con_pred_attn_2d_tensor.shape))
    print('con_test_label_1d_tensor.shape: ' + str(con_test_label_1d_tensor.shape))
    # convert logits into probabilities
    con_pred_prob_2d_tensor = F.softmax(con_pred_logits_2d_tensor, dim=1)

    tot_no_predictions = con_pred_prob_2d_tensor.shape[0]
    pred_prob_1_lst, pred_lst, pred_prob_arr_lst = [], [], []
    for pred_idx in range(tot_no_predictions):
        
        # ##################### @@@@ TO BE CHECKED BEFORE RELEASING THE CODE @@@ #################### #
        # ######################################### #
        out =  1 - con_pred_prob_2d_tensor[pred_idx,:]  # probabilities that the predicted label will be 0 or 1.
        # ######################################### #
        # ##################### @@@@ TO BE CHECKED BEFORE RELEASING THE CODE @@@ #################### #
        
        prediction = torch.argmax(out)
        pred_lst.append(prediction.item())
        # full_pred_prob_arr is a 1d array containing both the probabilities that the predicted label will 
        # be 0 or 1.
        full_pred_prob_arr = out.cpu().numpy()
        pred_prob_1_lst.append(full_pred_prob_arr[1])
        pred_prob_arr_lst.append(full_pred_prob_arr)
        if(pred_idx % 50 == 0):
            print('#### Prediction processing completed for ' + str(pred_idx+1) + ' test sample, out of ' + str(tot_no_predictions))
    # end of for loop: for pred_idx in range(tot_no_predictions):
    # create a df to store the prediction result
    print('\n creating a df to store the prediction result')
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
    
    # retrieve the original test pair list for the given species
    test_pair_list_path = os.path.join(root_path, 'dataset/preproc_data_docking_BM_' + str(docking_version), 'derived_feat/dock',  'dock_test.tsv')
    spec_test_pairs_df = pd.read_csv(test_pair_list_path, delimiter='\t', header=None, names=['prot1_id', 'prot2_id', 'label'])
    # prepend prot1_id and prot2_id columns in pred_result_df
    pred_result_df.insert(0, 'prot2_id', spec_test_pairs_df['prot2_id'])
    pred_result_df.insert(0, 'prot1_id', spec_test_pairs_df['prot1_id'])
    
    # save the pred_result_df
    test_tag = model_path.split('/')[-1]
    test_result_dir = os.path.join(root_path, f'dataset/proc_data_tl_feat_to_img/img_p2ip_trx/test_dock_{docking_version}/{test_tag}')
    try:
        # check if the test_result_dir already exists and if not, then create it
        if not os.path.exists(test_result_dir):
            print("The directory: " + str(test_result_dir) + " does not exist.. Creating it...")
            os.makedirs(test_result_dir)
    except OSError as ex:
        errorMessage = "Creation of the directory " + test_result_dir + " failed. Exception is: " + str(ex)
        raise Exception(errorMessage)
    else:
        print("Successfully created the directory %s " % test_result_dir)
    test_res_file_nm_with_loc = os.path.join(test_result_dir, 'pred_res.csv')
    pred_result_df.to_csv(test_res_file_nm_with_loc, index=False)
    print('pred_result_df is saved as : ' + str(test_res_file_nm_with_loc))

    # ## SCORE CALCULATION IS IRRELEVANT IN CASE OF DOCKING RELATED interaction map PREDICTION
    # # compute result metrics, such as AUPR, Precision, Recall, AUROC
    # score_dict = PPIPUtils.calcScores_DS(np.array(y_test_lst), np.array(pred_prob_1_lst))
    # # create score_df
    # score_df = pd.DataFrame({'Docking_BM': [docking_version]
    #                 , 'AUPR': [score_dict['AUPR']], 'Precision': [score_dict['Precision']]
    #                 , 'Recall': [score_dict['Recall']], 'AUROC': [score_dict['AUROC']]
    #                 })
    # # save the score_df
    # score_df_name_with_loc = os.path.join(test_result_dir, 'pred_score.csv')
    # score_df.to_csv(score_df_name_with_loc, index=False)
    # print('score_df is saved as : ' + str(score_df_name_with_loc))
    
    # ## ATTENTION CALCULATION AND SUBSEQUENT STORING IS NOT DONE HERE AS ATTENTION CALCULATION WOULD BE DONE SEPARATELY LATER
    # # Save raw attention maps for postprocessing
    # xai_result_dir = os.path.join(root_path, f'dataset/proc_data_tl_feat_to_img/img_p2ip_trx/xai_dock_{docking_version}/{test_tag}')
    # attn_maps_dir = os.path.join(xai_result_dir, 'attn_maps')
    # PPIPUtils.createFolder(attn_maps_dir, recreate_if_exists=False)
    # torch.save(con_pred_attn_2d_tensor, os.path.join(attn_maps_dir, f'attn_maps_dock_{docking_version}.pt'))
    print('#### inside the test_model() method - End')


def start(root_path='./', model_path='./', partial_model_name = 'ImgP2ipTrx', docking_version='5_5'):
    test_model(root_path=root_path, model_path=model_path, partial_model_name=partial_model_name, docking_version=docking_version)



if __name__ == '__main__':
    root_path = os.path.join('/scratch/pralaycs/Shubh_Working_Remote/PPI_Wkspc/PPI_Code/mat_p2ip_prj_working/')
    model_path = os.path.join(root_path, 'dataset/proc_data_tl_feat_to_img/img_p2ip_trx/train/CDAMViT_tlStructEsmc_r400p16')
    partial_model_name = 'ImgP2ipTrx'

    docking_version_lst = ['5_5']  # '5_5', '5_5'
    for docking_version in docking_version_lst:
        print('\n########## docking_version: ' + str(docking_version))
        start(root_path=root_path, model_path=model_path, partial_model_name=partial_model_name, docking_version=docking_version)
