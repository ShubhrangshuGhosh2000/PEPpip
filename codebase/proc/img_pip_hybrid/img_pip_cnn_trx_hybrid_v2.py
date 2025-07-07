import os, sys

from pathlib import Path
path_root = Path(__file__).parents[2]  # upto 'codebase' folder
sys.path.insert(0, str(path_root))
# print(sys.path)

import pandas as pd
from utils import PPIPUtils


def create_hybrid_score(spec_type = 'ecoli', img_pip_cnn_data_path = './', img_pip_trx_data_path = './', img_pip_hybrid_data_path = './'):
    print('inside create_hybrid_score() method - Start')
    print('\n########## spec_type: ' + str(spec_type))
    spec_data_path = os.path.join(img_pip_hybrid_data_path, spec_type)
    PPIPUtils.makeDir(spec_data_path)
    hybrid_pred_prob_1_lst, hybrid_pred_label_lst = [], []
    # read img_pip_cnnt and img_pip_trx specific prediction probabilities and consolidate them in a single df
    img_pip_cnn_res_df = pd.read_csv(os.path.join(img_pip_cnn_data_path, spec_type, spec_type + '_pred_res.csv')
                                            , usecols = ['prot1_id', 'prot2_id', 'actual_res', 'pred_prob_1'])
    img_pip_cnn_res_df.rename(columns={'pred_prob_1': 'img_pip_cnn_pred_prob_1'}, inplace=True)
    img_pip_trx_res_df = pd.read_csv(os.path.join(img_pip_trx_data_path, spec_type, spec_type + '_pred_res.csv')
                                            , usecols = ['prot1_id', 'prot2_id', 'actual_res', 'pred_prob_1'])
    img_pip_trx_res_df.rename(columns={'actual_res': 'actual_res_trx', 'pred_prob_1': 'img_pip_trx_pred_prob_1'}, inplace=True)
    combined_df = pd.concat([img_pip_cnn_res_df, img_pip_trx_res_df], axis=1)
    
    # iterate over combined_df and apply hybrid strategy
    for index, row in combined_df.iterrows():
        img_pip_cnn_pred_prob_1= row['img_pip_cnn_pred_prob_1']
        img_pip_cnn_pred_prob_0 = 1 - img_pip_cnn_pred_prob_1
        img_pip_trx_pred_prob_1 = row['img_pip_trx_pred_prob_1']
        img_pip_trx_pred_prob_0 = 1 - img_pip_trx_pred_prob_1
        # hybrid strategy:
        # Whenever img_pip_trx is predicting positive and img_pip_cnn is predicting negative for the same test sample, then the 
        # adjustment is done in such a way so that the prediction which is associated with the higher confidence level (in terms 
        # of the prediction probability) wins. The same is true for the reverse case and for all the other cases, follow img_pip_trx prediction.
        hybrid_pred_prob_1 = img_pip_cnn_pred_prob_1
        if((img_pip_trx_pred_prob_1 > 0.5) and (img_pip_cnn_pred_prob_0 > 0.5)):
            print('\nrow index for applying hybrid strategy: ' + str(index))
            print('img_pip_trx_pred_prob_1: ' + str(img_pip_trx_pred_prob_1) + ' : img_pip_cnn_pred_prob_0: ' + str(img_pip_cnn_pred_prob_0))
            adj_factor = img_pip_cnn_pred_prob_0 - 0.5 
            hybrid_pred_prob_1 = img_pip_trx_pred_prob_1 - adj_factor
            print('hybrid_pred_prob_1: ' + str(hybrid_pred_prob_1) + ' : actual_res: ' + str(row['actual_res']))
        elif((img_pip_trx_pred_prob_0 > 0.5) and (img_pip_cnn_pred_prob_1 > 0.5)):
            # if img_pip_trx predicts negative but img_pip_cnn predicts positive, then again apply
            # hybrid strategy but in reverse way
            print('\nrow index for applying hybrid strategy: ' + str(index))
            print('img_pip_trx_pred_prob_1: ' + str(img_pip_trx_pred_prob_1) + ' : img_pip_cnn_pred_prob_0: ' + str(img_pip_cnn_pred_prob_0))
            adj_factor = img_pip_trx_pred_prob_0 - 0.5 
            hybrid_pred_prob_1 = img_pip_cnn_pred_prob_1 - adj_factor
            print('hybrid_pred_prob_1: ' + str(hybrid_pred_prob_1) + ' : actual_res: ' + str(row['actual_res']))
        
        hybrid_pred_prob_1_lst.append(hybrid_pred_prob_1)
        # now check the hybrid_pred_prob_1 value and set the hybrid prediction label accordingly
        if(hybrid_pred_prob_1 > 0.5):
            hybrid_pred_label_lst.append(1)
        else:
            hybrid_pred_label_lst.append(0)
    # end of for loop: for index, row in combined_df.iterrows():
    combined_df['hybrid_pred_prob_1'] = hybrid_pred_prob_1_lst
    combined_df['hybrid_pred_label'] = hybrid_pred_label_lst
    # save the combined_df
    combined_df.to_csv(os.path.join(spec_data_path, 'combined_df_' + spec_type + '.csv'), index=False) 
    print('\n prediction result processing')
    # compute result metrics, such as AUPR, Precision, Recall, AUROC
    results = PPIPUtils.calcScores_DS(combined_df['actual_res'].to_numpy(), combined_df['hybrid_pred_prob_1'].to_numpy().reshape((-1, 1)))
    score_df = pd.DataFrame({'Species': [spec_type]
                            , 'AUPR': [results['AUPR']], 'Precision': [results['Precision']]
                            , 'Recall': [results['Recall']], 'AUROC': [results['AUROC']]
                            })
    # save score_df as CSV
    score_df.to_csv(os.path.join(spec_data_path, 'hybrid_score_' + spec_type + '.csv'), index=False) 
    print('inside create_hybrid_score() method - End')


if __name__ == '__main__':
    # root_path = os.path.join('/project/root/directory/path/here')
      

    img_pip_cnn_test_tag = 'ResNet_tlStructEsmc_r400n18DnoWS_epoch37_noWS'
    img_pip_trx_test_tag = 'CDAMViT_tlStructEsmc_r400p16'

    # img_pip_cnn_data_path = os.path.join('/img_pip/cross_species_prediction_result/path/here')
    img_pip_cnn_data_path = os.path.join(root_path, f'dataset/proc_data_tl_feat_to_img/img_p2ip_cnn/test/dscript_other_spec_{img_pip_cnn_test_tag}')
    img_pip_trx_data_path = os.path.join(root_path, f'dataset/proc_data_tl_feat_to_img/img_p2ip_trx/test_pp/dscript_other_spec_{img_pip_trx_test_tag}')
    img_pip_hybrid_data_path = os.path.join(root_path, f'dataset/proc_data_tl_feat_to_img/img_pip_hybrid/img_pip_cnn_trx_hybrid')
    PPIPUtils.createFolder(img_pip_hybrid_data_path, recreate_if_exists=False)

    spec_type_lst = ['ecoli', 'fly', 'mouse', 'worm', 'yeast', 'human']
    # spec_type_lst = ['ecoli']
    for spec_type in spec_type_lst:
        create_hybrid_score(spec_type, img_pip_cnn_data_path, img_pip_trx_data_path, img_pip_hybrid_data_path)
