import os
import sys
from pathlib import Path

path_root = Path(__file__).parents[2]  # upto 'codebase' folder
sys.path.insert(0, str(path_root))
# print(sys.path)

import pandas as pd
import torch
import joblib
import numpy as np
from torch.utils.data import DataLoader
from proc.img_p2ip.img_p2ip_dataset_train_full import ImgP2ipCustomDataset
# from preproc.img_p2ip_preproc.man_2d_feat_DS_img import Man2DfeatForImg


def calc_mean_sd_whole_img_dataset(root_path='./', spec_type='human', img_resoln=256, dbl_combi_flg=False, stage='test'):
    print('In calc_mean_sd_whole_img_dataset() method - Start')
    print('\n########## spec_type: ' + str(spec_type))
    print('img_resoln: ' + str(img_resoln) + ' : dbl_combi_flg: ' + str(dbl_combi_flg))
    preproc_data_path = os.path.join(root_path, 'dataset/preproc_data_tl_feat_to_img/train_test_list_dump')
    batch_size = 30
    if stage == "fit" or stage is None:  # for training
        pp_pair_lst_lsts_train, class_label_lst_train = None, None
        # pp_pair_lst_lsts_val, class_label_lst_val = None, None

        if(dbl_combi_flg):
            pp_pair_lst_lsts_train = joblib.load(os.path.join(preproc_data_path, 'pp_pair_dbl_train_' + spec_type + '.pkl'))
            class_label_lst_train = joblib.load(os.path.join(preproc_data_path, 'class_label_dbl_train_' + spec_type + '.pkl'))
        else:
            pp_pair_lst_lsts_train = joblib.load(os.path.join(preproc_data_path, 'pp_pair_train_' + spec_type + '.pkl'))
            class_label_lst_train = joblib.load(os.path.join(preproc_data_path, 'class_label_train_' + spec_type + '.pkl'))
        print('len(pp_pair_lst_lsts_train): ' + str(len(pp_pair_lst_lsts_train)) + '  len(class_label_lst_train): ' + str(len(class_label_lst_train)))

        pp_pair_lst_lsts_val = joblib.load(os.path.join(preproc_data_path, 'pp_pair_test_' + spec_type + '.pkl'))
        class_label_lst_val = joblib.load(os.path.join(preproc_data_path, 'class_label_test_' + spec_type + '.pkl'))
        print('len(pp_pair_lst_lsts_val): ' + str(len(pp_pair_lst_lsts_val)) + '  len(class_label_lst_val): ' + str(len(class_label_lst_val)))

        # load oneD_aa_feat_dict from pkl file
        oneD_aa_feat_dict_fl_nm_with_path = os.path.join(root_path, 'dataset/preproc_data_tl_feat_to_img/oneD_aa_feat', 'oneD_aa_feat_dict_' + spec_type + '_len_' + str(img_resoln) + '.pkl')
        oneD_aa_feat_dict = joblib.load(oneD_aa_feat_dict_fl_nm_with_path)

        # # load partial_twoD_prot_feat_dict from pkl file (partial because it can be safely loaded in the RAM)
        # partial_twoD_prot_feat_dict_file_nm_loc = os.path.join(self.root_path, 'dataset/preproc_data_DS/tl_2d_feat_dict_dump_img', self.spec_type, "partial_twoD_prot_feat_dict_oneThird.pkl")
        # partial_twoD_prot_feat_dict = joblib.load(partial_twoD_prot_feat_dict_file_nm_loc)

        # # load man_2d_feat_dict
        # print('loading man_2d_feat_dict ...')
        # man2d_featureFolder = os.path.join(root_path, 'dataset/preproc_data_DS/derived_feat/')
        # man2DfeatForImg = Man2DfeatForImg()
        # man_2d_feat_dict = man2DfeatForImg.load_2D_ManualFeatureData_DS(root_path=root_path, featureFolder=man2d_featureFolder, img_resoln=img_resoln, spec_type=spec_type)

        # create the custom torch dataset to be used by the torch dataloader
        print('creating the custom torch dataset to be used by the torch dataloader')
        # set 'transform' input argument value as None
        train_data = ImgP2ipCustomDataset(root_path=root_path, spec_type=spec_type, partial_twoD_prot_feat_dict=None, man_2d_feat_dict=None
                                          , oneD_aa_feat_dict=oneD_aa_feat_dict
                                          , pp_pair_lst_lsts = pp_pair_lst_lsts_train + pp_pair_lst_lsts_val
                                          , class_label_lst = class_label_lst_train + class_label_lst_val
                                          , img_resoln=img_resoln, transform=None)
        # create train and validation dataloaders
        train_loader = DataLoader(train_data, batch_size=batch_size, num_workers=os.cpu_count() - 5, pin_memory= False)
        # validation_loader = DataLoader(val_data, batch_size=10, num_workers=0, pin_memory= False)
        
        # calculate mean and standard deviation of the entire training set
        mean = 0.0
        for step, (images, _) in enumerate(train_loader):
            if(step % 500 == 0): print('step = ' +str(step) + ' completed out of ' + str(int((len(pp_pair_lst_lsts_train) + len(pp_pair_lst_lsts_val)) / batch_size)))
            batch_samples = images.size(0) 
            images = images.view(batch_samples, images.size(1), -1)
            mean += images.mean(2).sum(0)
        mean = mean / len(train_loader.dataset)

        var = 0.0
        for step, (images, _) in enumerate(train_loader):
            if(step % 500 == 0): print('step = ' +str(step) + ' completed out of ' + str(int((len(pp_pair_lst_lsts_train) + len(pp_pair_lst_lsts_val)) / batch_size)))
            batch_samples = images.size(0)
            images = images.view(batch_samples, images.size(1), -1)
            var += ((images - mean.unsqueeze(1))**2).sum([0,2])
        std = torch.sqrt(var / (len(train_loader.dataset)*224*224))

        mean = mean.numpy()
        std = std.numpy()
        print(str(spec_type) + " : img_resoln= " + str(img_resoln) + " : dbl_combi_flg= " + str(dbl_combi_flg) + " : mean= " + str(mean))
        print(str(spec_type) + " : img_resoln= " + str(img_resoln) + " : dbl_combi_flg= " + str(dbl_combi_flg) + " : std= " + str(std))

    # elif(stage == "test"):  # for testing
    #     print('stage: ' + stage)
    #     pp_pair_lst_lsts_test, class_label_lst_test = None, None
    #     pp_pair_lst_lsts_test = joblib.load(os.path.join(preproc_data_path, 'pp_pair_test_' + spec_type + '.pkl'))
    #     class_label_lst_test = joblib.load(os.path.join(preproc_data_path, 'class_label_test_' + spec_type + '.pkl'))
    #     print('len(pp_pair_lst_lsts_test): ' + str(len(pp_pair_lst_lsts_test)) + '  len(class_label_lst_test): ' + str(len(class_label_lst_test)))

    #     # load oneD_aa_feat_dict from pkl file
    #     oneD_aa_feat_dict_fl_nm_with_path = os.path.join(root_path, 'dataset/preproc_data_tl_feat_to_img/oneD_aa_feat', 'oneD_aa_feat_dict_' + spec_type + '_len_' + str(img_resoln) + '.pkl')
    #     oneD_aa_feat_dict = joblib.load(oneD_aa_feat_dict_fl_nm_with_path)

    #     # load oneD_prot_feat_dict from pkl file
    #     oneD_prot_feat_dict_fl_nm_with_path = os.path.join(root_path, 'dataset/preproc_data_DS', 'DS_seq_feat_dict_prot_t5_xl_uniref50_' + spec_type + '.pkl')
    #     oneD_prot_feat_dict = joblib.load(oneD_prot_feat_dict_fl_nm_with_path)
        
    #     # create the custom torch dataset to be used by the torch dataloader
    #     print('creating the custom torch dataset to be used by the torch dataloader')
    #     test_data = ImgP2ipCustomDataset(oneD_prot_feat_dict, oneD_aa_feat_dict, pp_pair_lst_lsts_test, class_label_lst_test, img_resoln)

    #     # create test dataloader
    #     test_data_loader = DataLoader(test_data, batch_size=10, num_workers=0, pin_memory= False)
    print('In calc_mean_sd_whole_img_dataset() method - End')
    return (mean, std)



if __name__ == '__main__':
    root_path = os.path.join('/project/root/directory/path/here')
    

    img_resoln_lst = [256, 400, 800]
    # img_resoln_lst = [256]
    dbl_combi_flg_lst = [False, True]
    # dbl_combi_flg_lst = [False]
    stage = 'fit'  # fit, test
    # spec_type_lst = ['ecoli', 'fly', 'mouse', 'worm', 'yeast', 'human']
    spec_type_lst = ['human']  # ######## ONLY human training data-set is relevant for the mean, std calculation to be used in the normalization
    
    spec_type_for_df_lst, img_resoln_for_df_lst, dbl_combi_flg_for_df_lst = [], [], []
    mean_0_lst, mean_1_lst, mean_2_lst = [], [], []
    std_0_lst, std_1_lst, std_2_lst = [], [], []
    for spec_type in spec_type_lst:
        for img_resoln in img_resoln_lst:
            for dbl_combi_flg in dbl_combi_flg_lst:
                mean, std = calc_mean_sd_whole_img_dataset(root_path, spec_type=spec_type, img_resoln=img_resoln, dbl_combi_flg=dbl_combi_flg, stage=stage)
                spec_type_for_df_lst.append(spec_type)
                img_resoln_for_df_lst.append(img_resoln)
                dbl_combi_flg_for_df_lst.append(dbl_combi_flg)
                mean_0_lst.append(mean[0]); mean_1_lst.append(mean[1]); mean_2_lst.append(mean[2]); 
                std_0_lst.append(std[0]); std_1_lst.append(std[1]); std_2_lst.append(std[2]); 
    # end of the outermost for loop: for spec_type in spec_type_lst:
    # create mean_std_train_df
    mean_std_train_df = pd.DataFrame({'spec': spec_type_for_df_lst, 'img_resol': img_resoln_for_df_lst
                                , 'dbl_combi_flg': dbl_combi_flg_for_df_lst
                                , 'mean_0': mean_0_lst, 'mean_1': mean_1_lst, 'mean_2': mean_2_lst
                                , 'std_0': std_0_lst, 'std_1': std_1_lst, 'std_2': std_2_lst
                                })
    # save mean_std_train_df
    mean_std_train_df.to_csv(os.path.join(root_path, 'dataset/preproc_data_tl_feat_to_img/train_test_list_dump', 'mean_std_train_full_manTl.csv'), index=False)
    
