import os
import sys
from pathlib import Path
path_root = Path(__file__).parents[2]
sys.path.insert(0, str(path_root))
import pandas as pd
import torch
import joblib
import numpy as np
from torch.utils.data import DataLoader
from proc.img_p2ip.img_p2ip_dataset_train_full_struct_esmc_v2 import ImgP2ipCustomDataset


def calc_mean_sd_whole_img_dataset(root_path='./', spec_type='human', img_resoln=256, dbl_combi_flg=False, stage='test'):
    preproc_data_path = os.path.join(root_path, 'dataset/preproc_data_DS/train_test_list_dump')
    batch_size = 24
    if stage == "fit" or stage is None:  
        pp_pair_lst_lsts_train, class_label_lst_train = None, None
        pp_pair_lst_lsts_val, class_label_lst_val = None, None
        if(dbl_combi_flg):
            pp_pair_lst_lsts_train = joblib.load(os.path.join(preproc_data_path, 'pp_pair_dbl_train_' + spec_type + '.pkl'))
            class_label_lst_train = joblib.load(os.path.join(preproc_data_path, 'class_label_dbl_train_' + spec_type + '.pkl'))
        else:
            pp_pair_lst_lsts_train = joblib.load(os.path.join(preproc_data_path, 'pp_pair_train_' + spec_type + '.pkl'))
            class_label_lst_train = joblib.load(os.path.join(preproc_data_path, 'class_label_train_' + spec_type + '.pkl'))
        pp_pair_lst_lsts_val = joblib.load(os.path.join(preproc_data_path, 'pp_pair_test_' + spec_type + '.pkl'))
        class_label_lst_val = joblib.load(os.path.join(preproc_data_path, 'class_label_test_' + spec_type + '.pkl'))
        train_data = ImgP2ipCustomDataset(root_path=root_path, spec_type=spec_type
                                          , pp_pair_lst_lsts = pp_pair_lst_lsts_train + pp_pair_lst_lsts_val
                                          , class_label_lst = class_label_lst_train + class_label_lst_val
                                          , img_resoln=img_resoln, transform=None)
        train_loader = DataLoader(train_data, batch_size=batch_size, num_workers=os.cpu_count() - 5, pin_memory= False)  #, multiprocessing_context='spawn')
        mean = 0.0
        for step, (images, _) in enumerate(train_loader):
            batch_samples = images.size(0) 
            images = images.view(batch_samples, images.size(1), -1)
            mean += images.mean(2).sum(0)
        mean = mean / len(train_loader.dataset)
        var = 0.0
        pixel_count = 0
        for step, (images, _) in enumerate(train_loader):
            batch_samples = images.size(0)
            images = images.view(batch_samples, images.size(1), -1)
            var += ((images - mean.unsqueeze(1))**2).sum([0,2])
            pixel_count += images.nelement()
        std = torch.sqrt(var / pixel_count)
        mean = mean.numpy()
        std = std.numpy()
    return (mean, std)


if __name__ == '__main__':
    root_path = os.path.join('/project/root/directory/path/here')
    img_resoln_lst = [400]
    dbl_combi_flg_lst = [False, True]
    stage = 'fit'  
    spec_type_lst = ['human']  
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
    mean_std_train_df = pd.DataFrame({'spec': spec_type_for_df_lst, 'img_resol': img_resoln_for_df_lst
                                , 'dbl_combi_flg': dbl_combi_flg_for_df_lst
                                , 'mean_0': mean_0_lst, 'mean_1': mean_1_lst, 'mean_2': mean_2_lst
                                , 'std_0': std_0_lst, 'std_1': std_1_lst, 'std_2': std_2_lst
                                })
    mean_std_train_df.to_csv(os.path.join(root_path, 'dataset/preproc_data_DS/train_test_list_dump', 'mean_std_train_full_tlStructEsmc.csv'), index=False)
