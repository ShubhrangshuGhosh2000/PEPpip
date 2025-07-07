import os
import sys
from pathlib import Path

path_root = Path(__file__).parents[3]  # upto 'codebase' folder
sys.path.insert(0, str(path_root))
# print(sys.path)

import lightning as L
import torch
import joblib
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torchvision import transforms
from proc.img_p2ip.dock_img_p2ip_dataset import DockImgP2ipCustomDataset


class DockImgP2ipCustomDataModule(L.LightningDataModule):
    def __init__(self, root_path='./', batch_size=16, workers=4, img_resoln=256, spec_type='human', docking_version='5_5', dbl_combi_flg=False, weighted_sampling=False):
        super().__init__()
        self.root_path = root_path
        self.batch_size = batch_size
        self.workers = workers
        self.img_resoln = img_resoln
        self.spec_type = spec_type
        self.docking_version=docking_version
        self.dbl_combi_flg = dbl_combi_flg
        self.weighted_sampling = weighted_sampling


    def setup(self, stage=None):
        # Loads in data from file and prepares PyTorch tensor datasets for each split (train, val, test).
        # Setup expects a ‘stage’ arg which is used to separate logic for ‘fit’ and ‘test’.
        # If you don’t mind loading all your datasets at once, you can set up a condition to allow for both ‘fit’ related setup and ‘test’ related setup to run whenever None is passed to stage.
        # ### Note:  this runs across all GPUs and it is safe to make state assignments here
        print('#### inside the setup() method -Start')
        preproc_data_path = os.path.join(self.root_path, 'dataset/preproc_data_docking_BM_' + str(self.docking_version), 'dock_test_list_dump')
        # first retrieve the mean and std of the entire training dataset irrespective of the model training or testing phase
        mean_std_train_df = pd.read_csv(os.path.join(self.root_path, 'dataset/preproc_data_tl_feat_to_img/train_test_list_dump', 'mean_std_train_manTl.csv'))
        specific_mean_std_row = mean_std_train_df[(mean_std_train_df['spec'] == 'human') & (mean_std_train_df['img_resol'] == self.img_resoln) & (mean_std_train_df['dbl_combi_flg'] == self.dbl_combi_flg)]
        mean_seq = (specific_mean_std_row['mean_0'].values[0], specific_mean_std_row['mean_1'].values[0], specific_mean_std_row['mean_2'].values[0])
        std_seq = (specific_mean_std_row['std_0'].values[0], specific_mean_std_row['std_1'].values[0], specific_mean_std_row['std_2'].values[0])
        # next, create custom transform
        transform = transforms.Compose([transforms.Normalize(mean_seq, std_seq)])
        self.transform = transform

        if stage == "fit" or stage is None:  # for training (NOT RELEVANT FOR DOCKING RELATED WORK)
            pass
        elif(stage == "test"):  # for testing
            print('stage: ' + stage)
            pp_pair_lst_lsts_test, class_label_lst_test = None, None
            pp_pair_lst_lsts_test = joblib.load(os.path.join(preproc_data_path, 'dock_pp_pair_test.pkl'))
            class_label_lst_test = joblib.load(os.path.join(preproc_data_path, 'dock_class_label_test.pkl'))
            print('len(pp_pair_lst_lsts_test): ' + str(len(pp_pair_lst_lsts_test)) + '  len(class_label_lst_test): ' + str(len(class_label_lst_test)))

            # load oneD_aa_feat_dict from pkl file
            oneD_aa_feat_dict_fl_nm_with_path = os.path.join(self.root_path, 'dataset/preproc_data_docking_BM_' + str(self.docking_version), 'dock_oneD_aa_feat', 'oneD_aa_feat_dict_len_' + str(self.img_resoln) + '.pkl')
            oneD_aa_feat_dict = joblib.load(oneD_aa_feat_dict_fl_nm_with_path)

            # # load partial_twoD_prot_feat_dict from pkl file (partial because it can be safely loaded in the RAM)
            # partial_twoD_prot_feat_dict_file_nm_loc = os.path.join(self.root_path, 'dataset/preproc_data_DS/tl_2d_feat_dict_dump_img', self.spec_type, "partial_twoD_prot_feat_dict_oneThird.pkl")
            # partial_twoD_prot_feat_dict = joblib.load(partial_twoD_prot_feat_dict_file_nm_loc)

            # # load man_2d_feat_dict
            # print('loading man_2d_feat_dict ...')
            # man2d_featureFolder = os.path.join(self.root_path, 'dataset/preproc_data_DS/derived_feat/')
            # man_2d_feat_dict = self.load_2D_ManualFeatureData_DS(root_path=self.root_path, featureFolder=man2d_featureFolder, img_resoln=self.img_resoln, spec_type=self.spec_type)

            # create the custom torch dataset to be used by the torch dataloader
            print('creating the custom torch dataset to be used by the torch dataloader')
            # self.test_data = ImgP2ipCustomDataset(twoD_prot_feat_dict, oneD_aa_feat_dict, pp_pair_lst_lsts_test, class_label_lst_test, self.img_resoln, self.transform)
            self.test_data = DockImgP2ipCustomDataset(root_path=self.root_path, spec_type=self.spec_type, docking_version=self.docking_version, partial_twoD_prot_feat_dict=None, man_2d_feat_dict=None,
                                                    oneD_aa_feat_dict=oneD_aa_feat_dict, pp_pair_lst_lsts=pp_pair_lst_lsts_test, class_label_lst=class_label_lst_test, 
                                                    img_resoln=self.img_resoln, transform=self.transform)
        print('#### inside the setup() method -End')


    def test_dataloader(self):
        print('#### inside the test_dataloader() method')
        return DataLoader(self.test_data, batch_size=int(self.batch_size)
        , num_workers=self.workers, pin_memory= True if(torch.cuda.is_available()) else False)
