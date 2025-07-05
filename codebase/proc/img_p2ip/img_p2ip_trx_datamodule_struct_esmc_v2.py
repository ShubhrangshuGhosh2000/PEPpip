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
from torch.utils.data import WeightedRandomSampler
from torchvision import transforms
from proc.img_p2ip.img_p2ip_trx_dataset_struct_esmc_v2 import ImgP2ipCustomDataset


class ImgP2ipCustomDataModule(L.LightningDataModule):
    def __init__(self, root_path='./', batch_size=16, workers=4, img_resoln=256, spec_type='human', dbl_combi_flg=False, weighted_sampling=False):
        super().__init__()
        self.root_path = root_path
        self.batch_size = batch_size
        self.workers = workers
        self.img_resoln = img_resoln
        self.spec_type = spec_type
        self.dbl_combi_flg = dbl_combi_flg
        self.weighted_sampling = weighted_sampling

    def setup(self, stage=None):
        # Loads in data from file and prepares PyTorch tensor datasets for each split (train, val, test).
        # Setup expects a ‘stage’ arg which is used to separate logic for ‘fit’ and ‘test’.
        # If you don’t mind loading all your datasets at once, you can set up a condition to allow for both ‘fit’ related setup and ‘test’ related setup to run whenever None is passed to stage.
        # ### Note:  this runs across all GPUs and it is safe to make state assignments here
        print('#### inside the setup() method -Start')
        preproc_data_path = os.path.join(self.root_path, 'dataset/preproc_data_DS/train_test_list_dump')
        # first retrieve the mean and std of the entire training dataset irrespective of the model training or testing phase
        mean_std_train_df = pd.read_csv(os.path.join(self.root_path, 'dataset/preproc_data_DS/train_test_list_dump', 'mean_std_train_full_tlStructEsmc.csv'))
        specific_mean_std_row = mean_std_train_df[(mean_std_train_df['spec'] == 'human') & (mean_std_train_df['img_resol'] == self.img_resoln) & (mean_std_train_df['dbl_combi_flg'] == self.dbl_combi_flg)]
        mean_seq = (specific_mean_std_row['mean_0'].values[0], specific_mean_std_row['mean_1'].values[0], specific_mean_std_row['mean_2'].values[0])
        std_seq = (specific_mean_std_row['std_0'].values[0], specific_mean_std_row['std_1'].values[0], specific_mean_std_row['std_2'].values[0])
        # next, create custom transform
        transform = transforms.Compose([transforms.Normalize(mean_seq, std_seq)])
        self.transform = transform

        if stage == "fit" or stage is None:  # for training
            pp_pair_lst_lsts_train, class_label_lst_train = None, None
            pp_pair_lst_lsts_val, class_label_lst_val = None, None

            if(self.dbl_combi_flg):
                pp_pair_lst_lsts_train = joblib.load(os.path.join(preproc_data_path, 'pp_pair_dbl_train_' + self.spec_type + '.pkl'))
                class_label_lst_train = joblib.load(os.path.join(preproc_data_path, 'class_label_dbl_train_' + self.spec_type + '.pkl'))
            else:
                pp_pair_lst_lsts_train = joblib.load(os.path.join(preproc_data_path, 'pp_pair_train_' + self.spec_type + '.pkl'))
                class_label_lst_train = joblib.load(os.path.join(preproc_data_path, 'class_label_train_' + self.spec_type + '.pkl'))
            print('len(pp_pair_lst_lsts_train): ' + str(len(pp_pair_lst_lsts_train)) + '  len(class_label_lst_train): ' + str(len(class_label_lst_train)))

            pp_pair_lst_lsts_val = joblib.load(os.path.join(preproc_data_path, 'pp_pair_test_' + self.spec_type + '.pkl'))
            class_label_lst_val = joblib.load(os.path.join(preproc_data_path, 'class_label_test_' + self.spec_type + '.pkl'))
            print('len(pp_pair_lst_lsts_val): ' + str(len(pp_pair_lst_lsts_val)) + '  len(class_label_lst_val): ' + str(len(class_label_lst_val)))

            # # load partial_twoD_prot_feat_dict from pkl file (partial because it can be safely loaded in the RAM)
            # partial_twoD_prot_feat_dict_file_nm_loc = os.path.join(self.root_path, 'dataset/preproc_data_DS/tl_2d_feat_dict_dump_img', self.spec_type, "partial_twoD_prot_feat_dict_oneThird.pkl")
            # partial_twoD_prot_feat_dict = joblib.load(partial_twoD_prot_feat_dict_file_nm_loc)

            # check weighted_sampling flag
            # 'weighted_sampling' flag is used to indicate whether WeightedRandomSampler would be used 
            # for the batch selection in the DataLoader. It is useful for the trainiing of the dataset with the skewed class distribution.
            print('self.weighted_sampling: ' + str(self.weighted_sampling))
            if(self.weighted_sampling):
                # need to get weight for every sample in the training dataset
                # refer https://towardsdatascience.com/address-class-imbalance-easily-with-pytorch-bb540497d2a6
                class_weights = torch.Tensor([100.0/90, 100.0/10])  # as class-0 entry is 90% of the entire training set
                sample_weights = [0] * len(class_label_lst_train)
                for idx, label in enumerate(class_label_lst_train):
                    sample_weights[idx] = class_weights[int(label)]
                self.sample_weights = sample_weights

            # create the custom torch dataset to be used by the torch dataloader
            print('creating the custom torch dataset to be used by the torch dataloader')
            self.train_data = ImgP2ipCustomDataset(root_path=self.root_path, spec_type=self.spec_type
                                                    , pp_pair_lst_lsts = pp_pair_lst_lsts_train
                                                    , class_label_lst = class_label_lst_train
                                                    , img_resoln=self.img_resoln, transform=self.transform)
            self.val_data = ImgP2ipCustomDataset(root_path=self.root_path, spec_type=self.spec_type
                                                    , pp_pair_lst_lsts = pp_pair_lst_lsts_val
                                                    , class_label_lst = class_label_lst_val 
                                                    , img_resoln=self.img_resoln, transform=self.transform)
        elif(stage == "test"):  # for testing
            print('stage: ' + stage)
            pp_pair_lst_lsts_test, class_label_lst_test = None, None
            pp_pair_lst_lsts_test = joblib.load(os.path.join(preproc_data_path, 'pp_pair_test_' + self.spec_type + '.pkl'))
            class_label_lst_test = joblib.load(os.path.join(preproc_data_path, 'class_label_test_' + self.spec_type + '.pkl'))
            print('len(pp_pair_lst_lsts_test): ' + str(len(pp_pair_lst_lsts_test)) + '  len(class_label_lst_test): ' + str(len(class_label_lst_test)))

            # create the custom torch dataset to be used by the torch dataloader
            print('creating the custom torch dataset to be used by the torch dataloader')
            # self.test_data = ImgP2ipCustomDataset(twoD_prot_feat_dict, oneD_aa_feat_dict, pp_pair_lst_lsts_test, class_label_lst_test, self.img_resoln, self.transform)
            self.test_data = ImgP2ipCustomDataset(root_path=self.root_path, spec_type=self.spec_type
                                                    , pp_pair_lst_lsts=pp_pair_lst_lsts_test
                                                    , class_label_lst=class_label_lst_test
                                                    , img_resoln=self.img_resoln, transform=self.transform)
        print('#### inside the setup() method -End')


    def train_dataloader(self):
        print('#### inside the train_dataloader() method')
        train_loader = None
        print('self.weighted_sampling: ' + str(self.weighted_sampling))
        if(self.weighted_sampling):
            weighted_random_sampler = WeightedRandomSampler(weights=self.sample_weights, num_samples = len(self.train_data.class_label_lst), replacement=False) 
            train_loader = DataLoader(self.train_data, sampler=weighted_random_sampler, batch_size=int(self.batch_size)
                            , num_workers=self.workers, pin_memory= True if(torch.cuda.is_available()) else False)
        else:
            train_loader = DataLoader(self.train_data, batch_size=int(self.batch_size)
                            , num_workers=self.workers, pin_memory= True if(torch.cuda.is_available()) else False
                            , shuffle=True)
        return train_loader
    
    def val_dataloader(self):
        print('#### inside the val_dataloader() method')
        return DataLoader(self.val_data, batch_size=int(self.batch_size)
        , num_workers=self.workers, pin_memory= True if(torch.cuda.is_available()) else False)


    def test_dataloader(self):
        print('#### inside the test_dataloader() method')
        return DataLoader(self.test_data, batch_size=int(self.batch_size)
        , num_workers=self.workers, pin_memory= True if(torch.cuda.is_available()) else False)
