import os
import sys
from pathlib import Path
path_root = Path(__file__).parents[3]  
sys.path.insert(0, str(path_root))
import lightning as L
import torch
import joblib
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torchvision import transforms
from proc.img_p2ip.dock_img_p2ip_dataset_struct_esmc_v2 import DockImgP2ipCustomDataset


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
        preproc_data_path = os.path.join(self.root_path, 'dataset/preproc_data_docking_BM_' + str(self.docking_version), 'dock_test_list_dump')
        mean_std_train_df = pd.read_csv(os.path.join(self.root_path, 'dataset/preproc_data_DS/train_test_list_dump', 'mean_std_train_full_tlStructEsmc.csv'))
        specific_mean_std_row = mean_std_train_df[(mean_std_train_df['spec'] == 'human') & (mean_std_train_df['img_resol'] == self.img_resoln) & (mean_std_train_df['dbl_combi_flg'] == self.dbl_combi_flg)]
        mean_seq = (specific_mean_std_row['mean_0'].values[0], specific_mean_std_row['mean_1'].values[0], specific_mean_std_row['mean_2'].values[0])
        std_seq = (specific_mean_std_row['std_0'].values[0], specific_mean_std_row['std_1'].values[0], specific_mean_std_row['std_2'].values[0])
        transform = transforms.Compose([transforms.Normalize(mean_seq, std_seq)])
        self.transform = transform
        if stage == "fit" or stage is None:  
            pass
        elif(stage == "test"):  
            pp_pair_lst_lsts_test, class_label_lst_test = None, None
            pp_pair_lst_lsts_test = joblib.load(os.path.join(preproc_data_path, 'dock_pp_pair_test.pkl'))
            class_label_lst_test = joblib.load(os.path.join(preproc_data_path, 'dock_class_label_test.pkl'))
            self.test_data = DockImgP2ipCustomDataset(root_path=self.root_path, spec_type=self.spec_type, docking_version=self.docking_version
                                                    , pp_pair_lst_lsts=pp_pair_lst_lsts_test
                                                    , class_label_lst=class_label_lst_test
                                                    , img_resoln=self.img_resoln, transform=self.transform)


    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=int(self.batch_size)
        , num_workers=self.workers, pin_memory= True if(torch.cuda.is_available()) else False)
