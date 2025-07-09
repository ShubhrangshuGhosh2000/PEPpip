import os
import sys
from pathlib import Path
path_root = Path(__file__).parents[3]  
sys.path.insert(0, str(path_root))
import lightning as L
import torch
import joblib
import pandas as pd
from torch.utils.data import DataLoader
from torch.utils.data import WeightedRandomSampler
from torchvision import transforms
from codebase.proc.img_p2ip.img_p2ip_trx_dataset_train_full import ImgP2ipCustomDataset


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
        preproc_data_path = os.path.join(self.root_path, 'dataset/preproc_data_DS/train_test_list_dump')
        mean_std_train_df = pd.read_csv(os.path.join(self.root_path, 'dataset/preproc_data_DS/train_test_list_dump', 'mean_std_train_full_tlStructEsmc.csv'))
        specific_mean_std_row = mean_std_train_df[(mean_std_train_df['spec'] == 'human') & (mean_std_train_df['img_resol'] == self.img_resoln) & (mean_std_train_df['dbl_combi_flg'] == self.dbl_combi_flg)]
        mean_seq = (specific_mean_std_row['mean_0'].values[0], specific_mean_std_row['mean_1'].values[0], specific_mean_std_row['mean_2'].values[0])
        std_seq = (specific_mean_std_row['std_0'].values[0], specific_mean_std_row['std_1'].values[0], specific_mean_std_row['std_2'].values[0])
        transform = transforms.Compose([transforms.Normalize(mean_seq, std_seq)])
        self.transform = transform
        if stage == "fit" or stage is None:  
            pp_pair_lst_lsts_train, class_label_lst_train = None, None
            pp_pair_lst_lsts_val, class_label_lst_val = None, None
            if(self.dbl_combi_flg):
                pp_pair_lst_lsts_train = joblib.load(os.path.join(preproc_data_path, 'pp_pair_dbl_train_' + self.spec_type + '.pkl'))
                class_label_lst_train = joblib.load(os.path.join(preproc_data_path, 'class_label_dbl_train_' + self.spec_type + '.pkl'))
            else:
                pp_pair_lst_lsts_train = joblib.load(os.path.join(preproc_data_path, 'pp_pair_train_' + self.spec_type + '.pkl'))
                class_label_lst_train = joblib.load(os.path.join(preproc_data_path, 'class_label_train_' + self.spec_type + '.pkl'))
            pp_pair_lst_lsts_val = joblib.load(os.path.join(preproc_data_path, 'pp_pair_test_' + self.spec_type + '.pkl'))
            class_label_lst_val = joblib.load(os.path.join(preproc_data_path, 'class_label_test_' + self.spec_type + '.pkl'))
            if(self.weighted_sampling):
                class_weights = torch.Tensor([100.0/90, 100.0/10])  
                sample_weights = [0] * len(class_label_lst_train + class_label_lst_val)
                for idx, label in enumerate(class_label_lst_train + class_label_lst_val):
                    sample_weights[idx] = class_weights[int(label)]
                self.sample_weights = sample_weights
            self.train_data = ImgP2ipCustomDataset(root_path=self.root_path, spec_type=self.spec_type
                                                    , pp_pair_lst_lsts = pp_pair_lst_lsts_train + pp_pair_lst_lsts_val
                                                    , class_label_lst = class_label_lst_train + class_label_lst_val
                                                    , img_resoln=self.img_resoln, transform=self.transform)
        elif(stage == "test"):  
            pp_pair_lst_lsts_test, class_label_lst_test = None, None
            pp_pair_lst_lsts_test = joblib.load(os.path.join(preproc_data_path, 'pp_pair_test_' + self.spec_type + '.pkl'))
            class_label_lst_test = joblib.load(os.path.join(preproc_data_path, 'class_label_test_' + self.spec_type + '.pkl'))
            self.test_data = ImgP2ipCustomDataset(root_path=self.root_path, spec_type=self.spec_type
                                                    , pp_pair_lst_lsts=pp_pair_lst_lsts_test
                                                    , class_label_lst=class_label_lst_test
                                                    , img_resoln=self.img_resoln, transform=self.transform)
    
    
    def train_dataloader(self):
        train_loader = None
        if(self.weighted_sampling):
            weighted_random_sampler = WeightedRandomSampler(weights=self.sample_weights, num_samples = len(self.train_data.class_label_lst), replacement=False) 
            train_loader = DataLoader(self.train_data, sampler=weighted_random_sampler, batch_size=int(self.batch_size)
                            , num_workers=self.workers, pin_memory= True if(torch.cuda.is_available()) else False)
        else:
            train_loader = DataLoader(self.train_data, batch_size=int(self.batch_size)
                            , num_workers=self.workers, pin_memory= True if(torch.cuda.is_available()) else False
                            , shuffle=True)
        return train_loader
    
    
    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=int(self.batch_size)
        , num_workers=self.workers, pin_memory= True if(torch.cuda.is_available()) else False)
