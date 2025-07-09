import os
import sys
from pathlib import Path
path_root = Path(__file__).parents[3]  
sys.path.insert(0, str(path_root))
from torch.utils.data import Dataset
import numpy as np
import torch
import joblib
class ImgP2ipCustomDataset(Dataset):
    def __init__(self, root_path='./', spec_type='human'
                , pp_pair_lst_lsts=None, class_label_lst=None, img_resoln=256, transform=None):
        super(ImgP2ipCustomDataset, self).__init__()
        self.root_path = root_path
        self.spec_type = spec_type
        self.pp_pair_lst_lsts = pp_pair_lst_lsts
        self.class_label_lst = class_label_lst
        self.img_resoln = img_resoln
        self.transform = transform


    def __len__(self):
        return len(self.class_label_lst)


    def __getitem__(self, idx):
        inner_pp_pair_lst = self.pp_pair_lst_lsts[idx]
        label = self.class_label_lst[idx]
        prot1_id, prot2_id = inner_pp_pair_lst[0], inner_pp_pair_lst[1]
        prot1_tl_2dArr_file_nm_loc = os.path.join(self.root_path, 'dataset/preproc_data_DS/tl_2d_feat_dict_dump_img', self.spec_type, f"prot_id_{prot1_id}.pkl")
        prot1_tl_2dArr = joblib.load(prot1_tl_2dArr_file_nm_loc)
        prot2_tl_2dArr_file_nm_loc = os.path.join(self.root_path, 'dataset/preproc_data_DS/tl_2d_feat_dict_dump_img', self.spec_type, f"prot_id_{prot2_id}.pkl")
        prot2_tl_2dArr = joblib.load(prot2_tl_2dArr_file_nm_loc)
        prot12_product_tl_2dArr = np.matmul(prot1_tl_2dArr, prot2_tl_2dArr.transpose())
        prot12_tl_2dArr_sameDim = np.zeros((self.img_resoln, self.img_resoln))
        prot12_tl_2dArr_sameDim[:prot12_product_tl_2dArr.shape[0], :prot12_product_tl_2dArr.shape[1]] = prot12_product_tl_2dArr
        prot1_struct_2dArr_file_nm_loc = os.path.join(self.root_path, 'dataset/preproc_data_DS/tl_struct_2d_feat_dict_dump_img', self.spec_type, f"prot_id_{prot1_id}.pkl")
        prot1_struct_2dArr = joblib.load(prot1_struct_2dArr_file_nm_loc)
        prot2_struct_2dArr_file_nm_loc = os.path.join(self.root_path, 'dataset/preproc_data_DS/tl_struct_2d_feat_dict_dump_img', self.spec_type, f"prot_id_{prot2_id}.pkl")
        prot2_struct_2dArr = joblib.load(prot2_struct_2dArr_file_nm_loc)
        prot12_product_struct_2dArr = np.matmul(prot1_struct_2dArr, prot2_struct_2dArr.transpose())
        prot12_struct_2dArr_sameDim = np.zeros((self.img_resoln, self.img_resoln))
        prot12_struct_2dArr_sameDim[:prot12_product_struct_2dArr.shape[0], :prot12_product_struct_2dArr.shape[1]] = prot12_product_struct_2dArr
        prot1_esmc_2dArr_file_nm_loc = os.path.join(self.root_path, 'dataset/preproc_data_DS/tl_esmc_2d_feat_dict_dump_img', self.spec_type, f"prot_id_{prot1_id}.pkl")
        prot1_esmc_2dArr = joblib.load(prot1_esmc_2dArr_file_nm_loc)
        prot1_esmc_2dArr = prot1_esmc_2dArr[1:prot1_esmc_2dArr.shape[0]-1, :]  
        prot2_esmc_2dArr_file_nm_loc = os.path.join(self.root_path, 'dataset/preproc_data_DS/tl_esmc_2d_feat_dict_dump_img', self.spec_type, f"prot_id_{prot2_id}.pkl")
        prot2_esmc_2dArr = joblib.load(prot2_esmc_2dArr_file_nm_loc)
        prot2_esmc_2dArr = prot2_esmc_2dArr[1:prot2_esmc_2dArr.shape[0]-1, :]  
        prot12_product_esmc_2dArr = np.matmul(prot1_esmc_2dArr, prot2_esmc_2dArr.transpose())
        prot12_esmc_2dArr_sameDim = np.zeros((self.img_resoln, self.img_resoln))
        prot12_esmc_2dArr_sameDim[:prot12_product_esmc_2dArr.shape[0], :prot12_product_esmc_2dArr.shape[1]] = prot12_product_esmc_2dArr
        prot_prot_3dArr = np.dstack((prot12_tl_2dArr_sameDim, prot12_struct_2dArr_sameDim, prot12_esmc_2dArr_sameDim))
        prot_prot_3dArr = prot_prot_3dArr.astype(np.float32)
        prot_prot_3dTensor = torch.tensor(prot_prot_3dArr, dtype=torch.float32)
        prot_prot_3dTensor = torch.permute(prot_prot_3dTensor, (2, 0, 1))  
        norm_prot_prot_3dTensor = prot_prot_3dTensor
        if(self.transform):
            norm_prot_prot_3dTensor = self.transform(prot_prot_3dTensor)
        label_tensor = torch.tensor(label, dtype=torch.long)
        sample = (norm_prot_prot_3dTensor, label_tensor)
        return sample


    def get_item_with_orig_input(self, idx):
        inner_pp_pair_lst = self.pp_pair_lst_lsts[idx]
        label = self.class_label_lst[idx]
        prot1_id, prot2_id = inner_pp_pair_lst[0], inner_pp_pair_lst[1]
        prot1_tl_2dArr_file_nm_loc = os.path.join(self.root_path, 'dataset/preproc_data_DS/tl_2d_feat_dict_dump_img', self.spec_type, f"prot_id_{prot1_id}.pkl")
        prot1_tl_2dArr = joblib.load(prot1_tl_2dArr_file_nm_loc)
        prot2_tl_2dArr_file_nm_loc = os.path.join(self.root_path, 'dataset/preproc_data_DS/tl_2d_feat_dict_dump_img', self.spec_type, f"prot_id_{prot2_id}.pkl")
        prot2_tl_2dArr = joblib.load(prot2_tl_2dArr_file_nm_loc)
        prot12_product_tl_2dArr = np.matmul(prot1_tl_2dArr, prot2_tl_2dArr.transpose())
        prot12_tl_2dArr_sameDim = np.zeros((self.img_resoln, self.img_resoln))
        prot12_tl_2dArr_sameDim[:prot12_product_tl_2dArr.shape[0], :prot12_product_tl_2dArr.shape[1]] = prot12_product_tl_2dArr
        prot1_struct_2dArr_file_nm_loc = os.path.join(self.root_path, 'dataset/preproc_data_DS/tl_struct_2d_feat_dict_dump_img', self.spec_type, f"prot_id_{prot1_id}.pkl")
        prot1_struct_2dArr = joblib.load(prot1_struct_2dArr_file_nm_loc)
        prot2_struct_2dArr_file_nm_loc = os.path.join(self.root_path, 'dataset/preproc_data_DS/tl_struct_2d_feat_dict_dump_img', self.spec_type, f"prot_id_{prot2_id}.pkl")
        prot2_struct_2dArr = joblib.load(prot2_struct_2dArr_file_nm_loc)
        prot12_product_struct_2dArr = np.matmul(prot1_struct_2dArr, prot2_struct_2dArr.transpose())
        prot12_struct_2dArr_sameDim = np.zeros((self.img_resoln, self.img_resoln))
        prot12_struct_2dArr_sameDim[:prot12_product_struct_2dArr.shape[0], :prot12_product_struct_2dArr.shape[1]] = prot12_product_struct_2dArr
        prot1_esmc_2dArr_file_nm_loc = os.path.join(self.root_path, 'dataset/preproc_data_DS/tl_esmc_2d_feat_dict_dump_img', self.spec_type, f"prot_id_{prot1_id}.pkl")
        prot1_esmc_2dArr = joblib.load(prot1_esmc_2dArr_file_nm_loc)
        prot1_esmc_2dArr = prot1_esmc_2dArr[1:prot1_esmc_2dArr.shape[0]-1, :]  
        prot2_esmc_2dArr_file_nm_loc = os.path.join(self.root_path, 'dataset/preproc_data_DS/tl_esmc_2d_feat_dict_dump_img', self.spec_type, f"prot_id_{prot2_id}.pkl")
        prot2_esmc_2dArr = joblib.load(prot2_esmc_2dArr_file_nm_loc)
        prot2_esmc_2dArr = prot2_esmc_2dArr[1:prot2_esmc_2dArr.shape[0]-1, :]  
        prot12_product_esmc_2dArr = np.matmul(prot1_esmc_2dArr, prot2_esmc_2dArr.transpose())
        prot12_esmc_2dArr_sameDim = np.zeros((self.img_resoln, self.img_resoln))
        prot12_esmc_2dArr_sameDim[:prot12_product_esmc_2dArr.shape[0], :prot12_product_esmc_2dArr.shape[1]] = prot12_product_esmc_2dArr
        prot_prot_3dArr = np.dstack((prot12_tl_2dArr_sameDim, prot12_struct_2dArr_sameDim, prot12_esmc_2dArr_sameDim))
        prot_prot_3dArr = prot_prot_3dArr.astype(np.float32)
        prot_prot_3dTensor = torch.tensor(prot_prot_3dArr, dtype=torch.float32)
        prot_prot_3dTensor = torch.permute(prot_prot_3dTensor, (2, 0, 1))  
        norm_prot_prot_3dTensor = prot_prot_3dTensor
        if(self.transform):
            norm_prot_prot_3dTensor = self.transform(prot_prot_3dTensor)
        label_tensor = torch.tensor(label, dtype=torch.long)
        orig_ht_b4_padding = prot12_product_tl_2dArr.shape[0]; orig_width_b4_padding = prot12_product_tl_2dArr.shape[1]
        sample = (prot_prot_3dTensor, norm_prot_prot_3dTensor, label_tensor, orig_ht_b4_padding, orig_width_b4_padding)
        return sample
