import os
import sys
from pathlib import Path

path_root = Path(__file__).parents[3]  # upto 'codebase' folder
sys.path.insert(0, str(path_root))
# print(sys.path)

from torch.utils.data import Dataset
import numpy as np
import torch
import joblib


# create custom dataset class
class ImgP2ipCustomDataset(Dataset):
    # constuctor
    def __init__(self, root_path='./', spec_type='human'
                , pp_pair_lst_lsts=None, class_label_lst=None, img_resoln=256, transform=None):
        super(ImgP2ipCustomDataset, self).__init__()
        # # twoD_prot_feat_dict contains all the 2d-tl features of the protein and is of length n x 1024 where n = protein length
        # self.twoD_prot_feat_dict = twoD_prot_feat_dict
        self.root_path = root_path
        self.spec_type = spec_type
        # pp_pair_lst_lsts is a list of lists where each inner list contains prot1_id and prot2_id
        self.pp_pair_lst_lsts = pp_pair_lst_lsts
        # class_label_lst is a list of class labels for each pair
        self.class_label_lst = class_label_lst
        # img_resoln can be 400
        self.img_resoln = img_resoln
        # for the image normalization
        self.transform = transform


    # return the size of the dataset i.e. the number of the pp pairs
    def __len__(self):
        return len(self.class_label_lst)


    # fetch a data sample for a given index
    def __getitem__(self, idx):
        inner_pp_pair_lst = self.pp_pair_lst_lsts[idx]
        label = self.class_label_lst[idx]
        prot1_id, prot2_id = inner_pp_pair_lst[0], inner_pp_pair_lst[1]

        # accommodate sequence based embeddings from ProtTrans
        prot1_tl_2dArr_file_nm_loc = os.path.join(self.root_path, 'dataset/preproc_data_DS/tl_2d_feat_dict_dump_img', self.spec_type, f"prot_id_{prot1_id}.pkl")
        prot1_tl_2dArr = joblib.load(prot1_tl_2dArr_file_nm_loc)
        prot2_tl_2dArr_file_nm_loc = os.path.join(self.root_path, 'dataset/preproc_data_DS/tl_2d_feat_dict_dump_img', self.spec_type, f"prot_id_{prot2_id}.pkl")
        prot2_tl_2dArr = joblib.load(prot2_tl_2dArr_file_nm_loc)
        prot12_product_tl_2dArr = np.matmul(prot1_tl_2dArr, prot2_tl_2dArr.transpose())
        prot12_tl_2dArr_sameDim = np.zeros((self.img_resoln, self.img_resoln))
        prot12_tl_2dArr_sameDim[:prot12_product_tl_2dArr.shape[0], :prot12_product_tl_2dArr.shape[1]] = prot12_product_tl_2dArr

        # accommodate structure-aware embeddings from Prose
        prot1_struct_2dArr_file_nm_loc = os.path.join(self.root_path, 'dataset/preproc_data_DS/tl_struct_2d_feat_dict_dump_img', self.spec_type, f"prot_id_{prot1_id}.pkl")
        prot1_struct_2dArr = joblib.load(prot1_struct_2dArr_file_nm_loc)
        prot2_struct_2dArr_file_nm_loc = os.path.join(self.root_path, 'dataset/preproc_data_DS/tl_struct_2d_feat_dict_dump_img', self.spec_type, f"prot_id_{prot2_id}.pkl")
        prot2_struct_2dArr = joblib.load(prot2_struct_2dArr_file_nm_loc)
        prot12_product_struct_2dArr = np.matmul(prot1_struct_2dArr, prot2_struct_2dArr.transpose())
        prot12_struct_2dArr_sameDim = np.zeros((self.img_resoln, self.img_resoln))
        prot12_struct_2dArr_sameDim[:prot12_product_struct_2dArr.shape[0], :prot12_product_struct_2dArr.shape[1]] = prot12_product_struct_2dArr

        # accommodate embeddings from ESM-C
        prot1_esmc_2dArr_file_nm_loc = os.path.join(self.root_path, 'dataset/preproc_data_DS/tl_esmc_2d_feat_dict_dump_img', self.spec_type, f"prot_id_{prot1_id}.pkl")
        prot1_esmc_2dArr = joblib.load(prot1_esmc_2dArr_file_nm_loc)
        prot1_esmc_2dArr = prot1_esmc_2dArr[1:prot1_esmc_2dArr.shape[0]-1, :]  # Remove BOS (Beginning Of Sequence) token and EOS (End Of Sequence) token
        prot2_esmc_2dArr_file_nm_loc = os.path.join(self.root_path, 'dataset/preproc_data_DS/tl_esmc_2d_feat_dict_dump_img', self.spec_type, f"prot_id_{prot2_id}.pkl")
        prot2_esmc_2dArr = joblib.load(prot2_esmc_2dArr_file_nm_loc)
        prot2_esmc_2dArr = prot2_esmc_2dArr[1:prot2_esmc_2dArr.shape[0]-1, :]  # Remove BOS (Beginning Of Sequence) token and EOS (End Of Sequence) token
        prot12_product_esmc_2dArr = np.matmul(prot1_esmc_2dArr, prot2_esmc_2dArr.transpose())
        prot12_esmc_2dArr_sameDim = np.zeros((self.img_resoln, self.img_resoln))
        prot12_esmc_2dArr_sameDim[:prot12_product_esmc_2dArr.shape[0], :prot12_product_esmc_2dArr.shape[1]] = prot12_product_esmc_2dArr

        prot_prot_3dArr = np.dstack((prot12_tl_2dArr_sameDim, prot12_struct_2dArr_sameDim, prot12_esmc_2dArr_sameDim))
        # print('prot_prot_3dArr:\n' + str(prot_prot_3dArr))
        # transform prot_prot_3dArr to the torch tensors
        prot_prot_3dArr = prot_prot_3dArr.astype(np.float32)
        prot_prot_3dTensor = torch.tensor(prot_prot_3dArr, dtype=torch.float32)
        # transform the prot_prot_3dTensor image from (H, W, C) format to (C, H, W)
        prot_prot_3dTensor = torch.permute(prot_prot_3dTensor, (2, 0, 1))  # transform to channel first format
        # # get normalized prot_prot_3dTensor image
        norm_prot_prot_3dTensor = prot_prot_3dTensor
        if(self.transform):
            norm_prot_prot_3dTensor = self.transform(prot_prot_3dTensor)
        # transform label to the torch tensors
        label_tensor = torch.tensor(label, dtype=torch.long)
        sample = (norm_prot_prot_3dTensor, label_tensor)
        return sample


    # fetch a data sample for a given index and return also the original image before normalization
    def get_item_with_orig_input(self, idx):
        inner_pp_pair_lst = self.pp_pair_lst_lsts[idx]
        label = self.class_label_lst[idx]
        prot1_id, prot2_id = inner_pp_pair_lst[0], inner_pp_pair_lst[1]

        # accommodate sequence based embeddings from ProtTrans
        prot1_tl_2dArr_file_nm_loc = os.path.join(self.root_path, 'dataset/preproc_data_DS/tl_2d_feat_dict_dump_img', self.spec_type, f"prot_id_{prot1_id}.pkl")
        prot1_tl_2dArr = joblib.load(prot1_tl_2dArr_file_nm_loc)
        prot2_tl_2dArr_file_nm_loc = os.path.join(self.root_path, 'dataset/preproc_data_DS/tl_2d_feat_dict_dump_img', self.spec_type, f"prot_id_{prot2_id}.pkl")
        prot2_tl_2dArr = joblib.load(prot2_tl_2dArr_file_nm_loc)
        prot12_product_tl_2dArr = np.matmul(prot1_tl_2dArr, prot2_tl_2dArr.transpose())
        prot12_tl_2dArr_sameDim = np.zeros((self.img_resoln, self.img_resoln))
        prot12_tl_2dArr_sameDim[:prot12_product_tl_2dArr.shape[0], :prot12_product_tl_2dArr.shape[1]] = prot12_product_tl_2dArr

        # accommodate structure-aware embeddings from Prose
        prot1_struct_2dArr_file_nm_loc = os.path.join(self.root_path, 'dataset/preproc_data_DS/tl_struct_2d_feat_dict_dump_img', self.spec_type, f"prot_id_{prot1_id}.pkl")
        prot1_struct_2dArr = joblib.load(prot1_struct_2dArr_file_nm_loc)
        prot2_struct_2dArr_file_nm_loc = os.path.join(self.root_path, 'dataset/preproc_data_DS/tl_struct_2d_feat_dict_dump_img', self.spec_type, f"prot_id_{prot2_id}.pkl")
        prot2_struct_2dArr = joblib.load(prot2_struct_2dArr_file_nm_loc)
        prot12_product_struct_2dArr = np.matmul(prot1_struct_2dArr, prot2_struct_2dArr.transpose())
        prot12_struct_2dArr_sameDim = np.zeros((self.img_resoln, self.img_resoln))
        prot12_struct_2dArr_sameDim[:prot12_product_struct_2dArr.shape[0], :prot12_product_struct_2dArr.shape[1]] = prot12_product_struct_2dArr

        # accommodate embeddings from ESM-C
        prot1_esmc_2dArr_file_nm_loc = os.path.join(self.root_path, 'dataset/preproc_data_DS/tl_esmc_2d_feat_dict_dump_img', self.spec_type, f"prot_id_{prot1_id}.pkl")
        prot1_esmc_2dArr = joblib.load(prot1_esmc_2dArr_file_nm_loc)
        prot1_esmc_2dArr = prot1_esmc_2dArr[1:prot1_esmc_2dArr.shape[0]-1, :]  # Remove BOS (Beginning Of Sequence) token and EOS (End Of Sequence) token
        prot2_esmc_2dArr_file_nm_loc = os.path.join(self.root_path, 'dataset/preproc_data_DS/tl_esmc_2d_feat_dict_dump_img', self.spec_type, f"prot_id_{prot2_id}.pkl")
        prot2_esmc_2dArr = joblib.load(prot2_esmc_2dArr_file_nm_loc)
        prot2_esmc_2dArr = prot2_esmc_2dArr[1:prot2_esmc_2dArr.shape[0]-1, :]  # Remove BOS (Beginning Of Sequence) token and EOS (End Of Sequence) token
        prot12_product_esmc_2dArr = np.matmul(prot1_esmc_2dArr, prot2_esmc_2dArr.transpose())
        prot12_esmc_2dArr_sameDim = np.zeros((self.img_resoln, self.img_resoln))
        prot12_esmc_2dArr_sameDim[:prot12_product_esmc_2dArr.shape[0], :prot12_product_esmc_2dArr.shape[1]] = prot12_product_esmc_2dArr

        prot_prot_3dArr = np.dstack((prot12_tl_2dArr_sameDim, prot12_struct_2dArr_sameDim, prot12_esmc_2dArr_sameDim))
        # print('prot_prot_3dArr:\n' + str(prot_prot_3dArr))
        # transform prot_prot_3dArr to the torch tensors
        prot_prot_3dArr = prot_prot_3dArr.astype(np.float32)
        prot_prot_3dTensor = torch.tensor(prot_prot_3dArr, dtype=torch.float32)
        # transform the prot_prot_3dTensor image from (H, W, C) format to (C, H, W)
        prot_prot_3dTensor = torch.permute(prot_prot_3dTensor, (2, 0, 1))  # transform to channel first format
        # # get normalized prot_prot_3dTensor image
        norm_prot_prot_3dTensor = prot_prot_3dTensor
        if(self.transform):
            norm_prot_prot_3dTensor = self.transform(prot_prot_3dTensor)
        # transform label to the torch tensors
        label_tensor = torch.tensor(label, dtype=torch.long)
        # return the original height (i.e. prot1_len) and original width (i.e. prot2_len) of prot_prot_3dTensor
        orig_ht_b4_padding = prot12_product_tl_2dArr.shape[0]; orig_width_b4_padding = prot12_product_tl_2dArr.shape[1]
        sample = (prot_prot_3dTensor, norm_prot_prot_3dTensor, label_tensor, orig_ht_b4_padding, orig_width_b4_padding)
        return sample

