import os
from torch.utils.data import Dataset
import numpy as np
import torch
import joblib

# create custom dataset class
class ImgP2ipCustomDataset(Dataset):
    # constuctor
    def __init__(self, root_path='./', spec_type='human', partial_twoD_prot_feat_dict=None, man_2d_feat_dict=None, oneD_aa_feat_dict=None
                 , pp_pair_lst_lsts=None, class_label_lst=None, img_resoln=256, transform=None):
        super(ImgP2ipCustomDataset, self).__init__()
        # # twoD_prot_feat_dict contains all the 2d-tl features of the protein and is of length n x 1024 where n = protein length
        # self.twoD_prot_feat_dict = twoD_prot_feat_dict
        self.root_path = root_path
        self.spec_type = spec_type
        self.partial_twoD_prot_feat_dict = partial_twoD_prot_feat_dict
        self.man_2d_feat_dict = man_2d_feat_dict
        # oneD_aa_feat_dict contains all the amino-acid feature-wise pooled protein sequences
        self.oneD_aa_feat_dict = oneD_aa_feat_dict
        # pp_pair_lst_lsts is a list of lists where each inner list contains prot1_id and prot2_id
        self.pp_pair_lst_lsts = pp_pair_lst_lsts
        # class_label_lst is a list of class labels for each pair
        self.class_label_lst = class_label_lst
        # img_resoln can be 256 or 400 or 800
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
        prot1_aa_1dArr = self.oneD_aa_feat_dict[str(prot1_id)]['aa_wise_1d_feat']
        prot1_aa_1dArr_800 = np.zeros(800)
        prot1_aa_1dArr_800[:prot1_aa_1dArr.shape[0]] = prot1_aa_1dArr
        prot1_aa_2dArr = prot1_aa_1dArr_800.reshape(prot1_aa_1dArr_800.shape[0], -1)
        prot2_aa_1dArr = self.oneD_aa_feat_dict[str(prot2_id)]['aa_wise_1d_feat']
        prot2_aa_1dArr_800 = np.zeros(800)
        prot2_aa_1dArr_800[:prot2_aa_1dArr.shape[0]] = prot2_aa_1dArr
        prot2_aa_2dArr = prot2_aa_1dArr_800.reshape(prot2_aa_1dArr_800.shape[0], -1)
        prot12_aa_feat_2dArr = np.matmul(prot1_aa_2dArr, prot2_aa_2dArr.transpose())
        prot12_aa_feat_2dArr = prot12_aa_feat_2dArr[:self.img_resoln, :self.img_resoln]

        # check for the protein level 2d feature sets in partial_twoD_prot_feat_dict (if it is not None)
        if( self.partial_twoD_prot_feat_dict != None):
            key_lst_in_partial_twoD_prot_feat_dict = list(self.partial_twoD_prot_feat_dict.keys())
            # for the prot1_id
            if( prot1_id in key_lst_in_partial_twoD_prot_feat_dict):  # if the protein id is present in partial_twoD_prot_feat_dict, then directly fetch from the dict
                prot1_2dArr_protLevel_orig = self.partial_twoD_prot_feat_dict[prot1_id]
            else:  # otherwise fetch from the saved pkl file
                prot1_2dArr_file_nm_loc = os.path.join(self.root_path, 'dataset/preproc_data_DS/tl_2d_feat_dict_dump_img', self.spec_type, f"prot_id_{prot1_id}.pkl")
                prot1_2dArr_protLevel_orig = joblib.load(prot1_2dArr_file_nm_loc)
            # for the prot2_id
            if( prot2_id in key_lst_in_partial_twoD_prot_feat_dict):  # if the protein id is present in partial_twoD_prot_feat_dict, then directly fetch from the dict
                prot2_2dArr_protLevel_orig = self.partial_twoD_prot_feat_dict[prot2_id]
            else:  # otherwise fetch from the saved pkl file
                prot2_2dArr_file_nm_loc = os.path.join(self.root_path, 'dataset/preproc_data_DS/tl_2d_feat_dict_dump_img', self.spec_type, f"prot_id_{prot2_id}.pkl")
                prot2_2dArr_protLevel_orig = joblib.load(prot2_2dArr_file_nm_loc)
        else:  # fetch the 2d feature sets of both the proteins from the saved pkl files
            prot1_2dArr_file_nm_loc = os.path.join(self.root_path, 'dataset/preproc_data_DS/tl_2d_feat_dict_dump_img', self.spec_type, f"prot_id_{prot1_id}.pkl")
            prot1_2dArr_protLevel_orig = joblib.load(prot1_2dArr_file_nm_loc)
            prot2_2dArr_file_nm_loc = os.path.join(self.root_path, 'dataset/preproc_data_DS/tl_2d_feat_dict_dump_img', self.spec_type, f"prot_id_{prot2_id}.pkl")
            prot2_2dArr_protLevel_orig = joblib.load(prot2_2dArr_file_nm_loc)

        # make each 2dArr in the shape (800 x 1024)
        prot1_2dArr_protLevel = np.zeros((800, 1024))
        prot1_2dArr_protLevel[:prot1_2dArr_protLevel_orig.shape[0], :prot1_2dArr_protLevel_orig.shape[1]] = prot1_2dArr_protLevel_orig
        prot2_2dArr_protLevel = np.zeros((800, 1024))
        prot2_2dArr_protLevel[:prot2_2dArr_protLevel_orig.shape[0], :prot2_2dArr_protLevel_orig.shape[1]] = prot2_2dArr_protLevel_orig
        prot12_mixed_2dArr = np.matmul(prot1_2dArr_protLevel, prot2_2dArr_protLevel.transpose())
        # print('prot12_mixed_2dArr.shape: \n' + str(prot12_mixed_2dArr.shape))
        modified_prot12_mixed_2dArr = prot12_mixed_2dArr[:self.img_resoln, :self.img_resoln]
        # print('modified_prot12_mixed_2dArr.shape: \n' + str(modified_prot12_mixed_2dArr.shape))

        # retrieve the 2d manual features for the respective proteins and multply them
        if(self.man_2d_feat_dict == None):
            prot1_2d_man_feat_arr_file_nm_loc = os.path.join(self.root_path, 'dataset/preproc_data_DS/man_2d_feat_dict_dump_img', self.spec_type, f"prot_id_{prot1_id}_res_{self.img_resoln}.pkl")
            prot1_2d_man_feat_arr = joblib.load(prot1_2d_man_feat_arr_file_nm_loc)
            prot2_2d_man_feat_arr_file_nm_loc = os.path.join(self.root_path, 'dataset/preproc_data_DS/man_2d_feat_dict_dump_img', self.spec_type, f"prot_id_{prot2_id}_res_{self.img_resoln}.pkl")
            prot2_2d_man_feat_arr = joblib.load(prot2_2d_man_feat_arr_file_nm_loc)
        else:
            prot1_2d_man_feat_arr = self.man_2d_feat_dict[prot1_id]
            prot2_2d_man_feat_arr = self.man_2d_feat_dict[prot2_id]
        # convert torch tensor arrays to numpy arrays
        prot1_2d_man_feat_arr = prot1_2d_man_feat_arr.numpy()
        prot2_2d_man_feat_arr = prot2_2d_man_feat_arr.numpy()
        # print(f"prot1_id: {prot1_id} :: prot2_id: {prot2_id}")
        # print(f"prot1_2d_man_feat_arr.shape: {prot1_2d_man_feat_arr.shape} :: prot2_2d_man_feat_arr.shape: {prot2_2d_man_feat_arr.shape}")
        prot12_mixed_man_feat_2dArr = np.matmul(prot1_2d_man_feat_arr, prot2_2d_man_feat_arr.transpose())

        # prot_prot_3dArr = np.dstack((prot1_expanded_2dArr, prot2_expanded_2dArr, modified_prot12_mixed_2dArr))
        prot_prot_3dArr = np.dstack((prot12_aa_feat_2dArr, prot12_mixed_man_feat_2dArr, modified_prot12_mixed_2dArr))
        # print('prot_prot_3dArr:\n' + str(prot_prot_3dArr))
        # transform prot_prot_3dArr to the torch tensors
        prot_prot_3dArr = prot_prot_3dArr.astype(np.float32)
        prot_prot_3dTensor = torch.tensor(prot_prot_3dArr, dtype=torch.float32)
        # transform the prot_prot_3dTensor image from (H, W, C) format to (C, H, W)
        prot_prot_3dTensor = torch.permute(prot_prot_3dTensor, (2, 0, 1))  # transform to channel first format
        # # get normalized prot_prot_3dTensor image
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
        prot1_aa_1dArr = self.oneD_aa_feat_dict[str(prot1_id)]['aa_wise_1d_feat']
        prot1_aa_1dArr_800 = np.zeros(800)
        prot1_aa_1dArr_800[:prot1_aa_1dArr.shape[0]] = prot1_aa_1dArr
        prot1_aa_2dArr = prot1_aa_1dArr_800.reshape(prot1_aa_1dArr_800.shape[0], -1)
        prot2_aa_1dArr = self.oneD_aa_feat_dict[str(prot2_id)]['aa_wise_1d_feat']
        prot2_aa_1dArr_800 = np.zeros(800)
        prot2_aa_1dArr_800[:prot2_aa_1dArr.shape[0]] = prot2_aa_1dArr
        prot2_aa_2dArr = prot2_aa_1dArr_800.reshape(prot2_aa_1dArr_800.shape[0], -1)
        prot12_aa_feat_2dArr = np.matmul(prot1_aa_2dArr, prot2_aa_2dArr.transpose())
        prot12_aa_feat_2dArr = prot12_aa_feat_2dArr[:self.img_resoln, :self.img_resoln]

        # check for the protein level 2d feature sets in partial_twoD_prot_feat_dict (if it is not None)
        if( self.partial_twoD_prot_feat_dict != None):
            key_lst_in_partial_twoD_prot_feat_dict = list(self.partial_twoD_prot_feat_dict.keys())
            # for the prot1_id
            if( prot1_id in key_lst_in_partial_twoD_prot_feat_dict):  # if the protein id is present in partial_twoD_prot_feat_dict, then directly fetch from the dict
                prot1_2dArr_protLevel_orig = self.partial_twoD_prot_feat_dict[prot1_id]
            else:  # otherwise fetch from the saved pkl file
                prot1_2dArr_file_nm_loc = os.path.join(self.root_path, 'dataset/preproc_data_DS/tl_2d_feat_dict_dump_img', self.spec_type, f"prot_id_{prot1_id}.pkl")
                prot1_2dArr_protLevel_orig = joblib.load(prot1_2dArr_file_nm_loc)
            # for the prot2_id
            if( prot2_id in key_lst_in_partial_twoD_prot_feat_dict):  # if the protein id is present in partial_twoD_prot_feat_dict, then directly fetch from the dict
                prot2_2dArr_protLevel_orig = self.partial_twoD_prot_feat_dict[prot2_id]
            else:  # otherwise fetch from the saved pkl file
                prot2_2dArr_file_nm_loc = os.path.join(self.root_path, 'dataset/preproc_data_DS/tl_2d_feat_dict_dump_img', self.spec_type, f"prot_id_{prot2_id}.pkl")
                prot2_2dArr_protLevel_orig = joblib.load(prot2_2dArr_file_nm_loc)
        else:  # fetch the 2d feature sets of both the proteins from the saved pkl files
            prot1_2dArr_file_nm_loc = os.path.join(self.root_path, 'dataset/preproc_data_DS/tl_2d_feat_dict_dump_img', self.spec_type, f"prot_id_{prot1_id}.pkl")
            prot1_2dArr_protLevel_orig = joblib.load(prot1_2dArr_file_nm_loc)
            prot2_2dArr_file_nm_loc = os.path.join(self.root_path, 'dataset/preproc_data_DS/tl_2d_feat_dict_dump_img', self.spec_type, f"prot_id_{prot2_id}.pkl")
            prot2_2dArr_protLevel_orig = joblib.load(prot2_2dArr_file_nm_loc)

        # make each 2dArr in the shape (800 x 1024)
        prot1_2dArr_protLevel = np.zeros((800, 1024))
        prot1_2dArr_protLevel[:prot1_2dArr_protLevel_orig.shape[0], :prot1_2dArr_protLevel_orig.shape[1]] = prot1_2dArr_protLevel_orig
        prot2_2dArr_protLevel = np.zeros((800, 1024))
        prot2_2dArr_protLevel[:prot2_2dArr_protLevel_orig.shape[0], :prot2_2dArr_protLevel_orig.shape[1]] = prot2_2dArr_protLevel_orig
        prot12_mixed_2dArr = np.matmul(prot1_2dArr_protLevel, prot2_2dArr_protLevel.transpose())
        # print('prot12_mixed_2dArr.shape: \n' + str(prot12_mixed_2dArr.shape))
        modified_prot12_mixed_2dArr = prot12_mixed_2dArr[:self.img_resoln, :self.img_resoln]
        # print('modified_prot12_mixed_2dArr.shape: \n' + str(modified_prot12_mixed_2dArr.shape))

        # retrieve the 2d manual features for the respective proteins and multply them
        if(self.man_2d_feat_dict == None):
            prot1_2d_man_feat_arr_file_nm_loc = os.path.join(self.root_path, 'dataset/preproc_data_DS/man_2d_feat_dict_dump_img', self.spec_type, f"prot_id_{prot1_id}_res_{self.img_resoln}.pkl")
            prot1_2d_man_feat_arr = joblib.load(prot1_2d_man_feat_arr_file_nm_loc)
            prot2_2d_man_feat_arr_file_nm_loc = os.path.join(self.root_path, 'dataset/preproc_data_DS/man_2d_feat_dict_dump_img', self.spec_type, f"prot_id_{prot2_id}_res_{self.img_resoln}.pkl")
            prot2_2d_man_feat_arr = joblib.load(prot2_2d_man_feat_arr_file_nm_loc)
        else:
            prot1_2d_man_feat_arr = self.man_2d_feat_dict[prot1_id]
            prot2_2d_man_feat_arr = self.man_2d_feat_dict[prot2_id]
        # convert torch tensor arrays to numpy arrays
        prot1_2d_man_feat_arr = prot1_2d_man_feat_arr.numpy()
        prot2_2d_man_feat_arr = prot2_2d_man_feat_arr.numpy()
        # print(f"prot1_id: {prot1_id} :: prot2_id: {prot2_id}")
        # print(f"prot1_2d_man_feat_arr.shape: {prot1_2d_man_feat_arr.shape} :: prot2_2d_man_feat_arr.shape: {prot2_2d_man_feat_arr.shape}")
        prot12_mixed_man_feat_2dArr = np.matmul(prot1_2d_man_feat_arr, prot2_2d_man_feat_arr.transpose())

        # prot_prot_3dArr = np.dstack((prot1_expanded_2dArr, prot2_expanded_2dArr, modified_prot12_mixed_2dArr))
        prot_prot_3dArr = np.dstack((prot12_aa_feat_2dArr, prot12_mixed_man_feat_2dArr, modified_prot12_mixed_2dArr))
        # print('prot_prot_3dArr:\n' + str(prot_prot_3dArr))
        # transform prot_prot_3dArr to the torch tensors
        prot_prot_3dArr = prot_prot_3dArr.astype(np.float32)
        prot_prot_3dTensor = torch.tensor(prot_prot_3dArr, dtype=torch.float32)
        # transform the prot_prot_3dTensor image from (H, W, C) format to (C, H, W)
        prot_prot_3dTensor = torch.permute(prot_prot_3dTensor, (2, 0, 1))  # transform to channel first format
        # get normalized prot_prot_3dTensor image
        norm_prot_prot_3dTensor = self.transform(prot_prot_3dTensor)
        # transform label to the torch tensors
        label_tensor = torch.tensor(label, dtype=torch.long)
        sample = (prot_prot_3dTensor, norm_prot_prot_3dTensor, label_tensor)
        return sample

