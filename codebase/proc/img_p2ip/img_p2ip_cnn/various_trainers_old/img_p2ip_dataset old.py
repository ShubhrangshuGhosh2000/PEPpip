from torch.utils.data import Dataset
import numpy as np
import torch

# create custom dataset class
class ImgP2ipCustomDataset(Dataset):
    # constuctor
    def __init__(self, oneD_prot_feat_dict, oneD_aa_feat_dict, pp_pair_lst_lsts, class_label_lst, img_resoln, transform):
        super(ImgP2ipCustomDataset, self).__init__()
        # oneD_prot_feat_dict contains all the 1d-tl features at the protein level and is of fixed length 1024
        self.oneD_prot_feat_dict = oneD_prot_feat_dict
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
        repeat = self.img_resoln
        prot1_id, prot2_id = inner_pp_pair_lst[0], inner_pp_pair_lst[1]

        prot1_1dArr = self.oneD_aa_feat_dict[str(prot1_id)]['aa_wise_1d_feat']
        prot1_2dArr = prot1_1dArr.reshape(prot1_1dArr.shape[0], -1)
        # print('prot1_2dArr:\n ' + str(prot1_2dArr))
        prot1_expanded_2dArr = np.repeat(prot1_2dArr, repeat, axis=1)
        # print('prot1_expanded_2dArr:\n ' + str(prot1_expanded_2dArr))
        prot2_1dArr = self.oneD_aa_feat_dict[str(prot2_id)]['aa_wise_1d_feat']
        prot2_2dArr = prot2_1dArr.reshape(-1, prot2_1dArr.shape[0])
        # print('prot2_2dArr:\n ' + str(prot2_2dArr))
        prot2_expanded_2dArr = np.repeat(prot2_2dArr, repeat, axis=0)
        # print('prot2_expanded_2dArr: \n' + str(prot2_expanded_2dArr))

        prot1_1dArr_protLevel = self.oneD_prot_feat_dict[prot1_id]['seq_feat']
        prot2_1dArr_protLevel = self.oneD_prot_feat_dict[prot2_id]['seq_feat']
        prot1_1dArr_protLevel_reshaped = prot1_1dArr_protLevel.reshape(-1,1)
        prot2_1dArr_protLevel_reshaped = prot2_1dArr_protLevel.reshape(1,-1)
        prot12_mixed_2dArr = np.matmul(prot1_1dArr_protLevel_reshaped, prot2_1dArr_protLevel_reshaped)
        # print('prot12_mixed_2dArr: \n' + str(prot12_mixed_2dArr))
        modified_prot12_mixed_2dArr = prot12_mixed_2dArr[:self.img_resoln, :self.img_resoln]
        # print('modified_prot12_mixed_2dArr: \n' + str(modified_prot12_mixed_2dArr.shape))

        prot_prot_3dArr = np.dstack((prot1_expanded_2dArr, prot2_expanded_2dArr, modified_prot12_mixed_2dArr))
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
        sample = (norm_prot_prot_3dTensor, label_tensor)
        return sample


    # fetch a data sample for a given index and return also the original image before normalization
    def get_item_with_orig_input(self, idx):
        inner_pp_pair_lst = self.pp_pair_lst_lsts[idx]
        label = self.class_label_lst[idx]
        repeat = self.img_resoln
        prot1_id, prot2_id = inner_pp_pair_lst[0], inner_pp_pair_lst[1]

        prot1_1dArr = self.oneD_aa_feat_dict[str(prot1_id)]['aa_wise_1d_feat']
        prot1_2dArr = prot1_1dArr.reshape(prot1_1dArr.shape[0], -1)
        # print('prot1_2dArr:\n ' + str(prot1_2dArr))
        prot1_expanded_2dArr = np.repeat(prot1_2dArr, repeat, axis=1)
        # print('prot1_expanded_2dArr:\n ' + str(prot1_expanded_2dArr))
        prot2_1dArr = self.oneD_aa_feat_dict[str(prot2_id)]['aa_wise_1d_feat']
        prot2_2dArr = prot2_1dArr.reshape(-1, prot2_1dArr.shape[0])
        # print('prot2_2dArr:\n ' + str(prot2_2dArr))
        prot2_expanded_2dArr = np.repeat(prot2_2dArr, repeat, axis=0)
        # print('prot2_expanded_2dArr: \n' + str(prot2_expanded_2dArr))

        prot1_1dArr_protLevel = self.oneD_prot_feat_dict[prot1_id]['seq_feat']
        prot2_1dArr_protLevel = self.oneD_prot_feat_dict[prot2_id]['seq_feat']
        prot1_1dArr_protLevel_reshaped = prot1_1dArr_protLevel.reshape(-1,1)
        prot2_1dArr_protLevel_reshaped = prot2_1dArr_protLevel.reshape(1,-1)
        prot12_mixed_2dArr = np.matmul(prot1_1dArr_protLevel_reshaped, prot2_1dArr_protLevel_reshaped)
        # print('prot12_mixed_2dArr: \n' + str(prot12_mixed_2dArr))
        modified_prot12_mixed_2dArr = prot12_mixed_2dArr[:self.img_resoln, :self.img_resoln]
        # print('modified_prot12_mixed_2dArr: \n' + str(modified_prot12_mixed_2dArr.shape))

        prot_prot_3dArr = np.dstack((prot1_expanded_2dArr, prot2_expanded_2dArr, modified_prot12_mixed_2dArr))
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

