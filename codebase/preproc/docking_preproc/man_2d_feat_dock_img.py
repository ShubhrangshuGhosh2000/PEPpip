import os
import sys
from pathlib import Path

path_root = Path(__file__).parents[2]  # upto 'codebase' folder
sys.path.insert(0, str(path_root))
# print(sys.path)

import joblib
import torch
import pandas as pd

from utils.feat_engg_manual_main import extract_prot_seq_2D_manual_feat_for_dock

class Man2DfeatForImg(object):
    def load_2D_ManualFeatureData_dock(self,root_path='./', featureFolder='./', img_resoln=800, docking_version = '4_0'):
        print('Inside the load_2D_ManualFeatureData_dock() method - Start')
        print('\n########## docking_version: ' + str(docking_version) + ' :: img_resoln: ' + str(img_resoln))

        # first, extract manual 2D features from the sequences
        print('\n ############ first, extract manual 2D features from the sequences \n')
        dock_seq_df = pd.read_csv(os.path.join(root_path,'dataset/preproc_data_docking_BM_' + str(docking_version), 'dock_seq.csv'))
        # iterate through dock_seq_df and populate a list of tuples where each tuple contains (prot_id, seq)
        fastas = []
        for index, row in dock_seq_df.iterrows():
            fastas.append((row['prot_id'], row['seq']))
        # extract_prot_seq_2D_manual_feat_for_dock(fastas, featureFolder, set(['PSSM', 'LabelEncoding', 'Blosum62', 'SkipGramAA7'])) 
        # extract_prot_seq_2D_manual_feat_for_dock(fastas, featureFolder, set(['PSSM'])) 

        # next load manual 2D features
        print('\n ############ next load manual 2D features \n')
        dataLookupSkip, dataMatrixSkip = self.loadEncodingFileWithPadding(os.path.join(featureFolder,'SkipGramAA7H5.encode'),img_resoln)
        dataLookupLabelEncode, dataMatrixLabelEncode = self.loadLabelEncodingFileWithPadding(os.path.join(featureFolder,'LabelEncoding.encode'),img_resoln)
        print("loading pssm_dict ...")
        # load the pssm values stored in pssm_dict
        pssm_dict_pkl_path = os.path.join(Path(__file__).parents[3], 'dataset/preproc_data_docking_BM_' + str(docking_version), 'derived_feat', 'pssm_dict.pkl')
        pssm_dict = joblib.load(pssm_dict_pkl_path)
        # trimming pssm_dict so that it occupies less memory (RAM)
        for prot_id in list(pssm_dict.keys()):
            pssm_dict[prot_id]['seq'] = None
        print("loaded pssm_dict ...\n")
        print("loading blosum62_dict ...")
        # load the pssm values stored in blosum62_dict
        blosum62_dict_pkl_path = os.path.join(Path(__file__).parents[3], 'dataset/preproc_data_docking_BM_' + str(docking_version), 'derived_feat', 'blosum62_dict.pkl')
        blosum62_dict = joblib.load(blosum62_dict_pkl_path)
        # trimming blosum62_dict so that it occupies less memory (RAM)
        for prot_id in list(blosum62_dict.keys()):
            blosum62_dict[prot_id]['seq'] = None
        print("loaded blosum62_dict ...\n")

        allProteinsSet = set(list(dataLookupSkip.keys())) & set(list(dataLookupLabelEncode.keys()))
        allProteinsList = list(allProteinsSet)

        self.encodingSize = self.inSize = dataMatrixSkip.shape[1] + dataMatrixLabelEncode.shape[1] + \
                                            pssm_dict[str(allProteinsList[0])]['pssm_val'].shape[1] + blosum62_dict[str(allProteinsList[0])]['blosum62_val'].shape[1]
        # self.dataLookup = {}
        # man_2d_feat_matrix = torch.zeros((len(allProteinsSet),img_resoln,self.encodingSize))
        man_2d_feat_dict = {}

        for itr, item in enumerate(allProteinsSet):
            if(itr % 50 == 0): print(f"\n ################# docking_version: {docking_version} :: img_resoln: {img_resoln} :: starting {itr}-th iteration out of {len(allProteinsSet)}")
            item = str(item)
            indiv_man_2d_feat_matrix = torch.zeros((img_resoln,self.encodingSize))
            skipData = dataMatrixSkip[dataLookupSkip[item],:,:].T
            labelEncodeData = dataMatrixLabelEncode[dataLookupLabelEncode[item],:,:].T
            indiv_man_2d_feat_matrix[:,:skipData.shape[1]] = skipData
            indiv_man_2d_feat_matrix[:,skipData.shape[1]:(skipData.shape[1] + labelEncodeData.shape[1])] = labelEncodeData

            # processing related to the current pssm-matrix - start
            cur_pssm_mat = pssm_dict[str(item)]['pssm_val']
            pssm_mat_nrows, pssm_mat_ncols = cur_pssm_mat.shape
            # if pssm_mat_nrows is greater than maxProteinLength, then chop the extra part
            if(pssm_mat_nrows > img_resoln):
                cur_pssm_mat = cur_pssm_mat[:img_resoln, :]
            # processing related to thge current pssm-matrix - end
            indiv_man_2d_feat_matrix[:cur_pssm_mat.shape[0], \
                            (skipData.shape[1] + labelEncodeData.shape[1]):(skipData.shape[1] + labelEncodeData.shape[1] + pssm_mat_ncols)] = cur_pssm_mat

            # processing related to the current blosum62-matrix - start
            cur_blosum62_mat = blosum62_dict[str(item)]['blosum62_val']
            blosum62_mat_nrows, blosum62_mat_ncols = cur_blosum62_mat.shape
            # if blosum62_mat_nrows is greater than maxProteinLength, then chop the extra part
            if(blosum62_mat_nrows > img_resoln):
                cur_blosum62_mat = cur_blosum62_mat[:img_resoln, :]
            # processing related to the current blosum62-matrix - end
            indiv_man_2d_feat_matrix[:cur_blosum62_mat.shape[0], \
                            (skipData.shape[1] + labelEncodeData.shape[1] + pssm_mat_ncols):(skipData.shape[1] + labelEncodeData.shape[1] + pssm_mat_ncols + blosum62_mat_ncols)] = cur_blosum62_mat

            man_2d_feat_dict[item]=indiv_man_2d_feat_matrix
            # save indiv_man_2d_feat_matrix as a pkl file
            man_2dArr_file_nm_loc = os.path.join(root_path, 'dataset/preproc_data_docking_BM_' + str(docking_version), 'dock_man_2d_feat_dict_dump_img', f"prot_id_{item}_res_{img_resoln}.pkl")
            joblib.dump(value=indiv_man_2d_feat_matrix, filename=man_2dArr_file_nm_loc, compress=3)
        # end of for loop
        print('Inside the load_2D_ManualFeatureData_dock() method - End')
        return man_2d_feat_dict


    def loadEncodingFileWithPadding(self,fileName,img_resoln=1200,zeroPadding='right',returnLookup = False):
        zeroPadding = zeroPadding.lower()
        lookupMatrix = []
        f = open(fileName)
        for line in f:
            line = line.strip()
            if len(line)==0:
                break #end of lookup matrix
            lookupMatrix.append([float(k) for k in line.split(',')])
        
        lookupMatrix = torch.tensor(lookupMatrix).long()
        
        #lookup for protein name to row index mapping
        proteinNameMapping = {}
        #list of aaIdx (list of tensors)
        aaIdxs = []
        
        #grab all protein data, and map it to our matrix
        for line in f:
            line = line.strip().split()
            name = line[0]
            aaIdx = torch.tensor([int(k) for k in line[1].split(',')]).long()
            aaIdx = aaIdx[:img_resoln]
            if name not in proteinNameMapping:
                proteinNameMapping[name] = len(proteinNameMapping)
                aaIdxs.append(aaIdx)
            else:
                aaIdxs[proteinNameMapping[name]] = aaIdx
        
        #create a torch matrix, will be a 3D array of number of proteins, maxProteinLength, inSize
        dataMatrix = torch.zeros((len(proteinNameMapping),lookupMatrix.shape[1],img_resoln))
        for i in range(0,len(aaIdxs)):
            
            #3 dimension indexing:
            #i,  -- protein index
            #: -- lookupMatrix.shape[1]
            #:x.shape[1],  -- protein length being assigned
            
            #each lookup row will be length lookupMatrix.shape[1], creating an N,lookupMatrix.shape[1] shaped matrix
            #transpose to get N to the first dimension
            x = lookupMatrix[aaIdxs[i],:].T
            if zeroPadding == 'right':
                dataMatrix[i,:,:x.shape[1]] = x
            elif zeroPadding == 'left':
                #calculate gap in front of string
                a = dataMatrix.shape[2]-x.shape[1]
                dataMatrix[i,:,a:] = x
            elif zeroPadding == 'edges':
                #calculate total gap
                a = dataMatrix.shape[2]-x.shape[1]
                #start at half gap
                b = a//2
                #end at half gap + sequence length
                c = b + x.shape[1]
                dataMatrix[i,:,b:c] = x
        
        print('loaded ',dataMatrix.shape[0],'proteins')
        if not returnLookup:
            return proteinNameMapping, dataMatrix
        else:
            return proteinNameMapping, dataMatrix, lookupMatrix


    # ##################################### added for the label-encoding purpose ###################################
    def loadLabelEncodingFileWithPadding(self,fileName,img_resoln=1200,zeroPadding='right',returnLookup = False):
        zeroPadding = zeroPadding.lower()
        lookupMatrix = []  # a list of list (each inner list contains 7 elements for one-hot encoding) 
        f = open(fileName)
        # ##### parsing the 1st part of 'LabelEncoding.encode' file
        for line in f:
            line = line.strip()
            if len(line)==0:
                break #end of lookup matrix
            lookupMatrix.append([float(k) for k in line.split(',')])
        
        lookupMatrix = torch.tensor(lookupMatrix).long()  # 2d tensor array with shape (7 x 7) 
        
        #lookup for protein name to row index mapping
        proteinNameMapping = {}
        #list of aaIdx (list of tensors)
        aaIdxs = []
        
        # ##### parsing the 2nd part of 'LabelEncoding.encode' file
        #grab all protein data, and map it to our matrix
        for line in f:
            line = line.strip().split()
            name = line[0]
            # changing the label-encoding range from 0-6 to 1-7
            # aaIdx = torch.tensor([int(k) for k in line[1].split(',')]).long()  # for int label-encoding (range 0-6)
            aaIdx = torch.tensor([int(k)+1 for k in line[1].split(',')]).long()  # for int label-encoding (range 1-7)
            # ## aaIdx = torch.tensor([(int(k)+1)/7.0 for k in line[1].split(',')]).float()  # for normalized float label-encoding
            aaIdx = aaIdx[:img_resoln]
            if name not in proteinNameMapping:
                proteinNameMapping[name] = len(proteinNameMapping)
                aaIdxs.append(aaIdx)
            else:
                # if the name already exists in proteinNameMapping, then just override the content 
                aaIdxs[proteinNameMapping[name]] = aaIdx
        
        #create a torch matrix, will be a 3D array of number of proteins, 1 (for label-encoding), maxProteinLength,
        dataMatrix = torch.zeros((len(proteinNameMapping),1,img_resoln))
        for i in range(0,len(aaIdxs)):
            
            #3 dimension indexing:
            #i,  -- protein index
            #: -- 1 (for label-encoding)
            #:x.shape[1],  -- protein length being assigned
            
            #each lookup row will be length lookupMatrix.shape[1], creating an N,lookupMatrix.shape[1] shaped matrix
            #transpose to get N to the first dimension
            # x = lookupMatrix[aaIdxs[i],:].T
            x = aaIdxs[i].T
            x = x.reshape(1, len(x))  # x.shape = (1, N)

            if zeroPadding == 'right':
                dataMatrix[i,:,:x.shape[1]] = x
            elif zeroPadding == 'left':
                #calculate gap in front of string
                a = dataMatrix.shape[2]-x.shape[1]
                dataMatrix[i,:,a:] = x
            elif zeroPadding == 'edges':
                #calculate total gap
                a = dataMatrix.shape[2]-x.shape[1]
                #start at half gap
                b = a//2
                #end at half gap + sequence length
                c = b + x.shape[1]
                dataMatrix[i,:,b:c] = x
        
        print('loaded ',dataMatrix.shape[0],'proteins')
        if not returnLookup:
            return proteinNameMapping, dataMatrix
        else:
            return proteinNameMapping, dataMatrix, lookupMatrix

if __name__ == '__main__':
    root_path = os.path.join('/project/root/directory/path/here')
    

    # for dumping each individual 2d manual feature matrix
    man2DfeatForImg = Man2DfeatForImg()
    docking_version_lst = ['4_0']  # '4_0', '5_5'
    img_resoln_lst = [400]  # 256, 400, 800
    for docking_version in docking_version_lst:
        man2d_featureFolder = os.path.join(root_path, 'dataset/preproc_data_docking_BM_' + str(docking_version), 'derived_feat')
        for img_resoln in img_resoln_lst:
            man2DfeatForImg.load_2D_ManualFeatureData_dock(root_path=root_path, featureFolder=man2d_featureFolder, img_resoln=img_resoln, docking_version=docking_version)

