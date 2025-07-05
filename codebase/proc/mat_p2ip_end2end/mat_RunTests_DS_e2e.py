import sys, os
import pandas as pd

from pathlib import Path
path_root = Path(__file__).parents[3]  # upto 'MAT_P2IP_PRJ' folder
sys.path.insert(0, str(path_root))
# print(sys.path)

from codebase.utils import dl_reproducible_result_util
from codebase.utils import PPIPUtils
from codebase.proc.mat_p2ip_DS.mat_p2ip_origMan_auxTlOtherMan.mat_MatP2ipNetwork_origMan_auxTlOtherMan_DS_test import MatP2ipNetworkModule
from codebase.utils.ProjectDataLoader import *
from codebase.utils.feat_engg_manual_main import extract_prot_seq_2D_manual_feat
from codebase.proc.mat_p2ip_DS.mat_p2ip_origMan_auxTlOtherMan import mat_RunTrainTest_origMan_auxTlOtherMan_DS
from codebase.proc.mat_p2ip_DS_hybrid.mat_p2ip_DS_hybrid import create_hybrid_score

from codebase.preproc.seq_to_tl_features_DS import parse_DS_to_fasta, prepare_tl_feat_for_DS_seq
from codebase.preproc.seq_to_manual_features_DS import prepare_manual_feat_for_DS_seq
from dscript_full.cross_spec_pred_code.cross_spec_pred import gen_x_spec_pred
import time


root_path = os.path.join('/project/root/directory/path/here')
root_path = os.path.join('/scratch/pralaycs/Shubh_Working_Remote/PPI_Wkspc/PPI_Code/mat_p2ip_prj_working')


def execute(spec_type = 'human'): 
    print('\n########## spec_type: ' + str(spec_type))
    resultsFolderName = os.path.join(root_path, 'dataset/proc_data_e2e/mat_res_origMan_auxTlOtherMan_' + spec_type + '/')
    # create results folders if they do not exist
    PPIPUtils.makeDir(resultsFolderName)
    hyp = {'fullGPU':True,'deviceType':'cuda'} 

    hyp['maxProteinLength'] = 800  # default: 50
    # # for normalization
    # hyp['featScaleClass'] = StandardScaler
    hyp['hiddenSize'] = 70  # default: 50
    hyp['numLayers'] = 4  # default: 6
    hyp['n_heads'] = 1  # default: 2
    hyp['layer_1_size'] = 1024  # default: 1024  # for the linear layers

    hyp['batchSize'] = 256  # default: 256  # for xai it is 5
    hyp['numEpochs'] = 10  # default: 100
    print('hyp: ' + str(hyp))

    trainSets, testSets, saves, pfs, featureFolder = loadDscriptData(resultsFolderName, spec_type)
    outResultsName = os.path.join(resultsFolderName, 'mat_res_origMan_auxTlOtherMan_' + spec_type + '_DS.txt')
    # specifying human_full model location
    human_full_model_loc = os.path.join(root_path, 'dataset/proc_data_DS/mat_res_origMan_auxTlOtherMan_human/DS_human_full.out')
    loads = [human_full_model_loc]
    mat_RunTrainTest_origMan_auxTlOtherMan_DS.runTest_DS(MatP2ipNetworkModule, outResultsName,trainSets,testSets,featureFolder,hyp,resultsAppend=False,saveModels=None,predictionsFLst = pfs,startIdx=0,loads=loads,spec_type=spec_type)


def extract_prot_seq_feat(spec_type = 'human', crit_lst = [], exec_time_lst = []):
    print('\n########## spec_type: ' + str(spec_type))

    t4 = time.time()
    # extract 1D TL-based features
    parse_DS_to_fasta(root_path, spec_type)
    prepare_tl_feat_for_DS_seq(root_path
                                    ,protTrans_model_path=os.path.join(root_path, '../ProtTrans_Models/')
                                    , protTrans_model_name = 'prot_t5_xl_uniref50'
                                    , spec_type = spec_type)

    t5 = time.time()
    crit_lst.append('mat_p2ip: prepare_tl_feat_for_DS_seq()'); exec_time_lst.append(round(t5-t4, 3))
    # extract 1D manual features
    prepare_manual_feat_for_DS_seq(root_path, spec_type)
    t6 = time.time()
    crit_lst.append('mat_p2ip: prepare_manual_feat_for_DS_seq()'); exec_time_lst.append(round(t6-t5, 3))

    # extract 2D manual features
    # ######### MUST RUN THE FOLLOWING IN THE SAME SHELL WHERE THE PROGRAM WILL RUN TO SET THE makeblastdb AND psiblast PATH (see genPSSM() method of PreprocessUtils.py)
    # export PATH=$PATH:/scratch/pralaycs/Shubh_Working_Remote/ncbi-blast-2.13.0+/bin
    # echo $PATH

    featureDir = os.path.join(root_path, 'dataset/preproc_data_DS/derived_feat/')
    extract_prot_seq_2D_manual_feat(featureDir+spec_type+'/',set(['PSSM', 'LabelEncoding', 'Blosum62', 'SkipGramAA7']), spec_type = spec_type) 
    t7 = time.time()
    crit_lst.append('mat_p2ip: extract_prot_seq_2D_manual_feat()'); exec_time_lst.append(round(t7-t6, 3))


if __name__ == '__main__':
    # #### For a new fresh run, remove the contents of 
    # #### proc_data_e2e, 
    # #### dscript_full/cross_spec_pred_result (don't delete <spec>.log)
    # ### preproc_data_DS (don't delete the derived_feat\<spec>\<spec>.fasta and <spec>_test.csv files)

    crit_lst, exec_time_lst = [], [] 
    t1 = time.time()
    spec_type = 'worm'  # human, ecoli, fly, mouse, worm, yeast
    extract_prot_seq_feat(spec_type, crit_lst, exec_time_lst)
    t2 = time.time()
    crit_lst.append('mat_p2ip: extract_prot_seq_feat()'); exec_time_lst.append(round(t2-t1, 3))
    execute(spec_type)
    t3 = time.time()
    crit_lst.append('mat_p2ip: execute()'); exec_time_lst.append(round(t3-t2, 3))
    crit_lst.append('mat_p2ip: entire_RunTests_DS'); exec_time_lst.append(round(t3-t1, 3))
    
    # invoke the dscript code
    print('invoking the dscript code')
    gen_x_spec_pred(spec_type)
    t8 = time.time()
    crit_lst.append('dscript: gen_x_spec_pred()'); exec_time_lst.append(round(t8-t3, 3))

    # invoking the hybrid scoring algo
    print('invoking the hybrid scoring algo')
    orig_dscript_data_path = os.path.join('/scratch/pralaycs/Shubh_Working_Remote/PPI_Wkspc/PPI_Code/mat_p2ip_prj_working/dscript_full/cross_spec_pred_result')
    mat_p2ip_DS_data_path = os.path.join(root_path, 'dataset/proc_data_e2e')
    mat_p2ip_hybrid_data_path = os.path.join(root_path, 'dataset/proc_data_e2e/mat_p2ip_DS_hybrid')
    
    create_hybrid_score(spec_type, orig_dscript_data_path, mat_p2ip_DS_data_path, mat_p2ip_hybrid_data_path)
    t9 = time.time()
    crit_lst.append('hybrid: create_hybrid_score()'); exec_time_lst.append(round(t9-t8, 3))
    crit_lst.append('Overall(mat_p2ip and dscript hybridized): entire_RunTests_DS_e2e'); exec_time_lst.append(round(t9-t1, 3))

    
    time_df = pd.DataFrame({'Criterion': crit_lst, 'Exec_time': exec_time_lst})
    # save time_df
    time_df.to_csv(os.path.join(root_path, 'dataset/proc_data_e2e/mat_res_origMan_auxTlOtherMan_' + spec_type, spec_type + '_execTime.csv'), index=False)
    print('##### End of End to End flow ######')

