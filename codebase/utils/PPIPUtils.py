import gzip
import io
import os
import os.path
import random
import shutil
import sys
import urllib.request as request
import zipfile
from pathlib import Path

import numpy as np
import requests
from sklearn import metrics

path_root = Path(__file__).parents[1]  # upto 'codebase' folder
sys.path.insert(0, str(path_root))


def calcScores_DS(labels, phats):
    print('generate the performance metrics for the prediction - D_Script way - Start')
    scores = {'AUPR': 0.000,'Precision':0.000,'Recall': 0.000,'AUROC': 0.000}
    aupr = metrics.average_precision_score(labels, phats)
    auroc = metrics.roc_auc_score(labels, phats)

    y_hat = np.round(phats).astype(int)
    precision = metrics.precision_score(labels, y_hat)
    recall = metrics.recall_score(labels,y_hat)
    tn, fp, fn, tp = metrics.confusion_matrix(labels, y_hat).ravel()
    npv = tn / (tn + fn)
    print('precision: ' + str(precision) + ' : recall: ' + str(recall) + ' : npv: ' + str(npv))

    scores['AUPR'] = aupr
    scores['Precision'] = precision
    scores['Recall'] = recall
    scores['NPV'] = npv
    scores['AUROC'] = auroc
    print('\n scores: ' + str(scores))
    print('generate the performance metrics for the prediction - D_Script way - End')
    return scores


def createKFoldsAllData(data,k,seed=1,balanced=False):
    data = np.asarray(data)
    classData = np.asarray(data[:,2],dtype=np.int32)
    posData = data[classData==1,:]
    negData = data[classData==0,:]
    return createKFolds(posData.tolist(),negData.tolist(),k,seed,balanced)


def createKFolds(pos,neg,k,seed=1,balanced=False):
    random.seed(seed)
    np.random.seed(seed)
    if len(pos[0]) == 2:
        fullPos = []
        for item in pos:
            fullPos.append((item[0],item[1],1))
        fullNeg = []
        for item in neg:
            fullNeg.append((item[0],item[1],0))
        pos = fullPos
        neg = fullNeg
    if balanced:
        pos = np.asarray(pos)
        neg = np.asarray(neg)
        posIdx = [x for x in range(0,len(pos))]
        negIdx = [x for x in range(0,len(neg))]
        random.shuffle(posIdx)
        random.shuffle(negIdx)
        numEntries = min(pos.shape[0],neg.shape[0])
        pos = pos[posIdx[:numEntries],:]
        neg = neg[negIdx[:numEntries],:]
        pos = pos.tolist()
        neg = neg.tolist()
                
    posIdx = [x for x in range(0,len(pos))]
    negIdx = [x for x in range(0,len(neg))]
    random.shuffle(posIdx)
    random.shuffle(negIdx)
    trainSplits = []
    testSplits = []
    pos = np.asarray(pos)
    neg = np.asarray(neg)
    
    for i in range(0,k):
        startP = int((i/k)*pos.shape[0])
        endP = int(((i+1)/k)* pos.shape[0])
        startN = int((i/k)*neg.shape[0])
        endN = int(((i+1)/k)* neg.shape[0])
        if i == k-1:
            endP = pos.shape[0]
            endN = neg.shape[0]
        a = pos[posIdx[:startP],:]
        b = pos[posIdx[endP:],:]
        c = neg[negIdx[:startN],:]
        d = neg[negIdx[endN:],:]
        lst = np.vstack((a,b,c,d))
        np.random.shuffle(lst)
        trainSplits.append(lst)
        
        e = pos[posIdx[startP:endP],:]
        f = neg[negIdx[startN:endN],:]
        lst = np.vstack((e,f))
        np.random.shuffle(lst)
        testSplits.append(lst)
    return trainSplits, testSplits


def makeDir(directory):
    if directory[-1] != '/' and directory[-1] != '\\':
        directory += '/'
    if os.path.isdir(directory):
        pass
    else:
        # os.makedirs(directory)
        os.makedirs(directory, exist_ok=True)  # #############  TEMP CODE TO AVOID 'directory already exists' error in the DDP model training


def formatScores(results,title):
    lst = []
    lst.append([title])
    lst.append(('Acc',results['ACC'],'AUC',results['AUC'],'Prec',results['Prec'],'Recall',results['Recall']))
    lst.append(('Thresholds',results['Thresholds']))
    lst.append(('Max Precision',results['Max Precision']))
    lst.append(('Avg Precision',results['Avg Precision']))
    return lst


# D_Script specific method
def formatScores_DS(results,title):
    lst = []
    lst.append([title])
    lst.append(('AUPR',results['AUPR'],'Precision',results['Precision'],'Recall',results['Recall'],'AUROC',results['AUROC']))
    return lst


def parseUniprotFasta(fileLocation, desiredProteins):
    f= gzip.open(fileLocation,'rb')
    curUniID = ''
    curAASeq = ''
    seqDict ={}
    desiredProteins = set(desiredProteins)
    for line in f:
        line = line.strip().decode('utf-8')
        if line[0] == '>':
            if curUniID in desiredProteins:
                seqDict[curUniID] = curAASeq
            line = line.split('|')
            curUniID = line[1]
            curAASeq = ''
        else:
            curAASeq += line
    f.close()
    if curUniID in desiredProteins:
        seqDict[curUniID] = curAASeq
    return seqDict


def downloadFile(downloadLocation, fileLocation):
    data = request.urlopen(downloadLocation)
    f = open(fileLocation,'wb')
    shutil.copyfileobj(data,f)
    f.close()


def unZip(fileLocation,newFileLocation):
    z = zipfile.ZipFile(fileLocation)
    z.extractall(newFileLocation)


def downloadZipFile(downloadLocation, fileLocation):
    r = requests.get(downloadLocation)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall(fileLocation)


def getUniprotFastaLocations(getBoth=True):
    # uniprotFastaLoc = currentdir+'/PPI_Datasets/uniprot_sprot.fasta.gz'
    # uniprotFastaLoc2 = currentdir+'/PPI_Datasets/uniprot_trembl.fasta.gz'
    uniprotFastaLoc = os.path.join(path_root, 'utils', 'uniprot_sprot.fasta')
    uniprotFastaLoc2 = os.path.join(path_root, 'utils', 'uniprot_trembl.fasta')
    if not os.path.exists(uniprotFastaLoc):
        downloadFile('https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/uniprot_sprot.fasta.gz',uniprotFastaLoc)
    if not os.path.exists(uniprotFastaLoc2) and getBoth:
        downloadFile('https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/uniprot_trembl.fasta.gz',uniprotFastaLoc2)
    return uniprotFastaLoc, uniprotFastaLoc2


def createFolder(dir_path, recreate_if_exists=False):
    try:
        # check if the dir_path already exists and if not, then create it
        if not os.path.exists(dir_path):
            print(f"The directory: '{dir_path}' does not exist. Creating it...")
            os.makedirs(dir_path)
        else:
            print(f"The directory '{dir_path}' already exists.")
            if(recreate_if_exists):
                print(f"As the input argument 'recreate_if_exists' is True, deleting the existing directory and recreating the same...")
                shutil.rmtree(dir_path, ignore_errors=False, onerror=None)
                os.makedirs(dir_path)
            else:
                print(f"As the input argument 'recreate_if_exists' is False, keeping the existing directory as it is ...")
    except OSError as ex:
        errorMessage = "Creation of the directory " + dir_path + " failed. Exception is: " + str(ex)
        raise Exception(errorMessage)
    else:
        print("Returning back from the createFolder() method.")


