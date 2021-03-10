import sys # required for relative imports in jupyter lab
sys.path.insert(0, '../')

import os

import numpy as np
import pandas as pd

from cosmosis.dataset import CDataset



class Ranzcr(CDataset):
    """https://www.kaggle.com/c/ranzcr-clip-catheter-line-classification/overview
    """
    def load_data(self, image_dir='./data/ranzcr/train/', 
                        target_csv='./data/ranzcr/train.csv',
                        target_type=None):
        """creates datadic['pt_id']=(cxr_image_filename,target)
        target_type = None,'ETT','NG','CATH','SWAN','CVC'
        """
        df = pd.read_csv(target_csv, header=0, index_col=0)
        
        data = {}
        for file in os.listdir(image_dir):
            targets = df.loc[[file[:-4]]].to_numpy()
            target = np.reshape(targets, -1)[:-1].astype(np.int64)

            if target_type == 'ETT':
                target = np.reshape(target[:3], -1)
            if target_type == 'NG':
                target = np.reshape(target[3:7], -1)
            if target_type == 'CATH':
                target = np.reshape(target[7:11], -1)
            if target_type == 'SWAN':
                target = np.reshape(target[10], -1)
            if target_type == 'CVC':
                target = np.reshape(target[7:10], -1)

            data[file[:-4]] = (image_dir+'/'+file, target)
   
        return data