import sys # required for relative imports in jupyter lab
sys.path.insert(0, '../')

import os

import numpy as np
import pandas as pd

from cosmosis.dataset import CDataset


class Ranzcr(CDataset):
    """https://www.kaggle.com/c/ranzcr-clip-catheter-line-classification/overview"""
    
    def load_data(self, image_dir='./data/ranzcr/train/', targets='./data/ranzcr/train.csv'):
        """creates datadic['pt_id']=(cxr_image_filename,target)"""
        df = pd.read_csv(targets, header=0, index_col=0)
        data = {}
        for file in os.listdir(image_dir):
            target = df.loc[[file[:-4]]].to_numpy()
            target = np.reshape(target, -1)[:-1].astype(np.float64)
            data[file[26:-4]] = (image_dir+'/'+file, target)
        return data