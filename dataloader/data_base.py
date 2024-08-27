from torch import tensor
import torch
import numpy as np
import scanpy as sc
import pandas as pd

from dataloader.data_sourse import DigitsDataset

class OlssonDataset(DigitsDataset):
    def __init__(self, data_name="Olsson", knn=15, sigma=1, n_components=50, train=True, datapath="~/data"):
        self.data_name = data_name
        adata = sc.read("datasets/Olsson.h5ad") # 382 * 50

        data = adata.X
        label_train_str = list(adata.obs['celltype'])
        label_train_str_set = list(set(label_train_str))
        label = tensor(
            np.array([label_train_str_set.index(i) for i in label_train_str]))
        
        self.def_fea_aim = 64
        self.data = tensor(data)
        self.graphwithpca = False
        self.cal_data_rfa(data, knn, sigma, n_components)
        self.label = label
        self.sadata = adata

class UCEPIbcDataset(DigitsDataset):
    def __init__(self, data_name="UCEPIbc", knn=15, sigma=1, n_components=50, train=True, datapath="~/data"):
        self.data_name = data_name

        adata = sc.read("data/ucepi_pca.h5ad")
        data = adata.obsm['X_pca']

        ################# begin
        batch_p = adata.obs['batch_p']
        batch_h = adata.obs['batch_h']
        batch_l = adata.obs['batch_l']
        batch_all = np.array(pd.concat([batch_p, batch_h, batch_l], axis=1))
        n_batch = [30, 3, 2]
        batch_hot = self.multi_one_hot(torch.tensor(batch_all), n_batch)
        len_n_batch = len(n_batch)
        len_batch = sum([n_batch[i] for i in range(len_n_batch)])

        label_celltype = pd.read_csv('/usr/data/DMT_Nature/new/data/SCP551/documentation/uc_epi_celltype.tsv', sep='\t', header=None)
        adata.obs['celltype'] = pd.Categorical(np.squeeze(label_celltype))
        label_train_str = list(np.squeeze(label_celltype.values))
        label_train_str_set = list(set(label_train_str))
        label = tensor(
            np.array([label_train_str_set.index(i) for i in label_train_str]))
        
        self.def_fea_aim = 64
        self.data = tensor(data)
        self.graphwithpca = False
        self.cal_data_rfa(data, knn, sigma, n_components)
        self.label = label
        self.sadata = adata
        self.batch_hot = batch_hot.float()
        self.n_batch = n_batch