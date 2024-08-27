from torch.utils import data
from sklearn.datasets import load_digits
from torch import tensor
import torchvision.datasets as datasets
from pynndescent import NNDescent
import os
import joblib
import torch
import numpy as np
import scanpy as sc
import scipy
from sklearn.decomposition import PCA
import pandas as pd
import dataloader.cal_sigma as cal_sigma
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import kneighbors_graph
import timeit
from scipy.sparse import csgraph
from scipy.io import mmread
from torch.nn import functional as F


class DigitsDataset(data.Dataset):
    def __init__(self, data_name="Digits", train=True, datapath="~/data"):
        self.data_name = data_name
        digit = load_digits()
        data = tensor(digit.data).float()
        label = tensor(digit.target)

        fea_name = []
        for i in range(8):
            for j in range(8):
                fea_name.append('{}_{}'.format(i, j))

        self.feature_name = np.array(fea_name)
        self.def_fea_aim = 64
        self.train_val_split(data, label, train)
        self.graphwithpca = False

    def cal_near_index(self, k=10, n_components=50, device="cuda", uselabel=False):
        data_name = self.data_name
        if data_name == "UCEPIPcaKnnBC":
            data_name = "UCEPIPcaKnn"
        filename = "save_near_index/data_name{}K{}components{}uselabel{}".format(
            data_name, k, n_components, uselabel)
        os.makedirs("save_near_index", exist_ok=True)
        if not os.path.exists(filename):
            X_rshaped = (
                self.data.reshape(
                    (self.data.shape[0], -1)).detach().cpu().numpy()
            )
            if self.graphwithpca:
                X_rshaped = PCA(n_components=n_components).fit_transform(X_rshaped)
            if not uselabel:
                index = NNDescent(X_rshaped, n_jobs=-1)
                neighbors_index, neighbors_dist = index.query(X_rshaped, k=k+1)
                neighbors_index = neighbors_index[:,1:]
            else:
                dis = pairwise_distances(X_rshaped)
                M = np.repeat(self.label.reshape(1, -1), X_rshaped.shape[0], axis=0)
                dis[(M-M.T)!=0] = dis.max()+1
                neighbors_index = dis.argsort(axis=1)[:, 1:k+1]
            joblib.dump(value=neighbors_index, filename=filename)

            print("save data to ", filename)
        else:
            print("load data from ", filename)
            neighbors_index = joblib.load(filename)
        self.neighbors_index = tensor(neighbors_index).to(device)

    def multi_one_hot(self, index_tensor, depth_list):
        one_hot_tensor = F.one_hot(index_tensor[:,0], num_classes=depth_list[0])
        for col in range(1, len(depth_list)):
            next_one_hot = F.one_hot(index_tensor[:,col], num_classes=depth_list[col])
            one_hot_tensor = torch.cat([one_hot_tensor, next_one_hot], 1)

        return one_hot_tensor

    def cal_data_rfa(self, data, knn, sigma, n_components):
        data_name = self.data_name
        if data_name == "UCEPIPcaKnnBC":
            data_name = "UCEPIPcaKnn"
        filename = "save_data_rfa/data_name{}_knn{}_sigma{}_components{}_data_rfa".format(data_name, knn, sigma, n_components)
        os.makedirs("save_data_rfa", exist_ok=True)
        if not os.path.exists(filename):
            if self.graphwithpca:
                data = PCA(n_components=n_components).fit_transform(data)
            data_rfa = self.compute_rfa(data, mode='features',
                k_neighbours=knn,
                distlocal= 'minkowski',
                distfn='MFIsym',
                connected=True,
                sigma=sigma)
            joblib.dump(value=data_rfa, filename=filename)
            print("save data to ", filename)
        else:
            print("load data from ", filename)
            data_rfa = joblib.load(filename)
        self.data_rfa = tensor(data_rfa)

    def connect_knn(self, KNN, distances, n_components, labels):
        """
        Given a KNN graph, connect nodes until we obtain a single connected
        component.
        """
        c = [list(labels).count(x) for x in np.unique(labels)]

        cur_comp = 0
        while n_components > 1:
            idx_cur = np.where(labels == cur_comp)[0]
            idx_rest = np.where(labels != cur_comp)[0]
            d = distances[idx_cur][:, idx_rest]
            ia, ja = np.where(d == np.min(d))
            i = ia
            j = ja

            KNN[idx_cur[i], idx_rest[j]] = distances[idx_cur[i], idx_rest[j]]
            KNN[idx_rest[j], idx_cur[i]] = distances[idx_rest[j], idx_cur[i]]

            nearest_comp = labels[idx_rest[j]]
            labels[labels == nearest_comp] = cur_comp
            n_components -= 1

        return KNN

    def compute_rfa(self, features, mode='features', k_neighbours=15, distfn='sym', 
        connected=False, sigma=1.0, distlocal='minkowski'):
        """
        Computes the target RFA similarity matrix. The RFA matrix of
        similarities relates to the commute time between pairs of nodes, and it is
        built on top of the Laplacian of a single connected component k-nearest
        neighbour graph of the data.
        """
        start = timeit.default_timer()
        if mode == 'features':
            KNN = kneighbors_graph(features,
                                k_neighbours,
                                mode='distance',
                                metric=distlocal,
                                include_self=False).toarray()

            if 'sym' in distfn.lower():
                KNN = np.maximum(KNN, KNN.T)
            else:
                KNN = np.minimum(KNN, KNN.T)

            n_components, labels = csgraph.connected_components(KNN)

            if connected and (n_components > 1):
                from sklearn.metrics import pairwise_distances
                distances = pairwise_distances(features, metric=distlocal)
                KNN = self.connect_knn(KNN, distances, n_components, labels)
        else:
            KNN = features

        if distlocal == 'minkowski':
            # sigma = np.mean(features)
            S = np.exp(-KNN / (sigma*features.shape[1]))
            # sigma_std = (np.max(np.array(KNN[KNN > 0])))**2
            # print(sigma_std)
            # S = np.exp(-KNN / (2*sigma*sigma_std))
        else:
            S = np.exp(-KNN / sigma)

        S[KNN == 0] = 0
        print("Computing laplacian...")
        L = csgraph.laplacian(S, normed=False)
        print(f"Laplacian computed in {(timeit.default_timer() - start):.2f} sec")

        print("Computing RFA...")
        start = timeit.default_timer()
        RFA = np.linalg.inv(L + np.eye(L.shape[0]))
        RFA[RFA==np.nan] = 0.0
        
        print(f"RFA computed in {(timeit.default_timer() - start):.2f} sec")

        return torch.Tensor(RFA)

    def read_mtx(self, filename, dtype='int32'):
        x = mmread(filename).astype(dtype)
        return x

    def prepare_data(self, fin, with_labels=True, normalize=False, n_pca=0):
        """
        Reads a dataset in CSV format from the ones in datasets/
        """
        df = pd.read_csv(fin + '.csv', sep=',')
        n = len(df.columns)

        if with_labels:
            x = np.double(df.values[:, 0:n - 1])
            labels = df.values[:, (n - 1)]
            labels = labels.astype(str)
            colnames = df.columns[0:n - 1]
        else:
            x = np.double(df.values)
            labels = ['unknown'] * np.size(x, 0)
            colnames = df.columns

        n = len(colnames)

        idx = np.where(np.std(x, axis=0) != 0)[0]
        x = x[:, idx]

        if normalize:
            s = np.std(x, axis=0)
            s[s == 0] = 1
            x = (x - np.mean(x, axis=0)) / s

        if n_pca:
            if n_pca == 1:
                n_pca = n

            nc = min(n_pca, n)
            pca = PCA(n_components=nc)
            x = pca.fit_transform(x)

        labels = np.array([str(s) for s in labels])

        return x, labels

    def train_val_split(self, data, label, train, split_int = 4):
        n_data = data.shape[0]
        g_cpu = torch.Generator()
        g_cpu.manual_seed(0)
        rand_perm = torch.randperm(n_data, generator=g_cpu)
        split_index = n_data * split_int // 5

        if train is True:
            self.data = data[rand_perm[:split_index]]
            self.label = label[rand_perm[:split_index]]
        else:
            self.data = data[rand_perm[split_index:]]
            self.label = label[rand_perm[split_index:]]
        print("train: {} size {}".format(train, self.data.shape))

    def to_device(self, device):
        self.labelstr = [[str(int(i)) for i in self.label]]
        self.data = self.data.to(device)
        self.label = self.label.to(device)
        self.data_rfa = self.data_rfa.to(device)

    def to_device_(self, device):
        self.labelstr = [[str(int(i)) for i in self.label]]
        self.data = self.data.to(device)
        self.label = self.label.to(device)

    def __getitem__(self, index):
        return index

    def __len__(self):
        return self.data.shape[0]

    def get_dim(
        self,
    ):
        return self.data[0].shape