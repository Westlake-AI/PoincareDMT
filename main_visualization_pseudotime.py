import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import numpy as np
import pandas as pd
import scanpy as sc
import pandas as pd
import numpy as np
import pytorch_lightning as pl
import torch as th
import itertools
import random

from dataloader import data_base
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.cluster import *
from sklearn import metrics

import warnings
warnings.filterwarnings("ignore")


sns.set_style('white', {'legend.frameon':True})

colors_palette = ['#1F77B4', '#FF7F0E', '#2CA02C', '#D62728', '#9467BD',
                  '#8C564B', '#E377C2', '#BCBD22', '#17BECF', '#40004B',
                  '#762A83', '#9970AB', '#C2A5CF', '#E7D4E8', '#F7F7F7',
                  '#D9F0D3', '#A6DBA0', '#5AAE61', '#1B7837', '#00441B',
                  '#8DD3C7', '#FFFFB3', '#BEBADA', '#FB8072', '#80B1D3',
                  '#FDB462', '#B3DE69', '#FCCDE5', '#D9D9D9', '#BC80BD',
                  '#CCEBC5', '#FFED6F', '#edf8b1', '#c7e9b4', '#7fcdbb',
                  '#41b6c4', '#1d91c0', '#225ea8', '#253494', '#081d58']


def get_geodesic_parameters(u, v, eps=1e-10):
    if all(u) == 0:
        u = np.array([eps, eps])
    if all(v) == 0:
        v = np.array([eps, eps])

    nu = u[0]**2 + u[1]**2
    nv = v[0]**2 + v[1]**2
    a = (u[1]*nv - v[1]*nu + u[1] - v[1]) / (u[0]*v[1] - u[1]*v[0])
    b = (v[0]*nu - u[0]*nv + v[0] - u[0]) / (u[0]*v[1] - u[1]*v[0])
    return a, b

def plot_poincare_disc(x, radius=None, labels=None, title_name=None, model_name=None,
    labels_name='labels', labels_order=None, labels_pos=None, labels_text=None,
                       file_name=None, coldict=None,
                       d1=4.5, d2=4.0, fs=9, ms=20,
                       u=None, v=None, alpha=1.0,
                       col_palette=plt.get_cmap("tab10"), print_labels=True,
                       bbox=(1.3, 0.7), leg=True, ft='png'):    

    idx = np.random.permutation(len(x))
    df = pd.DataFrame(x[idx, :], columns=['pm1', 'pm2'])
    
    fig = plt.figure(figsize=(d1, d2))
    ax = plt.gca()
    circle = plt.Circle((0, 0), radius=radius,  fc='none', color='black')
    ax.add_patch(circle)
    ax.plot(0, 0, '.', c=(0, 0, 0), ms=4)
    if title_name:
        ax.set_title(title_name, fontsize=fs)

    if not (labels is None):
        df[labels_name] = labels[idx]
        if labels_order is None:
            labels_order = np.unique(labels)      
        if coldict is None:
            coldict = dict(zip(labels_order, col_palette[:len(labels)]))
        sns.scatterplot(x="pm1", y="pm2", hue=labels_name, sizes=1, 
                        hue_order=labels_order,
                        palette=coldict,
                        alpha=alpha, edgecolor="none",
                        data=df, ax=ax, s=ms)
        
        if leg:
            ax.legend(fontsize=fs, loc='best', bbox_to_anchor=bbox, facecolor='white')
        else:
            ax.legend_.remove()
            
    else:
        sns.scatterplot(x="pm1", y="pm2",
                        data=df, ax=ax, s=ms)

        if leg == False:
            ax.legend_.remove()

    if not (u is None):     
        a, b = get_geodesic_parameters(u, v)        
        circle_geo = plt.Circle((-a/2, -b/2), radius=np.sqrt(a**2/4 + b**2/4 - 1),  fc='none', color='grey')
        ax.add_patch(circle_geo)

    fig.tight_layout()
    ax.axis('off')
    ax.axis('equal') 

    if print_labels:
        if labels_text is None:
            labels_list = np.unique(labels)
        else:
            labels_list = np.unique(labels_text)
        if labels_pos is None:  
            labels_pos = {}  
            for l in labels_list:
                ix_l = np.where(labels == l)[0]
                if model_name == 'dhv':
                    Dl = poincare_distance_dhv(th.DoubleTensor(x[ix_l, :])).numpy()
                if model_name == 'dv':
                    Dl = poincare_distance_dv(th.DoubleTensor(x[ix_l, :])).numpy()
                else:
                    Dl = poincare_distance_poinmaps(th.DoubleTensor(x[ix_l, :])).numpy()
                i = ix_l[np.argmin(Dl.sum(axis=0))]
                labels_pos[l] = i

        for l in labels_list:    
            ax.text(x[labels_pos[l], 0], x[labels_pos[l], 1], l, fontsize=fs)

    plt.tight_layout()

    if file_name:
        if ft == 'png':
            plt.savefig(file_name + '.' + ft, format=ft, dpi=600)
        else:
            plt.savefig(file_name + '.' + ft, format=ft)

    return labels_pos

def get_projected_coordinates(u):
    if u[0] == 0 or u[1] == 0:
        return [np.sign(u[0]), np.sign(u[1])]
    c = u[1] / u[0]
    x_c = np.sign(u[0])*np.sqrt(1 / (1+c**2))
    y_c = c*x_c
    
    return [x_c, y_c]

def get_confusion_matrix(classes, true_labels_oder, true_labels, fname='', title='Confusion matrix'):
    cm = np.zeros([len(np.unique(classes)), len(np.unique(true_labels))])
    for il, l in enumerate(np.unique(classes)):
        idx = np.where(classes == l)[0]
        for it, tl in enumerate(true_labels_oder):
            cm[il, it] = len(idx[np.where(true_labels[idx] == tl)[0]])
    plot_confusion_matrix(cm, np.unique(classes), true_labels_oder,
                          normalize=True,
                          title=title,
                          cmap=plt.cm.Blues, fname=fname)
    return cm

def plot_confusion_matrix(cm, classes, true_labels,
                          normalize=False,
                          title='Lineage matrix',
                          cmap=plt.cm.Blues, fs=9, fname=''):
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=0)[np.newaxis, :]
    
    fig = plt.figure(figsize=(3.5, 3.5))
    
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    x_tick_marks = np.arange(len(true_labels))
    y_tick_marks = np.arange(len(classes))
    plt.xticks(x_tick_marks, true_labels, rotation=45, fontsize=fs)
    plt.yticks(y_tick_marks, classes, fontsize=fs)

    fmt = '.2f' if normalize else '.0f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black", fontsize=fs)

    plt.tight_layout()
    plt.grid(False)

    plt.ylabel(title, fontsize=fs)
    plt.xlabel('stages', fontsize=fs)
    
    if fname != '':
        plt.savefig(fname + '.png', format='png', dpi=600)
        
    plt.show()
    plt.close()

def linear_scale(embeddings):
    embeddings = np.transpose(embeddings)
    sqnorm = np.sum(embeddings ** 2, axis=1, keepdims=True)
    dist = np.arccosh(1 + 2 * sqnorm / (1 - sqnorm))
    dist = np.sqrt(dist)
    dist /= dist.max()
    sqnorm[sqnorm == 0] = 1
    embeddings = dist * embeddings / np.sqrt(sqnorm)
    return np.transpose(embeddings)

def poincare_translation(v, x):
    """
    Computes the translation of x  when we move v to the center.
    Hence, the translation of u with -u should be the origin.
    """
    xsq = (x ** 2).sum(axis=1)
    vsq = (v ** 2).sum()
    xv = (x * v).sum(axis=1)
    a = np.matmul((xsq + 2 * xv + 1).reshape(-1, 1),
                  v.reshape(1, -1)) + (1 - vsq) * x
    b = xsq * vsq + 2 * xv + 1
    return np.dot(np.diag(1. / b), a)

def euclidean_distance_poinmaps(x):
    th.set_default_tensor_type('torch.DoubleTensor')
    # print('computing euclidean distance...')
    nx = x.size(0)
    x = x.contiguous()
    
    x = x.view(nx, -1)

    norm_x = th.sum(x ** 2, 1, keepdim=True).t()
    ones_x = th.ones(nx, 1)

    xTx = th.mm(ones_x, norm_x)
    xTy = th.mm(x, x.t())
    
    d = (xTx.t() + xTx - 2 * xTy)
    d[d < 0] = 0

    return d

def poincare_distance_poinmaps(x):
    # print('computing poincare distance...')
    eps = 1e-5
    boundary = 1 - eps
    
    nx = x.size(0)
    x = x.contiguous()
    x = x.view(nx, -1)
    
    norm_x = th.sum(x ** 2, 1, keepdim=True)
    sqdist = euclidean_distance_poinmaps(x) * 2    
    squnorm = 1 - th.clamp(norm_x, 0, boundary)

    x = (sqdist / th.mm(squnorm, squnorm.t())) + 1
    z = th.sqrt(th.pow(x, 2) - 1)
    
    return th.log(x + z)

def poincare_distance_dhv(x):
    # print('computing poincare distance...')
    import manifolds.poincare as poincare
    PoincareBall = poincare.PoincareBall()
    dist = PoincareBall.sqdist_xu_mobius_v2(x, x, c=1)
    dist = dist.clamp(min=1e-12)
    
    return dist

def poincare_distance_dv(x):
    # print('computing poincare distance...')
    import manifolds.poincare as poincare
    PoincareBall = poincare.PoincareBall()
    dist = PoincareBall.sqdist_xu_mobius_v2(x, x, c=1)
    dist = dist.clamp(min=1e-22)
    
    return dist

def init_scanpy_root(data, head_name, true_labels):
    head_idx = np.where(true_labels == head_name)[0]
    if len(head_idx) > 1:
        D = pairwise_distances(data[head_idx, :], metric='euclidean')
        iroot = head_idx[np.argmin(D.sum(axis=0))]
    else:
        iroot = head_idx[0]
        
    return iroot

def init_scanpy_louvain(data, true_labels, k=30, n_pcs=20, computeEmbedding=True):
    adata = sc.AnnData(data)
    if computeEmbedding:
        if n_pcs:
            sc.pp.pca(adata, n_comps=n_pcs)
            sc.pp.neighbors(adata, n_neighbors=k, n_pcs=n_pcs)
        else:
            sc.pp.neighbors(adata, n_neighbors=k)
        
    
        sc.tl.louvain(adata, resolution=0.9)
        louvain_labels = np.array(list(adata.obs['louvain']))
    else:
        louvain_labels = []

    return iroot, louvain_labels

def get_scores(true_lineages, labels):
    ARS = metrics.adjusted_rand_score(true_lineages, labels)
    FMS = metrics.fowlkes_mallows_score(true_lineages, labels)
    print(f"ARI = {ARS:.2f}")
    print(f"FMS = {FMS:.2f}")
    
    return ARS, FMS
    
def kMedoids(D, k, tmax=10000):
    # determine dimensions of distance matrix D
    m, n = D.shape

    if k > n:
        raise Exception('too many medoids')

    # find a set of valid initial cluster medoid indices since we
    # can't seed different clusters with two points at the same location
    valid_medoid_inds = set(range(n))
    invalid_medoid_inds = set([])
    rs, cs = np.where(D == 0)
    # the rows, cols must be shuffled because we will keep the first duplicate below
    index_shuf = list(range(len(rs)))
    np.random.shuffle(index_shuf)
    rs = rs[index_shuf]
    cs = cs[index_shuf]
    for r, c in zip(rs, cs):
        # if there are two points with a distance of 0...
        # keep the first one for cluster init
        if r < c and r not in invalid_medoid_inds:
            invalid_medoid_inds.add(c)
    valid_medoid_inds = list(valid_medoid_inds - invalid_medoid_inds)

    if k > len(valid_medoid_inds):
        raise Exception('too many medoids (after removing {} duplicate points)'.format(
            len(invalid_medoid_inds)))

    # randomly initialize an array of k medoid indices

    M = np.array(valid_medoid_inds)
    np.random.shuffle(M)
    M = np.sort(M[:k])

    # for i in np.unique(spectral_labels):
    #     idx = list(np.where(spectral_labels == i)[0])
    #     M[i] = random.sample(idx, 1)[0]

    # create a copy of the array of medoid indices
    Mnew = np.copy(M)

    # initialize a dictionary to represent clusters
    C = {}
    for t in range(tmax):
        # determine clusters, i. e. arrays of data indices
        J = np.argmin(D[:, M], axis=1)
        for kappa in range(k):
            C[kappa] = np.where(J == kappa)[0]
        # update cluster medoids
        for kappa in range(k):
            J = np.mean(D[np.ix_(C[kappa], C[kappa])], axis=1)
            j = np.argmin(J)
            Mnew[kappa] = C[kappa][j]
        np.sort(Mnew)
        # check for convergence
        if np.array_equal(M, Mnew):
            break
        M = np.copy(Mnew)
    else:
        # final update of cluster memberships
        J = np.argmin(D[:, M], axis=1)
        for kappa in range(k):
            C[kappa] = np.where(J == kappa)[0]

    labels = np.empty((n))
    for k in C.keys():
        labels[list(C[k])] = str(k)

    # return results
    return labels

def detect_cluster(D, n_clusters=2, clustering_name='spectral', k=15, distances='Poincaré'):
    
    if clustering_name == 'spectral':
        similarity = np.exp(-D**2)
        clustering = SpectralClustering(n_clusters=n_clusters, eigen_solver='arpack', 
                                        affinity='precomputed', n_neighbors=k).fit(similarity)
        labels = clustering.labels_
    elif clustering_name == 'kmedoids':
        clustering = kMedoids(D, n_clusters, tmax=10000)
        labels = clustering
    else:
        clustering = AgglomerativeClustering(linkage='average', n_clusters=n_clusters, 
                                             affinity='precomputed').fit(D)
        labels = clustering.labels_
    
    scores = get_scores(true_lineages, labels)
    model.plot('rot', labels=labels, file_name=fout + '_' + clustering_name + '_' + distances, 
               title_name=f"{clustering_name} {distances}\nARI = {scores[0]:.2f}, FMS = {scores[1]:.2f}", 
               zoom=None, bbox=(1.1, 0.8))
    
    title = f"{clustering_name} {distances}"
    return scores, title

def get_clustering_score_table(louvain_labels, true_lineages):
    nc = len(np.unique(louvain_labels))
    scores = []
    titles = []
    scores.append(get_scores(true_lineages, louvain_labels))
    titles.append('louvain')

    model.plot('rot', labels=louvain_labels, file_name=fout + '_louvain', 
                   title_name=f"louvain", 
                   zoom=None, bbox=(1.1, 0.8))


    for cname in ['spectral', 'agglomerative', 'kmedoids']:
        s, t = detect_cluster(model.distances, n_clusters=nc, clustering_name=cname, k=15, distances='Poincaré')
        scores.append(s)
        titles.append(t)

    for cname in ['spectral', 'agglomerative', 'kmedoids']:
        s, t = detect_cluster(pairwise_distances(data), 
                              n_clusters=nc, clustering_name=cname, k=15, distances='raw')
        scores.append(s)
        titles.append(t)
            
    title_name_dict = {'X_pca': 'PCA',
                       'X_tsne': 't-SNE',
                       'X_umap': 'UMAP', 
                       'X_diffmap': 'DiffusionMaps', 
                       'X_draw_graph_fa': 'ForceAtlas2'}

    for embedding_name in ['X_pca', 'X_tsne', 'X_umap', 'X_diffmap', 'X_draw_graph_fa']:
        for cname in ['spectral', 'agglomerative', 'kmedoids']:
            s, t = detect_cluster(pairwise_distances(adata.obsm[embedding_name]), 
                                  n_clusters=nc, clustering_name=cname, k=15, distances=title_name_dict[embedding_name])
            scores.append(s)
            titles.append(t)
            
    return titles, scores

def plot_dp_on_poincare(coordinates, distances, cell=0, fs=9, title_name='None', file_name=None, d1=3.5, d2=3.1, ms=1):
    fig = plt.figure(figsize=(d1, d2))
    radius_new = np.sqrt(coordinates[:,0]**2 + coordinates[:,1]**2).max()
    circle = plt.Circle((0, 0), radius=radius_new, color='black', fc="None")    
    cm = plt.cm.get_cmap('rainbow')

    mycmap = distances
    
    plt.gca().add_patch(circle)
    plt.plot(0, 0, 'x', c=(0, 0, 0), ms=4)

    if title_name:
        # plt.title(title_name, fontsize=fs)
        plt.scatter(coordinates[:, 0], coordinates[:, 1], c=mycmap, s=ms, cmap=cm)
        plt.plot(coordinates[cell, 0], coordinates[cell, 1], 'd', c='red')

        plt.plot(0, 0, 'x', c=(1, 1, 1), ms=4)    
        plt.axis('off')
        plt.axis('equal')        

        plt.axis('off')
        plt.axis('equal')
        cb = plt.colorbar()
        cb.ax.tick_params(labelsize=fs) 

        if file_name:
            plt.savefig(file_name + '.png', format='png', dpi=600)

        plt.show()
        plt.close(fig)

def pseudtotime_comparison(adata, col_names, n_branchings=2):
    sc.tl.dpt(adata, n_branchings=n_branchings)
    diffpt = np.array(adata.obs['dpt_pseudotime'])
    diffpt[diffpt == np.inf] = 1.2
    model.plot('rot', labels=np.array(adata.obs['dpt_groups']), 
           file_name=fout + '_dpt', title_name='Poincaré map', zoom=None, bbox=(1.1, 0.8))
    
    real_time = np.zeros(len(true_lineages))
    for branch in np.unique(true_lineages):
        idx = np.where(true_lineages == branch)[0]
        real_time[idx] = idx - idx[0]

    model.plot_pseudotime(data, col_names, true_labels, file_name=fout + '_realtime', 
                          fs=8, idx=[], pm_pseudotime=real_time)
    
    model.plot_pseudotime(data, col_names, true_labels, file_name=fout + '_poincare', fs=8, idx=[], 
                          pm_pseudotime=model.distances[model.iroot])
    
    model.plot_pseudotime(data, col_names, true_labels, 
                          file_name=fout + '_diffuion', fs=8, idx=[], pm_pseudotime=diffpt)
    
    return real_time, diffpt, model.distances[model.iroot]

def plotBenchamrks(adata, true_labels, fname_benchmark, pl_size=2.4, n1=2, n2=4, ms=3, fs=9, coldict=None, methods=['X_pca', 'X_umap', 'X_draw_graph_fa']):
    labels_order=np.unique(true_labels)
    if coldict is None:
        coldict = dict(zip(labels_order, colors_palette[:len(labels_order)]))

    fig, axs = plt.subplots(n1, n2, sharex=False, sharey=False, figsize=(n2*pl_size, n1*pl_size))
    methods=['X_pca', 'X_tsne', 'X_umap', 'X_diffmap', 'X_draw_graph_fa', 'X_phate', 'X_pacmap', 'X_scdhmap']
    title_name_dict = {'X_pca': 'PCA',
                        'X_tsne': 't-SNE',  
                        'X_umap': 'UMAP', 
                        'X_diffmap': 'DiffusionMaps', 
                        'X_draw_graph_fa': 'ForceAtlas2',
                        'X_phate': 'PHATE',
                        'X_pacmap':'PacMAP',
                        'X_scdhmap':'scDHMAP'}

    l=0
    for i in range(n1):
        for j in range(n2):
            if l < len(methods):
                method=methods[l]
                title_name=title_name_dict[method]
                axs_names=['x1', 'x2']
                if method == 'X_diffmap':
                    x=adata.obsm[method]
                else:
                    x=adata.obsm[method]
                idx = np.random.permutation(len(x))
                df = pd.DataFrame(x[idx, :], columns=axs_names)
                df['labels'] = true_labels[idx]
                axs[i, j].set_title(title_name, fontsize=fs)
                axs[i, j].axis('equal')
                axs[i, j].grid('off')
                axs[i, j].set_xticks([])
                axs[i, j].set_yticks([])
                sns.scatterplot(x=axs_names[0], y=axs_names[1], hue='labels',
                                        hue_order=labels_order,
                                        palette=coldict,
                                        alpha=1.0, edgecolor="none",
                                        data=df, ax=axs[i, j], s=ms)
                axs[i, j].set_xlabel(axs_names[0], fontsize=fs)
                axs[i, j].set_ylabel(axs_names[1], fontsize=fs)
                axs[i, j].legend_.remove()
                fig.tight_layout()
            else:
                axs[i, j].axis('off')
                axs[i, j].grid('off')
                axs[i, j].yaxis.set_tick_params(labelsize=fs)
                axs[i, j].xaxis.set_tick_params(labelsize=fs)
            l += 1
    fig.tight_layout()        
    plt.savefig(fname_benchmark + 'benchmarks.png', format='png', dpi=600, bbox_inches='tight')
    plt.show()


class PoincareMaps:
    def __init__(self, coordinates, model_name='dhv', cpalette=None):
        self.coordinates = coordinates
        self.distances = None       
        self.radius = np.sqrt(coordinates[:,0]**2 + coordinates[:,1]**2)
        self.iroot = np.argmin(self.radius)
        self.labels_pos = None
        self.model_name = model_name
        if cpalette is None:
            self.colors_palette = colors_palette
        else:
            self.colors_palette = cpalette
        
    def find_iroot(self, labels, head_name):
        head_idx = np.where(labels == head_name)[0]
        if len(head_idx) > 1:
            D = self.distances[head_idx, :][head_idx]
            self.iroot = head_idx[np.argmin(D.sum(axis=0))]
        else:
            self.iroot = head_idx[0]            

    def get_distances(self):
        if self.model_name == 'dhv':
            self.distances = poincare_distance_dhv(th.DoubleTensor(self.coordinates)).numpy()
        if self.model_name == 'dv':
            self.distances = poincare_distance_dv(th.DoubleTensor(self.coordinates)).numpy()
        else:
            self.distances = poincare_distance_poinmaps(th.DoubleTensor(self.coordinates)).numpy()

    def rotate(self):
        self.coordinates_rotated = poincare_translation(-self.coordinates[self.iroot, :], self.coordinates)     

    def plot(self, pm_type='ori', labels=None, 
        labels_name='labels', print_labels=False, labels_text=None,
        labels_order=None, coldict=None, file_name=None, title_name=None, alpha=1.0,
        zoom=None, show=True, d1=4.5, d2=4.0, fs=9, ms=20, bbox=(1.3, 0.7), u=None, v=None, leg=True, ft='png'):                            
        if pm_type == 'ori':
            coordinates = self.coordinates
            radius_new = np.sqrt(coordinates[:,0]**2 + coordinates[:,1]**2).max()
        
        elif pm_type == 'rot':
            coordinates = self.coordinates_rotated
            radius_new = np.sqrt(coordinates[:,0]**2 + coordinates[:,1]**2).max()

        if labels_order is None:
            labels_order = np.unique(labels)

        if not (zoom is None):
            if zoom == 1:
                coordinates = np.array(linear_scale(coordinates))
            else:           
                radius = np.sqrt(coordinates[:,0]**2 + coordinates[:,1]**2)
                idx_zoom = np.where(radius <= 1/zoom)[0]
                coordinates = coordinates[idx_zoom, :]
                coordinates = np.array(linear_scale(coordinates))    
                coordinates[np.isnan(coordinates)] = 0
                labels = labels[idx_zoom]

        
        self.labels_pos = plot_poincare_disc(coordinates, radius=radius_new, title_name=title_name, model_name=self.model_name,
            print_labels=print_labels, labels_text=labels_text,
            labels=labels, labels_name=labels_name, labels_order=labels_order, labels_pos = self.labels_pos,
                       file_name=file_name, coldict=coldict, u=u, v=v, alpha=alpha,
                       d1=d1, d2=d2, fs=fs, ms=ms, col_palette=self.colors_palette, bbox=bbox, leg=leg, ft=ft)

    def detect_lineages(self, n_lin=2, clustering_name='spectral', k=15, rotated=False):
        pc_proj = []

        if rotated:
            x = self.coordinates_rotated
        else:
            x = self.coordinates
        
        for i in range(len(x)):
            pc_proj.append(get_projected_coordinates(x[i]))
        
        if clustering_name == 'spectral':
            clustering = SpectralClustering(n_clusters=n_lin, eigen_solver='arpack', affinity="nearest_neighbors", n_neighbors=k).fit(pc_proj)      
        elif clustering_name == 'dbs':
            clustering = DBSCAN(eps=1/180, min_samples=10).fit(pc_proj)
        elif clustering_name == 'kmeans':
            clustering = KMeans(n_clusters=n_lin).fit(pc_proj)
        else:
            clustering = AgglomerativeClustering(linkage='ward', n_clusters=n_lin).fit(pc_proj)

        self.lineages = clustering.labels_

    def detect_cluster(self, n_clusters=2, clustering_name='spectral', k=15):
        if clustering_name == 'spectral':
            similarity = np.exp(-self.distances**2)
            clustering = SpectralClustering(n_clusters=n_clusters, eigen_solver='arpack', affinity='precomputed', n_neighbors=k).fit(similarity)
        else:
            clustering = AgglomerativeClustering(linkage='average', n_clusters=n_clusters, affinity='precomputed').fit(self.distances**2)

        self.clusters = clustering.labels_

    def plot_distances(self, cell=None, pm_type='rot', ss=10, eps=4.0, file_name=None, title_name=None, idx_zoom=None, show=False, fs=8, ms=3):
        if cell is None:
            cell = self.iroot
            
        if pm_type == 'ori':
            coordinates = self.coordinates
        elif pm_type == 'rot':
            coordinates = self.coordinates_rotated

        fig = plt.figure(figsize=(5, 5))
        circle = plt.Circle((0, 0), radius=1, color='black', fc="None")    
        cm = plt.cm.get_cmap('rainbow')
        
        mycmap = np.minimum(list(self.distances[:, cell]), eps)
        
        plt.gca().add_patch(circle)
        plt.plot(0, 0, 'x', c=(0, 0, 0), ms=ms)
        if title_name:
            plt.title(title_name, fontsize=fs)
        plt.scatter(coordinates[:, 0], coordinates[:, 1], c=mycmap, s=ss, cmap=cm)
        plt.plot(coordinates[cell, 0], coordinates[cell, 1], 'd', c='red')

        plt.plot(0, 0, 'x', c=(1, 1, 1), ms=ms)    
        plt.axis('off')
        plt.axis('equal')        
        
        plt.axis('off')
        plt.axis('equal')
        cb = plt.colorbar()
        cb.ax.tick_params(labelsize=fs) 


        if file_name:
            plt.savefig(file_name + '.png', format='png', dpi=600)


    def plot_distances_between_clusters(self, labels, pm_type='rot', eps = 4.0, file_name=None, fs=9):
        if pm_type == 'ori':
            poincare_coord = self.coordinates
        elif pm_type == 'rot':
            poincare_coord = self.coordinates_rotated

        cell_list = np.unique(labels)
        n = len(labels)    
        
        n_plt = len(cell_list)
        # n2 = int(np.sqrt(n_plt))
        n2 = 3
        n1 = n_plt // n2
        
        if n1*n2 < n_plt:
            n1 += 1

        if n1 == 1:
            n1 = 2
        
        if n2 == 1:
            n2 = 2

        f, axs = plt.subplots(n1, n2, sharey=False, figsize=(n2*3, n1*3))

        l=0
        for i in range(n1):
            for j in range(n2):
                if l < n_plt:
                    cell = np.random.choice(np.where(labels == cell_list[l])[0])

                    mycmap = self.distances[cell]

                    circle = plt.Circle((0, 0), radius=1, color='black', fc="None")
                    axs[i, j].add_patch(circle)
                    axs[i, j].axis('off')
                    axs[i, j].axis('equal')
                    axs[i, j].plot(0, 0, 'x', c=(0, 0, 0), ms=6)
                    cm = plt.cm.get_cmap('rainbow')
                    sc = axs[i, j].scatter(poincare_coord[:, 0], poincare_coord[:, 1], c=mycmap, s=15, cmap=cm)    
                    axs[i, j].set_title(cell_list[l], fontsize=fs)
                    axs[i, j].plot(poincare_coord[cell, 0], poincare_coord[cell, 1], 'd', c='red')
                else:
                    axs[i, j].axis('off')
                    axs[i, j].axis('equal')
                l+=1
        if file_name:
            plt.savefig(file_name + '.png', format='png', dpi=600)

    def plot_markers(self, data, markesnames, pm_type='rot', file_name=None, fs=8, sc=3):
        if pm_type == 'ori':
            poincare_coord = self.coordinates
        elif pm_type == 'rot':
            poincare_coord = self.coordinates_rotated

        n_plt = np.size(data, 1)    
        
        # n2 = int(np.sqrt(n_plt))
        n2 = 3
        n1 = n_plt // n2
        
        if n1*n2 < n_plt:
            n1 += 1

        if n1 == 1:
            n1 = 2

        if n2 == 1:
            n2 = 2 
        
        f, axs = plt.subplots(n1, n2, sharey=False, figsize=(n2*sc, n1*sc))

        cm = plt.cm.get_cmap('jet')

        l=0
        for i in range(n1):
            for j in range(n2):            
                axs[i, j].axis('off')
                axs[i, j].axis('equal')
                if l < n_plt:
                    circle = plt.Circle((0, 0), radius=1, color='black', fc="none")
                    axs[i, j].add_patch(circle)
                    axs[i, j].plot(0, 0, 'x', c=(0, 0, 0), ms=3)
                    axs[i, j].axis('equal')
                    sc = axs[i, j].scatter(poincare_coord[:, 0], poincare_coord[:, 1], c=data[:, l], s=5, cmap=cm)    
                    axs[i, j].set_title(markesnames[l], fontsize=fs)
                    plt.colorbar(sc, ax=axs[i,j])

                    if l == n_plt:
                        axs[i, j].legend(np.unique(labels), loc='center left', bbox_to_anchor=(1, 0.5), fontsize=fs)
                l+=1

        if file_name:
            plt.savefig(file_name + '.png', format='png', dpi=600)


import argparse

parser = argparse.ArgumentParser(description="*** author")
parser.add_argument('--name', type=str, default='digits_T', )
parser.add_argument("--offline", type=int, default=0)
parser.add_argument("--project_name", type=str, default="test")
parser.add_argument("--data_path", type=str, default="./data")
parser.add_argument("--knn", type=int, default=5)
parser.add_argument("--sigma", type=float, default=1.0)
parser.add_argument("--n_components", type=int, default=10)
parser.add_argument('--method', type=str, default='pca', choices=['pca', 'tsne', 'umap', 'phate', 'diffmap', 'forceatlas2', 'ivis', 'poinmaps', 'scphere_wn', 'scdhmap'])

# data set param
parser.add_argument(
    "--data_name",
    type=str,
    default="Olsson",
    choices=[
        "Olsson",
    ],
)

# baseline param
parser.add_argument(
    "--metric",
    type=str,
    default="euclidean",
)
parser.add_argument('--perplexity', type=int, default=10)
parser.add_argument('--min_dist', type=float, default=0.1)

args = pl.Trainer.add_argparse_args(parser)
args = args.parse_args()

dataset_f = getattr(data_base, args.data_name + "Dataset")

data_train = dataset_f(
    data_name=args.data_name,
    knn = args.knn,
    sigma = args.sigma,
    n_components = args.n_components,
    train=True,
    datapath=args.data_path,
)

data_name = args.data_name
data = data_train.data.numpy().reshape(data_train.data.shape[0], -1)
label = np.array(data_train.label)
sadata = data_train.sadata
true_labels = np.array(sadata.obs['celltype'])
ms = 5
fs = 10
col_dict = None
dv_coord = None

sc.settings.set_figure_params(dpi=600)

#### Visualization and Structure Preservation

if data_name == 'Olsson':
    sadata.obsm['X_pca'] = np.load('logs/log_Olsson_baseline/path_pca/latent.npy')
    sadata.obsm['X_tsne'] = np.load('logs/log_Olsson_baseline/path_tsne_40/latent.npy')
    sadata.obsm['X_umap'] = np.load('logs/log_Olsson_baseline/path_umap_40_0.7/latent.npy')
    sadata.obsm['X_diffmap'] = np.load('logs/log_Olsson_baseline/path_diffmap_10/latent.npy')
    sadata.obsm['X_draw_graph_fa'] = np.load('logs/log_Olsson_baseline/path_forceatlas2_10/latent.npy')
    sadata.obsm['X_phate'] = np.load('logs/log_Olsson_baseline/path_phate/latent.npy')
    sadata.obsm['X_pacmap'] = np.load('logs/log_Olsson_baseline/path_pacmap_40_0.6_1.5/latent.npy')
    sadata.obsm['X_scdhmap'] = np.load('logs/log_Olsson_baseline/path_scdhmap/latent.npy')
    dv_coord = np.load('logs/log_Olsson_dv_poin/path_euclidean_poin_dist_mobiusm_v2_dsml_100.0_0.001_-1.0_5_eu_poin_2_100_2.0_Adam_leaky_relu_5_1.0_1.0_1000.0_no_no_dmt_aug_no_0.01_0.0_0.0_100.0_500_300_100/latent.npy')
    poincare_coord = np.load('logs/log_Olsson_baseline/path_poincaremaps_15_2.0_2.0/latent.npy')
    dhv_coord = np.load('logs/log_Olsson_new/path_Ours/latent.npy')

    # ablation 
    # dhv_coord = np.load('logs/log_Olsson_ablation/log_Olsson_global_loss/path_Ours/latent_5_30_0.01_0.1_1.0_10.0_10.npy')
    # dhv_coord = np.load('logs/log_Olsson_ablation/log_Olsson_hnn/path_Ours/latent_10_20_0.1_0.01_1.0_0.1_10.npy')
    # dhv_coord = np.load('logs/log_Olsson_ablation/log_Olsson_local_loss/path_Ours/latent_5_10_0.03_0.03_1.0_10.0_10.npy')
    # dhv_coord = np.load('logs/log_Olsson_ablation/log_Olsson_data_augmentation/path_Ours/latent_20_10_0.01_0.01_1.0_10.0_10.npy')

if data_name == 'MyeloidProgenitors' or data_name == 'Krumsiek11' or data_name == 'ToggleSwitch':
    root = 'root'
elif data_name == "Moignard2015":
    root = "PS"
elif data_name == 'Olsson':
    root = "HSPC-1"
    col_dict = {'Eryth': '#1F77B4',
                    'Gran': '#FF7F0E',
                    'HSPC-1': '#2CA02C',
                    'HSPC-2': '#D62728',
                    'MDP': '#9467BD',
                    'Meg': '#8C564B',
                    'Mono': '#E377C2',
                    'Multi-Lin': '#BCBD22',
                    'Myelocyte': '#17BECF'}
elif data_name == 'Paulpca' or data_name == 'PaulPcaKnn' or data_name == 'Paul':
    root  = "root"
    col_dict = {'12Baso': '#0570b0', '13Baso': '#034e7b',
        '11DC': '#ffff33', 
        '18Eos': '#2CA02C', 
        '1Ery': '#fed976', '2Ery': '#feb24c', '3Ery': '#fd8d3c', '4Ery': '#fc4e2a', '5Ery': '#e31a1c', '6Ery': '#b10026',
        '9GMP': '#999999', '10GMP': '#4d4d4d',
        '19Lymph': '#35978f', 
        '7MEP': '#E377C2', 
        '8Mk': '#BCBD22', 
        '14Mo': '#4eb3d3', '15Mo': '#7bccc4',
        '16Neu': '#6a51a3','17Neu': '#3f007d',
        'root': '#000000'}
elif data_name == 'Planaria':
    root = 'neoblast 1'
    col_dict = {'neoblast 1': '#CCCCCC',
	  'neoblast 2': '#7f7f7f',
	  'neoblast 3': '#E6E6E6',
	  'neoblast 4': '#D6D6D6',
	  'neoblast 5': '#C7C7C7',
	  'neoblast 6': '#B8B8B8',
	  'neoblast 7': '#A8A8A8',
	  'neoblast 8': '#999999',
	  'neoblast 9': '#8A8A8A',
	  'neoblast 10':  '#7A7A7A',
	  'neoblast 11':  '#6B6B6B',
	  'neoblast 12':  '#5C5C5C',
	  'neoblast 13':  '#4D4D4D',
	  'epidermis DVb neoblast': 'lightsteelblue',
	  'pharynx cell type progenitors':  'slategray',
	  'spp-11+ neurons':  '#CC4C02',
	  'npp-18+ neurons':  '#EC7014',
	  'otf+ cells 1': '#993404',
	  'ChAT neurons 1': '#FEC44F',
	  'neural progenitors': '#FFF7BC',
	  'otf+ cells 2': '#662506',
	  'cav-1+ neurons': '#eec900',
	  'GABA neurons': '#FEE391',
	  'ChAT neurons 2': '#FE9929',
	  'muscle body':  'firebrick',
	  'muscle pharynx': '#CD5C5C',
	  'muscle progenitors': '#FF6347',
	  'secretory 1':  'mediumpurple',
	  'secretory 3':  'purple',
	  'secretory 4':  '#CBC9E2',
	  'secretory 2':  '#551a8b',
	  'early epidermal progenitors':  '#9ECAE1',
	  'epidermal neoblasts':  '#C6DBEF',
	  'activated early epidermal progenitors':  'lightblue',
	  'late epidermal progenitors 2': '#4292C6',
	  'late epidermal progenitors 1': '#6BAED6',
	  'epidermis DVb':  'dodgerblue',
	  'epidermis':  '#2171B5',
	  'pharynx cell type': 'royalblue',
	  'protonephridia': 'pink',
	  'ldlrr-1+ parenchymal cells': '#d02090',
	  'phagocytes': 'forestgreen',
	  'aqp+ parenchymal cells': '#cd96cd',
	  'pigment': '#cd6889',
	  'pgrn+ parenchymal cells':  'mediumorchid',
	  'psap+ parenchymal cells':  'deeppink',
	  'glia': '#cd69c9',
	  'goblet cells': 'yellow',
	  'parenchymal progenitors':  'hotpink',
	  'psd+ cells': 'darkolivegreen',
	  'gut progenitors':  'limegreen',
	  'branchNe': '#4292c6', 'neutrophil': '#08306b',
			'branchMo': '#9e9ac8', 'monocyte': '#54278f',
			'branchEr': '#fc9272', 'erythrocyt': '#cb181d',
			'megakaryoc': '#006d2c',  'branchMe': '#74c476',
			'proghead': '#525252', 'root': '#000000',
			'interpolation': '#525252',
			'Eryth': '#1F77B4',
			'Gran': '#FF7F0E',
			'HSPC-1': '#2CA02C',
			'HSPC-2': '#D62728',
			'MDP': '#9467BD',
			'Meg': '#8C564B',
			'Mono': '#E377C2',
			'Multi-Lin': '#BCBD22',
			'Myelocyte': '#17BECF',
		    '12Baso': '#0570b0', '13Baso': '#034e7b',
           '11DC': '#ffff33', 
           '18Eos': '#2CA02C', 
           '1Ery': '#fed976', '2Ery': '#feb24c', '3Ery': '#fd8d3c', '4Ery': '#fc4e2a', '5Ery': '#e31a1c', '6Ery': '#b10026',
           '9GMP': '#999999', '10GMP': '#4d4d4d',
           '19Lymph': '#35978f', 
           '7MEP': '#E377C2', 
           '8Mk': '#BCBD22', 
           '14Mo': '#4eb3d3', '15Mo': '#7bccc4',
           '16Neu': '#6a51a3','17Neu': '#3f007d',
           'root': '#000000'}
elif data_name == 'UCEPIbc':
    root = 'Stem'
elif data_name == 'UCIMMbc':
    col_dict = {'CD4+ Activated Fos-hi': '#1f77b4',
        'CD4+ Activated Fos-lo': '#ff7f0e',
        'CD4+ Memory': '#2ca02c',
        'CD4+ PD1+': '#d62728',
        'CD69+ Mast': '#9467bd',
        'CD69- Mast': '#8c564b',
        'CD8+ IELs': '#e377c2',
        'CD8+ IL17+': '#7f7f7f',
        'CD8+ LP': '#bcbd22',
        'Cycling B':  '#17becf',
        'Cycling Monocytes':  '#aec7e8',
        'Cycling T':  '#ffbb78',
        'DC1':  '#98df8a',
        'DC2': '#ff9896',
        'Follicular':  '#c5b0d5',
        'GC':  '#c49c94',
        'ILCs':  '#f7b6d2',
        'Inflammatory Monocytes': '#c7c7c7',
        'MT-hi': '#dbdb8d',
        'Macrophages': '#9edae5',
        'NKs': '#393b79',
        'Plasma': '#637939',
        'Tregs': '#8c6d31',
    }
elif data_name == 'CELEGANPCA100':
    root = 'Germline'
    col_dict = {'ABarpaaa_lineage': '#91003f', # embryonic lineage
                        'Germline': '#7f2704', 
                        # Somatic gonad precursor cell
                        'Z1_Z4': '#800026',
                        # Two embryonic hypodermal cells that may provide a scaffold for the early organization of ventral bodywall muscles
                        'XXX': '#fb8072',
                        'Ciliated_amphid_neuron': '#c51b8a','Ciliated_non_amphid_neuron': '#fa9fb5',
                        # immune
                        'Coelomocyte': '#ffff33', 'T': '#54278f',
                        # Exceratory
                        'Excretory_cell': '#004529', 
                        'Excretory_cell_parent': '#006837',
                        'Excretory_duct_and_pore': '#238443', 
                        'Parent_of_exc_duct_pore_DB_1_3': '#41ab5d',
                        'Excretory_gland': '#78c679',
                        'Parent_of_exc_gland_AVK': '#addd8e',
                        'Rectal_cell': '#d9f0a3',
                        'Rectal_gland': '#f7fcb9',
                        'Intestine': '#7fcdbb',
                        # esophagus, crop, gizzard (usually) and intestine
                        'Pharyngeal_gland': '#fed976', 
                        'Pharyngeal_intestinal_valve': '#feb24c', 
                        'Pharyngeal_marginal_cell': '#fd8d3c', 
                        'Pharyngeal_muscle': '#fc4e2a', 
                        'Pharyngeal_neuron': '#e31a1c',
                        # hypodermis (epithelial)
                        'Parent_of_hyp1V_and_ant_arc_V': '#a8ddb5', 
                        'hyp1V_and_ant_arc_V': '#ccebc5',
                        'Hypodermis': '#253494', 
                        'Seam_cell': '#225ea8',
                        'Arcade_cell': '#1d91c0',
                        # set of six cells that form a thin cylindrical sheet between pharynx and ring neuropile
                        'GLR': '#1f78b4',
                        # Glia, also called glial cells or neuroglia, are non-neuronal cells in the central nervous system
                        'Glia': '#377eb8',
                        # head mesodermal cell: the middle layer of cells or tissues of an embryo
                        'Body_wall_muscle': '#9e9ac8',
                        'hmc': '#54278f', 
                        'hmc_and_homolog': '#02818a', 
                        'hmc_homolog': '#bcbddc',
                        'Intestinal_and_rectal_muscle': '#41b6c4',
                        # Postembryonic mesoblast: the mesoderm of an embryo in its earliest stages.
                        'M_cell': '#3f007d',
                        # pharyngeal gland cel
                        'G2_and_W_blasts': '#abdda4', 
                        'unannotated': '#969696'}
    true_time_labels = np.array(sadata.obs['embryo_time'])
    root = '< 100'
    cdict = {
        'red':   ((0.0, 1.0, 1.0),  # 红色起始
                (0.25, 1.0, 1.0), # 浅红色
                (0.5, 1.0, 1.0),  # 浅橙色
                (0.75, 0.0, 0.0), # 浅蓝色过渡，红色结束
                (1.0, 0.0, 0.0)), # 蓝色
        'green': ((0.0, 0.0, 0.0),  # 红色起始时绿色为0
                (0.25, 0.2, 0.2),# 浅红色，绿色略增
                (0.5, 0.6, 0.6), # 浅橙色，绿色适量
                (0.75, 0.7, 0.7),# 浅蓝色，绿色增强
                (1.0, 0.0, 0.0)),# 蓝色结束时绿色为0
        'blue':  ((0.0, 0.0, 0.0),  # 红色起始时蓝色为0
                (0.25, 0.3, 0.3),# 浅红色，蓝色略增
                (0.5, 0.4, 0.4), # 浅橙色，蓝色略增
                (0.75, 1.0, 1.0),# 浅蓝色，蓝色增强
                (1.0, 1.0, 1.0)) # 蓝色饱和
    }
    from matplotlib.colors import LinearSegmentedColormap
    red_to_blue = LinearSegmentedColormap('CustomColormap', segmentdata=cdict)
    time_segments = ['< 100', '100-130', '130-170', '170-210', '210-270', 
                    '270-330', '330-390', '390-450', '450-510', '510-580', 
                    '580-650', '> 650']
    colors = red_to_blue(np.linspace(0, 1, len(time_segments)))
    import matplotlib
    col_dict = {time_segments[i]: matplotlib.colors.rgb2hex(colors[i]) for i in range(len(time_segments))}
    true_labels = true_time_labels

iroot = init_scanpy_root(data, root, true_labels)

# Eu methods
plotBenchamrks(sadata, true_labels, f"benchmarks/{data_name}", ms=ms, coldict=col_dict)

### Poincaré map
fout = f"benchmarks/{data_name + '_Poincaré map'}"
model = PoincareMaps(poincare_coord, model_name='Poincaré map')
model.plot('ori', labels=true_labels, file_name=fout + '_ori', 
        title_name='Poincaré map', zoom=None, bbox=(1.0, 0.7), ms=ms, coldict=col_dict, leg=False)
model.iroot = iroot
model.rotate()
model.plot('rot', labels=true_labels, file_name=fout + '_rotated', 
        title_name='Poincaré map rotated', zoom=None, show=True, bbox=(1.0, 0.7), ms=ms, coldict=col_dict, leg=False)

### DV
if dv_coord is not None:
    fout = f"benchmarks/{data_name + '_DV_poin'}"
    model = PoincareMaps(dv_coord, model_name='dv')
    model.plot('ori', labels=true_labels, file_name=fout + '_ori', 
            title_name='DV', zoom=None, bbox=(1.0, 0.7), ms=ms, coldict=col_dict, leg=True)
    model.iroot = iroot
    model.rotate()
    model.plot('rot', labels=true_labels, file_name=fout + '_rotated', 
            title_name='DV rotated', zoom=None, show=True, bbox=(1.0, 0.7), ms=ms, coldict=col_dict, leg=True)

# PoincareDMT
fout = f"benchmarks/{data_name + '_DHV'}"
model = PoincareMaps(dhv_coord, model_name='dhv')
model.plot('ori', labels=true_labels, file_name=fout + '_ori', 
        title_name='DHV', zoom=None, bbox=(1.0, 0.7), ms=ms, coldict=col_dict, leg=True)
model.iroot = iroot
model.rotate()
model.plot('rot', labels=true_labels, file_name=fout + '_rotated', 
        title_name='DHV rotated', zoom=None, show=True, bbox=(1.0, 0.7), ms=ms, coldict=col_dict, leg=True)


### pseudotime
sc.pp.neighbors(sadata, n_neighbors=10)
sc.pp.neighbors(sadata, n_neighbors=10, method='gauss', knn=False)
sc.tl.diffmap(sadata)
sc.tl.dpt(sadata, n_branchings=1)

### Analysis
np.random.seed(seed=2018)
true_labels_ori = np.copy(true_labels)
model_name = args.data_name
col_dict = None
fin = f"datasets/{model_name}"
fout = f"benchmarks/{model_name}"
model = PoincareMaps(dhv_coord, model_name='dhv')
model.get_distances()

# plot poincare rorated
iroot = random.choice(np.where(true_labels == root)[0])
model.iroot = iroot
model.rotate()

sadata.uns['iroot'] = iroot
plot_dp_on_poincare(model.coordinates_rotated, 
                    model.distances[model.iroot, :], cell=model.iroot, d1=10, d2=10, ms=ms, 
                    title_name='poincaredmt_pseudotime', file_name=fout+'_poincaredmt_pseudotime_rotated_root', fs=fs)
