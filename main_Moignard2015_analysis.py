import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import numpy as np
import pandas as pd
import scanpy as sc
import pandas as pd
import numpy as np
import pytorch_lightning as pl
from dataloader import data_base
import torch as th
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.cluster import *
from sklearn import metrics
import random
import itertools

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
        #         i = np.random.choice(np.where(labels == l)[0])
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

    # ax.set_ylim([-1.01, 1.01])
    # ax.set_xlim([-1.01, 1.01])

    plt.tight_layout()

    if file_name:
        if ft == 'png':
            plt.savefig(file_name + '.' + ft, format=ft, dpi=600)
        else:
            plt.savefig(file_name + '.' + ft, format=ft)

    return labels_pos

def get_projected_coordinates(u):
#     ax + by = 0
#  y = cx, where c = -a/b = y/x, if b != 0
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
#     classes = [ str(l) for l in np.unique(model.lineages)]
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
#     plt.title(title, fontsize=fs)
#     plt.colorbar()
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

def plot_dp_on_poincare(coordinates, distances, cell=0, fs=9, title_name='None', file_name=None, d1=3.5, d2=3.1):
    fig = plt.figure(figsize=(d1, d2))
    radius_new = np.sqrt(coordinates[:,0]**2 + coordinates[:,1]**2).max()
    circle = plt.Circle((0, 0), radius=radius_new, color='black', fc="None")    
    cm = plt.cm.get_cmap('rainbow')

    mycmap = distances
    
    plt.gca().add_patch(circle)
    plt.plot(0, 0, 'x', c=(0, 0, 0), ms=4)

    if title_name:
        # plt.title(title_name, fontsize=fs)
        plt.scatter(coordinates[:, 0], coordinates[:, 1], c=mycmap, s=0.1, cmap=cm)
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

def plotBenchamrks(adata, true_labels, labels_order, fname_benchmark, pl_size=2.4, n1=2, n2=3, ms=3, fs=9, coldict=None, methods=['X_pca', 'X_umap', 'X_draw_graph_fa']):
    # labels_order=np.unique(true_labels)
    if coldict is None:
        coldict = dict(zip(labels_order, colors_palette[:len(labels_order)]))

    fig, axs = plt.subplots(n1, n2, sharex=False, sharey=False, figsize=(n2*pl_size, n1*pl_size))
    methods=['X_pca', 'X_tsne', 'X_umap', 'X_diffmap', 'X_draw_graph_fa', 'X_phate']
    title_name_dict = {'X_pca': 'PCA',
                        'X_tsne': 't-SNE',  
                        'X_umap': 'UMAP', 
                        'X_diffmap': 'DiffusionMaps', 
                        'X_draw_graph_fa': 'ForceAtlas2',
                        'X_phate': 'PHATE'}

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

    def plot_cluster_markers(self, data, label, cluster, markesnames, pm_type='rot', file_name=None, fs=8, sc=3):
        if pm_type == 'ori':
            poincare_coord = self.coordinates
        elif pm_type == 'rot':
            poincare_coord = self.coordinates_rotated

        data = data[label == cluster]
        poincare_coord = poincare_coord[label == cluster]

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
parser.add_argument("--knn", type=int, default=10)
parser.add_argument("--sigma", type=float, default=2.0)
parser.add_argument('--method', type=str, default='pca', choices=['pca', 'tsne', 'umap', 'phate', 'diffmap', 'forceatlas2', 'ivis', 'poinmaps', 'scphere_wn', 'scdhmap'])

# data set param
parser.add_argument(
    "--data_name",
    type=str,
    default="Moignard2015",
    choices=[
        "Moignard2015",
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
    train=True,
    datapath=args.data_path,
)

data = data_train.data.numpy().reshape(data_train.data.shape[0], -1)
label = np.array(data_train.label)
labelstr = np.array(data_train.label_train_str)
sadata = data_train.sadata
true_labels = np.array(sadata.obs['celltype'])
col_names = data_train.col_names
data_name = args.data_name
ms = 5
fs = 9
col_dict = None
dv_coord = None

sc.settings.set_figure_params(dpi=600)
# sc.pp.neighbors(sadata, n_neighbors=10)
sc.pp.neighbors(sadata, n_neighbors=10, method='gauss', knn=False)
sc.tl.diffmap(sadata)
sc.tl.dpt(sadata, n_branchings=1)

#### Visualization and Structure Preservation
if data_name == 'Moignard2015':
    sadata.obsm['X_pca'] = np.load('logs/log_Moignard2015_baseline/path_pca/latent.npy')
    sadata.obsm['X_tsne'] = np.load('logs/log_Moignard2015_baseline/path_tsne_40/latent.npy')
    sadata.obsm['X_umap'] = np.load('logs/log_Moignard2015_baseline/path_umap_30_0.5/latent.npy')
    sadata.obsm['X_diffmap'] = np.load('logs/log_Moignard2015_baseline/path_diffmap_40/latent.npy')
    sadata.obsm['X_draw_graph_fa'] = np.load('logs/log_Moignard2015_baseline/path_forceatlas2_30/latent.npy')
    sadata.obsm['X_phate'] = np.load('logs/log_Moignard2015_baseline/path_phate/latent.npy')
    poincare_coord = np.load('logs/log_Moignard2015_poin_maps/path_30_1.0_2.0/result/latent.npy')
    dv_coord = np.load('logs/log_Moignard2015_dv_poin/path_euclidean_poin_dist_mobiusm_v2_dsml_100.0_0.001_-1.0_5_eu_poin_2_500_2.0_Adam_leaky_relu_5_1.0_1.0_1000.0_no_no_dmt_aug_no_0.0005_0.0_0.0_100.0_500_300_100/latent_epoch_299.npy')
    dhv_coord = np.load('logs/log_Moignard2015_new/path_Ours/latent.npy')

root = "PS"
# labels_order = ['Meso', '4SFG', '4SG', 'HF', 'NP', 'PS']
true_labels_order = ['4SFG', '4SG', 'HF', 'NP', 'PS']
color_dict = {'PS': '#005a32',
            'NP': '#78c679',
            'HF': '#d9f0a3',
            '4SFG': '#08519c',
            '4SG': '#bd0026',
            'meso': '#636363'}

iroot = init_scanpy_root(data, root, true_labels)

#### Eu methods
plotBenchamrks(sadata, true_labels, true_labels_order, f"benchmarks/{data_name}", ms=ms, coldict=color_dict)

### Poincaré map
fout = f"benchmarks/{data_name + '_Poincaré map'}"
model = PoincareMaps(poincare_coord, model_name='Poincaré map')
model.plot('ori', labels=true_labels, file_name=fout + '_ori', 
        title_name='Poincaré map', zoom=None, bbox=(1.1, 0.8), ms=ms, coldict=color_dict)
model.iroot = iroot
model.rotate()
model.plot('rot', labels=true_labels, file_name=fout + '_rotated', 
        title_name='Poincaré map rotated', zoom=None, show=True, bbox=(1.1, 0.8), ms=ms, coldict=color_dict)

### DV
if dv_coord is not None:
    fout = f"benchmarks/{data_name + '_DV_poin'}"
    model = PoincareMaps(dv_coord, model_name='dv')
    model.plot('ori', labels=true_labels, file_name=fout + '_ori', 
            title_name='DV', zoom=None, bbox=(1.1, 0.8), ms=ms, coldict=color_dict)
    model.iroot = iroot
    model.rotate()
    model.plot('rot', labels=true_labels, file_name=fout + '_rotated', 
            title_name='DV rotated', zoom=None, show=True, bbox=(1.1, 0.8), ms=ms, coldict=color_dict)

## DHV
fout = f"benchmarks/{data_name + '_DHV'}"
model = PoincareMaps(dhv_coord, model_name='dhv')
model.plot('ori', labels=true_labels, labels_order=true_labels_order, file_name=fout + '_ori', 
        title_name='DHV', zoom=None, bbox=(1.0, 0.7), ms=ms, coldict=color_dict)
model.iroot = iroot
model.rotate()
model.plot('rot', labels=true_labels, file_name=fout + '_rotated', 
        title_name='DHV rotated', zoom=None, show=True, bbox=(1.0, 0.7), ms=ms, coldict=color_dict)
plot_dp_on_poincare(model.coordinates_rotated, 
                    np.array(sadata.obs['dpt_order']), cell=model.iroot,
                    title_name='dpt_pseudotime', file_name=fout+'_dpt_pseudotime_rotated_PS', fs=9)

### Analysis of Moignard2015
np.random.seed(seed=2018)
find = lambda searchList, elem: [[i for i, x in enumerate(searchList) if x == e][0] for e in elem]
idx_ery = find(col_names, ['HbbbH1', 'Gata1', 'Nfe2', 'Gfi1b', 'Ikaros', 'Myb'])
idx_ery_sample = find(col_names, ['Gfi1b', 'Ikaros', 'Myb'])
idx_endo = find(col_names, ['Erg', 'Sox7', 'Sox17', 'HoxB4', 'Cdh5'])
true_labels_order = ['4SFG', '4SG', 'HF', 'NP', 'PS']
true_labels_ori = np.copy(true_labels)
model_name = 'Moignard2015'
col_dict = None
fin = f"datasets/{model_name}"
fout = f"benchmarks/{model_name}"
model = PoincareMaps(dhv_coord, model_name='dhv')
model.plot('ori', labels=true_labels, file_name=fout + '_ori', 
           title_name='Poincaré map', coldict=color_dict, labels_order=true_labels_order, 
           zoom=None, bbox=(1.0, 0.7), ms=ms)
model.get_distances()

# # markers gene
# print('Hematopoetic genes')
# model.plot_markers(data[:,idx_ery], col_names[idx_ery], 
#                    file_name=fout + '_ery_markers', pm_type='ori', sc=3, fs=9)
# print('Endothelial genes')
# model.plot_markers(data[:,idx_endo], col_names[idx_endo], 
#                    file_name=fout + '_endo_markers', pm_type='ori', sc=3, fs=9)
# model.plot_markers(data, col_names, 
#                    file_name=fout + '_markers_all', 
#                    pm_type='ori', sc=3, fs=9)

# markers gene cluster finding
print('PS genes')
ps_ery = find(col_names, ['Hhex', 'Kdr', 'Gfi1b', 'Fli1', 'Cdh5', 'HbbbH1'])
model.plot_cluster_markers(data[:,ps_ery], true_labels, 'PS', col_names[ps_ery], 
                   file_name=fout + '_ps_markers', pm_type='ori', sc=3, fs=9)

print('HF genes')
ps_ery = find(col_names, ['HbbbH1', 'Gfi1b', 'Myb', 'Sox17', 'Cdh5', 'Meis1'])
model.plot_cluster_markers(data[:,ps_ery], true_labels, 'HF', col_names[ps_ery], 
                   file_name=fout + '_hf_markers', pm_type='ori', sc=3, fs=9)

print('NP genes')
ps_ery = find(col_names, ['Fli1', 'Hhex', 'Cdh5', 'Kdr', 'HbbbH1', 'Cdh1'])
model.plot_cluster_markers(data[:,ps_ery], true_labels, 'NP', col_names[ps_ery], 
                   file_name=fout + '_np_markers', pm_type='ori', sc=3, fs=9)

print('4SG genes')
ps_ery = find(col_names, ['HbbbH1', 'Cdh5', 'Kdr', 'Itga2b', 'Fli1', 'Hhex'])
model.plot_cluster_markers(data[:,ps_ery], true_labels, '4SG', col_names[ps_ery], 
                   file_name=fout + '_4sg_markers', pm_type='ori', sc=3, fs=9)

print('4SFG genes')
ps_ery = find(col_names, ['Cdh5', 'Itga2b', 'HbbbH1', 'Sfpi1', 'Lyl1', 'Meis1'])
model.plot_cluster_markers(data[:,ps_ery], true_labels, '4SFG', col_names[ps_ery], 
                   file_name=fout + '_4sfg_markers', pm_type='ori', sc=3, fs=9)

# clusters
np.random.seed(seed=2018)
title_name='clusters'
model.detect_cluster(n_clusters=5, clustering_name='spectral', k=30)
col_dict_clust = dict(zip(np.unique(model.clusters), colors_palette[:len(np.unique(model.clusters))]))
model.plot('ori', labels=model.clusters, file_name=fout+'_clusters', labels_name='clusters', title_name='DHV', zoom=None, bbox=(1.0, 0.7), ms=ms)
get_confusion_matrix(model.clusters, true_labels_order, true_labels_ori, title=title_name, fname=fout+'_cm')
true_labels = np.copy(true_labels_ori)
meso_name = 1.0
true_labels[model.clusters == meso_name] = 'Meso'

# plot poincare rorated
fname='{0}_{1}'.format(fout, 'rotated_Haghverdi')
model.iroot = 532
model.rotate()
zoom_scale=1.2
color_dict['Meso'] = '#636363'
title_name='rotated_Haghverdi'
model.plot('rot', labels=true_labels_ori, 
           d1=1.5*4.8, d2=1.5*4.6, fs=9, bbox=(1.0, 0.7), ms=ms, 
           file_name=fname, coldict=color_dict)

sadata.uns['iroot'] = 532
sc.pp.neighbors(sadata, n_neighbors=10, method='gauss', knn=False)
sc.tl.diffmap(sadata)
sc.tl.dpt(sadata, n_branchings=1)
model.plot('rot', labels=true_labels_ori, 
           d1=4.0, d2=3.7, fs=9, bbox=(1.0, 0.7), ms=ms,
           file_name=fname + "rotated_hagh", coldict=color_dict)
plot_dp_on_poincare(model.coordinates_rotated, ### without dpt_pseudotime
                    np.array(sadata.obs['dpt_pseudotime']), cell=532,
                    title_name='dpt_pseudotime', file_name=fout+'_dpt_pseudotime_rotated', fs=fs)
plot_dp_on_poincare(model.coordinates_rotated, 
                    model.distances[model.iroot, :], cell=532,
                    title_name='dpt_pseudotime', file_name=fout+'_pmt_pseudotime_rotated', fs=fs)
plot_dp_on_poincare(model.coordinates, ### without dpt_pseudotime
                    np.array(sadata.obs['dpt_pseudotime']), cell=532,
                    title_name='dpt_pseudotime', file_name=fout+'_dpt_pseudotime', fs=fs)
title_name = 'haghverdi_ori'
model.plot_distances(cell=None, pm_type='ori', eps=10.0, file_name=fout + '_' + title_name,
                     title_name=None, idx_zoom=None, show=False, fs=fs)

idx_4 = np.where(model.clusters == meso_name)[0]
idx_not4 = np.where(model.clusters != meso_name)[0]
idx_4_ps = idx_4[np.where(true_labels_ori[idx_4] == 'PS')[0]]
idx_not4_ps = idx_not4[np.where(true_labels_ori[idx_not4] == 'PS')[0]]
tips_n4 = pd.DataFrame(data[idx_not4_ps, :], columns=col_names)
tips_n4['stage'] = true_labels[idx_not4_ps] 
tips = pd.DataFrame(data[idx_4_ps, :], columns=col_names)
tips['stage'] = true_labels[idx_4_ps]
f, ax = plt.subplots(3, 1, figsize=(3, 4), sharex=True)
for i, gene_name in enumerate(['Kdr', 'Runx1', 'Cdh1']):
    ax[i].hist(tips_n4[gene_name], color='#bdbdbd')
    ax[i].hist(tips[gene_name], color=col_dict_clust[meso_name])
#     plt.legend(['others', 'cluster2'])
    # plt.title('Runx1')
    ax[i].grid('off')
    ax[i].xaxis.set_tick_params(labelsize=fs)
    ax[i].yaxis.set_tick_params(labelsize=fs)
    ax[i].set_title(gene_name, fontsize=fs)
fname='{0}/{1}'.format(fin, 'meso_hist')
plt.tight_layout()
plt.savefig(fout + '_mesogenes.png', format='png', dpi=400)

#### update mesodermal cluster
model.plot('ori', labels=true_labels, file_name=fout + '_reannotated', labels_name='clusters', 
           title_name='DHV', zoom=None, bbox=(1.0, 0.7), ms=ms, coldict=color_dict)
model.plot_distances_between_clusters(true_labels, pm_type='ori', eps=3.0, 
                                      file_name=fout + '_distbwclusters', fs=fs)

####  find the right root of the hierarchy
head_idx = np.where(true_labels == 'PS')[0]
D = pairwise_distances(data[head_idx, :], metric='euclidean')
iroot_ps = head_idx[np.argmin(D.sum(axis=0))]
print('iroot_ps = ', iroot_ps)
head_idx = np.where(true_labels == 'PS')[0]
idx_leaf1 = random.choice(np.where(true_labels == '4SFG')[0])
idx_leaf2 = random.choice(np.where(true_labels == '4SG')[0])
idx1 = random.choice(np.where(model.distances[idx_leaf2, head_idx]/np.max(model.distances[idx_leaf2, head_idx]) < 0.4))
print(idx1)
iroot_ps = head_idx[idx1]
fname='{0}_{1}'.format(fout, 'rotated_PS')
model.iroot = 2080
model.rotate()
zoom_scale=1.2
title_name='rotation'
model.plot('rot', labels=true_labels_ori, 
           d1=4.0, d2=3.7, fs=9, bbox=(1.0, 0.7), ms=ms,
           file_name=fname, coldict=color_dict)
sadata.uns['iroot'] = model.iroot
sc.tl.dpt(sadata, n_branchings=1)
plot_dp_on_poincare(model.coordinates_rotated, 
                    np.array(sadata.obs['dpt_pseudotime']), cell=model.iroot,
                    title_name='dpt_pseudotime', file_name=fout+'_dpt_pseudotime_rotated_PS', fs=fs)
plot_dp_on_poincare(model.coordinates_rotated, 
                    model.distances[model.iroot, :], cell=model.iroot,
                    title_name='dpt_pseudotime', file_name=fout+'_pmt_pseudotime_rotated_PS', fs=fs)

#### lineage analysis
np.random.seed(seed=2018)
title_name='lineages'
model.detect_lineages(n_lin=5, clustering_name='ward', k=30, rotated=True)
model.plot('rot', labels=true_labels, 
           d1=4.5, d2=4.0, fs=9, bbox=(1.0, 0.7), ms=ms,
           file_name=fname + 'forlin', coldict=color_dict)
model.plot('rot', labels=model.lineages, 
           d1=4.5, d2=4.0, fs=9, bbox=(1.0, 0.7), ms=ms,
           file_name=fname + 'lin')
plot_dp_on_poincare(model.coordinates_rotated, 
                    np.array(sadata.obs['dpt_pseudotime']), cell=model.iroot,
                    title_name='dpt_pseudotime', file_name=fout+'_dpt_pseudotime_rotated_my_myroot', fs=fs)
model.lineages[true_labels == 'Meso'] = -1
model.lineages[model.lineages == 4] = -1
# model.lineages[model.lineages == 4] = 3
model.plot('rot', labels=model.lineages, file_name=fout + '_lineages', 
            fs=9, bbox=(1.0, 0.7), ms=ms, title_name=title_name)
get_confusion_matrix(model.lineages, ['PS', 'NP', 'HF', '4SFG', '4SG', 'Meso'], true_labels, title=title_name, 
                     fname=fout + '_lineages_cm')
print('Hematopoetic genes')
model.plot_markers(data[:,idx_ery], col_names[idx_ery], 
                   file_name=fout + '_ery_markers', pm_type='rot', sc=3, fs=9)
print('Endothelial genes')
model.plot_markers(data[:,idx_endo], col_names[idx_endo], 
                   file_name=fout + '_endo_markers', pm_type='rot', sc=3, fs=9)

#### violinplot
# old root
sadata.uns['iroot'] = 532
sc.tl.dpt(sadata, n_branchings=1)
true_labels_list = np.unique(true_labels)
sns.set(style="whitegrid")
title_name = 'Diffusion_pseudotime'
fig = plt.figure(figsize=(2.8, 2.))
tips = pd.DataFrame(np.array(sadata.obs['dpt_pseudotime']), columns=['Diffusion pseudotime'])
tips['stages'] = true_labels
ax = sns.violinplot(x="stages", y='Diffusion pseudotime', 
                    order = ['PS', 'NP', 'HF', '4SFG', '4SG'], 
                    palette = color_dict, data=tips)
ax.grid('off')
plt.ylabel('')
plt.xlabel('')
plt.xticks(fontsize=fs)
plt.yticks(fontsize=fs)
plt.title('Diffusion pseudotime (old root)', fontsize=fs)
fig.tight_layout()
plt.savefig(fout + title_name + '_violin.png', format='png', dpi=600)
# new root
true_labels_list = np.unique(true_labels)
sns.set(style="whitegrid")
title_name = 'DHV_pseudotime'
fig = plt.figure(figsize=(2.8, 2.))
tips = pd.DataFrame(model.distances[model.iroot], columns=['DHV pseudotime'])
tips['stages'] = true_labels
ax = sns.violinplot(x="stages", y='DHV pseudotime', 
                    order = ['PS', 'NP', 'HF', '4SFG', '4SG'], 
                    palette = color_dict, data=tips)
ax.grid('off')
plt.ylabel('')
plt.xlabel('')
plt.xticks(fontsize=fs)
plt.yticks(fontsize=fs)
plt.title('DHV pseudotime (new root)', fontsize=fs)
fig.tight_layout()
plt.savefig(fout + title_name + '_violin.png', format='png', dpi=600)
# single lineage
model.iroot
sns.set(style="whitegrid")
n1 = 1
n2 = 4
fs=9
pl_size=2
fig, axs = plt.subplots(n1, n2, sharex=True, sharey=True, figsize=(n2*pl_size , n1*pl_size))
true_labels_list = np.unique(true_labels)
l=0
# for i in range(n1):
for j in range(n2):
    if l < len(np.unique(model.lineages))-1:
        idx_lin = np.where(model.lineages == l)[0]
        tips = pd.DataFrame(model.distances[model.iroot][idx_lin], columns=['pseudotime'])
        tips['stages'] = true_labels[idx_lin]
        ax = sns.violinplot(x="stages", y='pseudotime', 
                            order = ['PS', 'NP', 'HF', '4SFG', '4SG'], 
                            palette = color_dict,
                            data=tips, ax=axs[j])
#         axs[i, j] = ax
        axs[j].grid('off')        
        axs[j].set_title('Lineage '+str(l), fontsize=fs)
        axs[j].set_ylabel('')
        axs[j].set_xlabel('')            
        axs[j].xaxis.set_tick_params(labelsize=fs)
        axs[j].yaxis.set_tick_params(labelsize=fs)
    else:
        axs[j].grid('off')
        axs[j].axis('off')
    l += 1
fig.tight_layout()
plt.savefig(fout + 'lineages' + '_violin.png', format='png', dpi=600)

