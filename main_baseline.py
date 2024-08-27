import os
import pytorch_lightning as pl
import torch
import numpy as np
from dataloader import data_base
import scanpy as sc
import anndata

from sklearn.manifold import TSNE
import phate
from sklearn.decomposition import PCA
import umap
import pacmap
import forceatlas2
import plotly.graph_objects as go
import wandb
from eval.eval_xu import Eval_all, Eval_all_sample, Eval_all_poincaremaps, Eval_all_sample_poincaremaps, Eval_all_scdhmap, Eval_all_sample_scdhmap
from poincare_maps import *

torch.set_num_threads(2)

def up_mainfig_emb(data, ins_emb,
    label, n_clusters=10, num_cf_example=2,
):
    color = np.array(label)
    import plotly.express as px

    Curve = ins_emb[:, 0]
    Curve2 = ins_emb[:, 1]

    ml_mx = max(Curve)
    ml_mn = min(Curve)
    ap_mx = max(Curve2)
    ap_mn = min(Curve2)

    if ml_mx > ap_mx:
        mx = ml_mx
    else:
        mx = ap_mx

    if ml_mn < ap_mn:
        mn = ml_mn
    else:
        mn = ap_mn

    mx = mx + mx * 0.2
    mn = mn - mn * 0.2

    layout = go.Layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        width=1000,
        height=1000,
        autosize=False,
    )
    fig = go.Figure(layout=layout)
    color_set_list = list(set(color.tolist()))
    for c in color_set_list:
        m = color == c
        fig.add_trace(
            go.Scatter(
                mode="markers",
                x=ins_emb[m, 0],
                y=ins_emb[m, 1],
                marker_line_width=0,
                name=c,
                marker=dict(
                    size=[3] * ins_emb.shape[0],
                )
            )
        )

    return fig


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description="*** author")
    parser.add_argument('--name', type=str, default='digits_T', )
    parser.add_argument("--offline", type=int, default=0)
    parser.add_argument("--project_name", type=str, default="test")
    parser.add_argument("--data_path", type=str, default="./data")
    parser.add_argument("--knn", type=int, default=15)
    parser.add_argument("--sigma", type=float, default=2.0)
    parser.add_argument("--gamma", type=float, default=2.0)
    parser.add_argument("--n_components", type=int, default=100)
    parser.add_argument('--method', type=str, default='poincaremaps', choices=['pca', 'tsne', 'umap', 'pacmap', 'phate', 'diffmap', 'forceatlas2', 'forceatlas2_v2', 'ivis', 'poincaremaps', 'scphere_wn', 'scdhmap'])

    # data set param
    parser.add_argument(
        "--data_name",
        type=str,
        default="Paul",
        choices=[
            "MyeloidProgenitors",
            "Moignard2015",
            "Olsson",
            "Krumsiek11",
            "Planaria",
            "CELEGAN",
            "CELEGANPCA100",
            "UCEPI",
            "UCEPIPcaKnn",
            'PaulPcaKnn',
            'Paul',
            'ToggleSwitch',
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
    parser.add_argument('--MN_ratio', type=float, default=0.5)
    parser.add_argument('--FP_ratio', type=float, default=2.0)

    args = pl.Trainer.add_argparse_args(parser)
    args = args.parse_args()

    dataset_f = getattr(data_base, args.data_name + "Dataset")

    runname = 'baseline_{}_{}'.format(
        args.data_name, 
        args.method)
    
    wandb.init(
        name=runname,
        project="sc_lineage_paul_final_" + args.project_name,
        entity="sky-yongjie-xu",
        mode="offline" if bool(args.offline) else "online",
        save_code=True,
        config=args,
    )

    data_train = dataset_f(
        data_name=args.data_name,
        knn = args.knn,
        sigma = args.sigma,
        n_components = args.n_components,
        train=True,
        datapath=args.data_path,
    )

    data = data_train.data.numpy().reshape(data_train.data.shape[0], -1)
    label = np.array(data_train.label)

    from sklearn.decomposition import PCA
    data = PCA(n_components=20).fit_transform(data)

    if args.method == 'pca':
        save_path = args.data_name + '_baseline' + '/path' + '_' + args.method
        if not os.path.exists('logs/log_' + save_path):
            os.makedirs('logs/log_' + save_path)
        os.chdir(r'logs/log_' + save_path)
        latent = PCA(n_components=2).fit_transform(data)

    if args.method == 'tsne':
        save_path = args.data_name + '_baseline' + '/path' + '_' + args.method + '_' + str(args.perplexity)
        if not os.path.exists('logs/log_' + save_path):
            os.makedirs('logs/log_' + save_path)
        os.chdir(r'logs/log_' + save_path)
        latent = TSNE(perplexity=args.perplexity).fit_transform(data)

    if args.method == 'umap':
        save_path = args.data_name + '_baseline' + '/path' + '_' + args.method + '_' + str(args.perplexity) + '_' + str(args.min_dist)
        if not os.path.exists('logs/log_' + save_path):
            os.makedirs('logs/log_' + save_path)
        os.chdir(r'logs/log_' + save_path)
        latent = umap.UMAP(n_neighbors=args.perplexity, min_dist=args.min_dist).fit_transform(data)

    if args.method == 'pacmap':
        save_path = args.data_name + '_baseline' + '/path' + '_' + args.method + '_' + str(args.perplexity) + '_' + str(args.MN_ratio) + '_' + str(args.FP_ratio)
        if not os.path.exists('logs/log_' + save_path):
            os.makedirs('logs/log_' + save_path)
        os.chdir(r'logs/log_' + save_path)
        pacmap_embedding = pacmap.PaCMAP(n_components=2, n_neighbors=args.perplexity, MN_ratio=args.MN_ratio, FP_ratio=args.FP_ratio)
        latent = pacmap_embedding.fit_transform(data, init="pca")

    if args.method == 'phate':
        save_path = args.data_name + '_baseline' + '/path' + '_' + args.method
        if not os.path.exists('logs/log_' + save_path):
            os.makedirs('logs/log_' + save_path)
        os.chdir(r'logs/log_' + save_path)
        latent = phate.PHATE().fit_transform(data)

    if args.method == 'diffmap':
        save_path = args.data_name + '_baseline' + '/path' + '_' + args.method + '_' + str(args.perplexity)
        if not os.path.exists('logs/log_' + save_path):
            os.makedirs('logs/log_' + save_path)
        os.chdir(r'logs/log_' + save_path)
        sadata = anndata.AnnData(X=np.array(data))
        sc.pp.neighbors(sadata, n_neighbors=args.perplexity)
        sc.tl.diffmap(sadata)
        latent = sadata.obsm['X_diffmap'][:,1:3].copy()

    if args.method == 'forceatlas2':
        save_path = args.data_name + '_baseline' + '/path' + '_' + args.method + '_' + str(args.perplexity)
        if not os.path.exists('logs/log_' + save_path):
            os.makedirs('logs/log_' + save_path)
        os.chdir(r'logs/log_' + save_path)
        sadata = anndata.AnnData(X=np.array(data))
        sc.pp.neighbors(sadata, n_neighbors=args.perplexity)
        sc.tl.louvain(sadata, resolution=0.9)
        sc.tl.paga(sadata)
        sc.pl.paga(sadata)
        sc.tl.draw_graph(sadata, init_pos='paga')
        latent = sadata.obsm['X_draw_graph_fr'].copy()

    if args.method == 'forceatlas2_v2':
        save_path = args.data_name + '_baseline' + '/path' + '_' + args.method + '_' + str(args.perplexity)
        if not os.path.exists('logs/log_' + save_path):
            os.makedirs('logs/log_' + save_path)
        os.chdir(r'logs/log_' + save_path)
        sadata = anndata.AnnData(X=np.array(data))
        sc.pp.neighbors(sadata, n_neighbors=args.perplexity)
        adj_matrix = sadata.obsp['connectivities']
        positions = forceatlas2.forceatlas2(adj_matrix.todense())
        sadata.obsm['X_forceatlas2'] = np.array(positions)
        latent = np.array(positions)

    # if args.method == 'poincaremaps':
    #     latent = np.load('logs/log_poincaremaps/log_{}_poin_maps/path_{}_{}_{}/result/latent.npy'.format(args.data_name, args.knn, args.sigma, args.gamma))
    #     save_path = args.data_name + '_baseline' + '/path' + '_' + args.method + '_' + str(args.knn) + '_' + str(args.sigma) + '_' + str(args.gamma)
    #     if not os.path.exists('logs/log_' + save_path):
    #         os.makedirs('logs/log_' + save_path)
    #     os.chdir(r'logs/log_' + save_path)

    if args.method == 'poincaremaps':
        save_path = args.data_name + '_baseline' + '/path' + '_' + args.method + '_' + str(args.knn) + '_' + str(args.sigma) + '_' + str(args.gamma)
        if not os.path.exists('logs/log_' + save_path):
            os.makedirs('logs/log_' + save_path)
        os.chdir(r'logs/log_' + save_path)
        sadata = anndata.AnnData(X=np.array(data))
        data = torch.DoubleTensor(data)
        poincare_coord, _ = compute_poincare_maps(data, None,
                                'pmap_res/Bif',
                                mode='features', k_neighbours=args.knn, 
                                distlocal='minkowski', sigma=args.sigma, gamma=args.gamma,
                                color_dict=None, epochs=1000,
                                batchsize=-1, lr=0.1, earlystop=0.0001, cuda=0)
        sadata.obsm['X_poincaremaps'] = poincare_coord
        latent = poincare_coord

    if args.method == 'scdhmap':
        latent = np.load('logs/log_{}_baseline/path_scdhmap/latent.npy'.format(args.data_name))

    np.save('latent.npy', latent)

    if args.method == 'poincaremaps':
        if args.data_name == 'CELEGAN' or args.data_name == 'CELEGANPCA100' or args.data_name == 'UCEPI' or args.data_name == 'UCEPIPcaKnn' or args.data_name == 'Planaria':
            result_index = Eval_all_sample_poincaremaps(data, latent, label, data_name=args.data_name)
        else:
            result_index = Eval_all_poincaremaps(data, latent, label)
    if args.method == 'scdhmap':
        if args.data_name == 'CELEGAN' or args.data_name == 'CELEGANPCA100' or args.data_name == 'UCEPI' or args.data_name == 'UCEPIPcaKnn' or args.data_name == 'Planaria':
            result_index = Eval_all_sample_scdhmap(data, latent, label, data_name=args.data_name)
        else:
            result_index = Eval_all_scdhmap(data, latent, label)
    else:
        if args.data_name == 'CELEGAN' or args.data_name == 'CELEGANPCA100' or args.data_name == 'UCEPI' or args.data_name == 'UCEPIPcaKnn' or args.data_name == 'Planaria':
            result_index = Eval_all_sample(data, latent, label, metric_e=args.metric, data_name=args.data_name)
        else:
            result_index = Eval_all(data, latent, label, metric_e=args.metric)

    result_dict = {
        'metric/Q_local':result_index[0],
        'metric/Q_global':result_index[1],
        }
    wandb.log(result_dict)
    wandb.log(
        {
            "main_easy/fig_easy": up_mainfig_emb(data, latent, label),
        }
    )