import os
import pytorch_lightning as pl
import torch
import numpy as np
from dataloader import data_base
import scanpy as sc
import anndata

from sklearn.manifold import TSNE
import phate
import umap
import forceatlas2
import plotly.graph_objects as go


torch.set_num_threads(2)


def up_mainfig_emb(data, ins_emb,
    label, n_clusters=10, num_cf_example=2,
):
    color = np.array(label)

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
    parser.add_argument("--knn", type=int, default=5)
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=2.0)
    parser.add_argument("--n_components", type=int, default=50)
    parser.add_argument('--method', type=str, default='scvi_umap', choices=['harmony_tsne', 'harmony_umap', 'harmony_phate', 'harmony_diffmap', 'harmony_forceatlas2', 'harmony_forceatlas2_v2', 'scvi_tsne', 'scvi_umap', 'scvi_phate', 'scvi_diffmap', 'scvi_forceatlas2', 'scvi_forceatlas2_v2'])

    # data set param
    parser.add_argument(
        "--data_name",
        type=str,
        default="UCEPIbc",
        choices=[
            'UCEPIbc',
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

    runname = 'baseline_{}_{}'.format(
        args.data_name, 
        args.method)

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

    if args.method == 'harmony_tsne' or args.method == 'scvi_tsne':
        save_path = args.data_name + '_baseline' + '/path' + '_' + args.method + '_' + str(args.perplexity)
        if not os.path.exists('log_' + save_path):
            os.makedirs('log_' + save_path)
            os.makedirs('log_' + save_path + '/result')
        os.chdir(r'log_' + save_path)
        latent = TSNE(perplexity=args.perplexity).fit_transform(data)

    if args.method == 'harmony_umap' or args.method == 'scvi_umap':
        save_path = args.data_name + '_baseline' + '/path' + '_' + args.method + '_' + str(args.perplexity) + '_' + str(args.min_dist)
        if not os.path.exists('log_' + save_path):
            os.makedirs('log_' + save_path)
            os.makedirs('log_' + save_path + '/result')
        os.chdir(r'log_' + save_path)
        latent = umap.UMAP(n_neighbors=args.perplexity, min_dist=args.min_dist).fit_transform(data)

    if args.method == 'harmony_phate' or args.method == 'scvi_phate':
        save_path = args.data_name + '_baseline' + '/path' + '_' + args.method
        if not os.path.exists('log_' + save_path):
            os.makedirs('log_' + save_path)
            os.makedirs('log_' + save_path + '/result')
        os.chdir(r'log_' + save_path)
        latent = phate.PHATE().fit_transform(data)

    if args.method == 'harmony_diffmap' or args.method == 'scvi_diffmap':
        save_path = args.data_name + '_baseline' + '/path' + '_' + args.method + '_' + str(args.perplexity)
        if not os.path.exists('log_' + save_path):
            os.makedirs('log_' + save_path)
            os.makedirs('log_' + save_path + '/result')
        os.chdir(r'log_' + save_path)
        sadata = anndata.AnnData(X=np.array(data))
        sc.pp.neighbors(sadata, n_neighbors=args.perplexity)
        sc.tl.diffmap(sadata)
        latent = sadata.obsm['X_diffmap'][:,1:3].copy()

    if args.method == 'harmony_forceatlas2' or args.method == 'scvi_forceatlas2':
        save_path = args.data_name + '_baseline' + '/path' + '_' + args.method + '_' + str(args.perplexity)
        if not os.path.exists('log_' + save_path):
            os.makedirs('log_' + save_path)
            os.makedirs('log_' + save_path + '/result')
        os.chdir(r'log_' + save_path)
        sadata = anndata.AnnData(X=np.array(data))
        sc.pp.neighbors(sadata, n_neighbors=args.perplexity)
        sc.tl.louvain(sadata, resolution=0.9)
        sc.tl.paga(sadata)
        sc.pl.paga(sadata)
        sc.tl.draw_graph(sadata, init_pos='paga')
        latent = sadata.obsm['X_draw_graph_fr'].copy()

    if args.method == 'harmony_forceatlas2_v2' or args.method == 'scvi_forceatlas2_v2':
        save_path = args.data_name + '_baseline' + '/path' + '_' + args.method + '_' + str(args.perplexity)
        if not os.path.exists('log_' + save_path):
            os.makedirs('log_' + save_path)
            os.makedirs('log_' + save_path + '/result')
        os.chdir(r'log_' + save_path)
        sadata = anndata.AnnData(X=np.array(data))
        sc.pp.neighbors(sadata, n_neighbors=args.perplexity)
        adj_matrix = sadata.obsp['connectivities']
        positions = forceatlas2.forceatlas2(adj_matrix.todense())
        sadata.obsm['X_forceatlas2'] = np.array(positions)
        latent = np.array(positions)

    np.save('latent.npy', latent)

    from eval.eval_bc import Eval_all_structure_all
    result_index_local, result_index_global, acc_mean, silhouette_mean = Eval_all_structure_all(data_train.sadata, data, latent, label, data_train.n_batch[0], metric_e=args.metric)

