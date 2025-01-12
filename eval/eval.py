import numpy as np
import torch
import manifolds.poincare as poincare
import timeit
import pandas as pd
import os
import joblib


def DistanceSquared(x, y=None, metric="euclidean"):
    if metric == "euclidean":
        if y is not None:
            m, n = x.size(0), y.size(0)
            xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
            yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
            dist = xx + yy
            dist = torch.addmm(dist, mat1=x, mat2=y.t(), beta=1, alpha=-2)
            dist = dist.clamp(min=1e-12)
        else:
            m, n = x.size(0), x.size(0)
            xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
            yy = xx.t()
            dist = xx + yy
            dist = torch.addmm(dist, mat1=x, mat2=x.t(), beta=1, alpha=-2)
            dist = dist.clamp(min=1e-12)
            dist[torch.eye(dist.shape[0]) == 1] = 1e-12

    if metric == 'poin_dist_mobiusm_v2':
        if y is None:
            y = x
        PoincareBall = poincare.PoincareBall()
        dist = PoincareBall.sqdist_xu_mobius_v2(x, y, c=1)
        dist = dist.clamp(min=1e-12)

    return dist

def euclidean_distance(x):
    torch.set_default_tensor_type('torch.DoubleTensor')
    # print('computing euclidean distance...')
    nx = x.size(0)
    x = x.contiguous()
    
    x = x.view(nx, -1)

    norm_x = torch.sum(x ** 2, 1, keepdim=True).t()
    ones_x = torch.ones(nx, 1)

    xTx = torch.mm(ones_x, norm_x)
    xTy = torch.mm(x, x.t())
    
    d = (xTx.t() + xTx - 2 * xTy)
    d[d < 0] = 0

    return d

def poincare_distance(x):
    # print('computing poincare distance...')
    eps = 1e-5
    boundary = 1 - eps
    
    nx = x.size(0)
    x = x.contiguous()
    x = x.view(nx, -1)
    
    norm_x = torch.sum(x ** 2, 1, keepdim=True)
    sqdist = euclidean_distance(x) * 2    
    squnorm = 1 - torch.clamp(norm_x, 0, boundary)

    x = (sqdist / torch.mm(squnorm, squnorm.t())) + 1
    z = torch.sqrt(torch.pow(x, 2) - 1)
    
    return torch.log(x + z)

def sq_distances(x):
    sq_norms_x = torch.sum(x**2, dim=1, keepdim=True)
    sq_norms_y = torch.transpose(sq_norms_x, 0, 1)
    dotprods = torch.matmul(x, torch.transpose(x, 0, 1))
    d = sq_norms_x + sq_norms_y - 2. * dotprods
    d_scaled = 1+2*d/((1-sq_norms_x)*(1-sq_norms_y))
    return sq_norms_x, sq_norms_y, d_scaled

def hyp_distances(x, q=1, c=1.0):
    sq_norms_x, sq_norms_y, d_xy = sq_distances(x)
    hyp_dist = torch.acosh(d_xy)**q
    return sq_norms_x, d_xy, hyp_dist

def get_scalars(qs):
	lcmc = np.copy(qs)
	N = len(qs)
	for j in range(N):
		lcmc[j] = lcmc[j] - j/N    
	K_max = np.argmax(lcmc) + 1

	Qlocal = np.mean(qs[:K_max])
	Qglobal = np.mean(qs[K_max:])

	return Qlocal, Qglobal, K_max

def get_coRanking(Rank_high, Rank_low):
    start = timeit.default_timer()
    n = len(Rank_high)
    coRank = np.zeros([n-1, n-1])

    for i in range(n):
        for j in range(n):
            k = int(Rank_high[i, j])
            l = int(Rank_low[i, j])
            if (k > 0) and (l > 0):
                coRank[k-1][l-1] += 1
	
    print(f"Co-ranking: time = {(timeit.default_timer() - start):.2f} sec")
    return coRank

def get_score(Rank_high, Rank_low, fname=None):	
	coRank = get_coRanking(Rank_high, Rank_low)
	start = timeit.default_timer()
	n = len(Rank_high) + 1

	df_score = pd.DataFrame(columns=['Qnx', 'Bnx'])

	Qnx = 0
	Bnx = 0
	for K in range(1, n-1):
		Fk = list(range(K))

		Qnx += sum(coRank[:K, K-1]) + sum(coRank[K-1, :K]) - coRank[K-1, K-1]
		Bnx += sum(coRank[:K, K-1]) - sum(coRank[K-1, :K])

		df_score.loc[len(df_score)] = [Qnx /(K*n), Bnx/(K*n)]

	if not (fname is None):
		df_score.to_csv(fname, sep = ',', index=False)

	# print(df_score.mean()[['Qnx', 'Bnx']])
	Qlocal, Q_global, Kmax = get_scalars(df_score['Qnx'].values)
	print(f"Qlocal = {Qlocal:.2f}, Q_global = {Q_global:.2f}, Kmax = {Kmax}")
	print(f"Time = {(timeit.default_timer() - start):.2f} sec")
	return df_score, Qlocal, Q_global, Kmax

def get_ranking(D):
    start = timeit.default_timer()
    n = len(D)

    Rank = np.zeros([n, n])
    for i in range(n):
        # tmp = D[i, :10]
        idx = np.array(list(range(n)))
        
        sidx = np.argsort(D[i, :])
        Rank[i, idx[sidx][1:]] = idx[1:]-np.ones(n-1)

    print(f"Ranking: time = {(timeit.default_timer() - start):.1f} sec")
    return Rank

def get_quality_metrics(coord_high, coord_low, metric_s='euclidean', metric_e='euclidean', fname=None):
    coord_high = torch.tensor(coord_high)
    coord_low = torch.tensor(coord_low)
    D_high = DistanceSquared(coord_high, metric=metric_s)	
    D_low = DistanceSquared(coord_low, metric=metric_e)

    Rank_high = get_ranking(D_high)
    Rank_low = get_ranking(D_low)

    df_score, Qlocal, Q_global, Kmax = get_score(Rank_high, Rank_low, fname=fname)

    return df_score, Qlocal, Q_global, Kmax

def Eval_all(coord_high, coord_low, label, metric_e='euclidean'):
    
    ######## local and global structure preservation
    df_score, Q_local, Q_global, Kmax = get_quality_metrics(coord_high, coord_low, metric_s='euclidean', metric_e=metric_e)

    return [Q_local, Q_global]

def Eval_all_sample(coord_high, coord_low, label, metric_e='euclidean', data_name='celegan'):
    filename = "/usr/data/evnet-zelin/save_sample/data_name{}_data_sample".format(data_name)
    os.makedirs("save_sample", exist_ok=True)
    if not os.path.exists(filename):
        import random
        N = range(coord_high.shape[0])
        m = 5000
        sample = random.sample(N, m)
        joblib.dump(value=sample, filename=filename)
        print("save data to ", filename)
    else:
        print("load data from ", filename)
        sample = joblib.load(filename)

    coord_high = coord_high[sample]
    coord_low = coord_low[sample]

    ######## local and global structure preservation
    df_score, Q_local, Q_global, Kmax = get_quality_metrics(np.float32(coord_high), coord_low, metric_s='euclidean', metric_e=metric_e) # poin_dist_mobiusm_v2, lor_dist_v2, euclidean, cosine

    return [Q_local, Q_global]



