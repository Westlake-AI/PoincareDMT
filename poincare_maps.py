import numpy as np
import torch
import torch as th
from tqdm import tqdm
import timeit
from sklearn.neighbors import kneighbors_graph
from scipy.sparse import csgraph
from torch.optim.optimizer import Optimizer
from torch import nn
from torch.autograd import Function
from torch.utils.data import TensorDataset, DataLoader

spten_t = th.sparse.FloatTensor
eps = 1e-5
boundary = 1 - eps



def grad(x, v, sqnormx, sqnormv, sqdist):
    alpha = (1 - sqnormx)
    beta = (1 - sqnormv)        
    z = 1 + 2 * sqdist / (alpha * beta)
    a = ((sqnormv - 2 * torch.sum(x * v, dim=-1) + 1) /
            torch.pow(alpha, 2)).unsqueeze(-1).expand_as(x)
    a = a * x - v / alpha.unsqueeze(-1).expand_as(v)
    z = torch.sqrt(torch.pow(z, 2) - 1)
    z = torch.clamp(z * beta, min=eps).unsqueeze(-1)
    return 4 * a / z.expand_as(x)


class PoincareDistance(Function):
    @staticmethod
    def forward(self, u, v):  
        self.save_for_backward(u, v)
        self.squnorm = torch.clamp(torch.sum(u * u, dim=-1), 0, boundary)
        self.sqvnorm = torch.clamp(torch.sum(v * v, dim=-1), 0, boundary)
        self.sqdist = torch.sum(torch.pow(u - v, 2), dim=-1)
        x = self.sqdist / ((1 - self.squnorm) * (1 - self.sqvnorm)) * 2 + 1
        # arcosh
        z = torch.sqrt(torch.pow(x, 2) - 1)
        return torch.log(x + z)

    @staticmethod
    def backward(self, g):    
        u, v = self.saved_tensors
        g = g.unsqueeze(-1)
        gu = grad(u, v, self.squnorm, self.sqvnorm, self.sqdist)
        gv = grad(v, u, self.sqvnorm, self.squnorm, self.sqdist)
        return g.expand_as(gu) * gu, g.expand_as(gv) * gv

    
def klSym(preds, targets):
    # preds = preds + eps
    # targets = targets + eps
    logPreds = preds.clamp(1e-20).log()
    logTargets = targets.clamp(1e-20).log()
    diff = targets - preds
    return (logTargets * diff - logPreds * diff).sum() / len(preds)


class PoincareEmbedding(nn.Module):
    def __init__(self,
                 size,
                 dim,
                 dist=PoincareDistance,
                 max_norm=1,
                 Qdist='laplace',
                 lossfn='klSym',
                 gamma=1.0,
                 cuda=0):
        super(PoincareEmbedding, self).__init__()

        self.dim = dim
        self.size = size
        self.lt = nn.Embedding(size, dim, max_norm=max_norm)
        self.lt.weight.data.uniform_(-1e-4, 1e-4)
        self.dist = dist
        self.Qdist = Qdist
        self.lossfnname = lossfn
        self.gamma = gamma

        self.sm = nn.Softmax(dim=1)
        self.lsm = nn.LogSoftmax(dim=1)

        if lossfn == 'kl':
            self.lossfn = nn.KLDivLoss()
        elif lossfn == 'klSym':
            self.lossfn = klSym
        elif lossfn == 'mse':
            self.lossfn = nn.MSELoss()
        else:
            raise NotImplementedError

        if cuda:
            self.lt.cuda()

    def forward(self, inputs):
        embs_all = self.lt.weight.unsqueeze(0)
        embs_all = embs_all.expand(len(inputs), self.size, self.dim)

        embs_inputs = self.lt(inputs).unsqueeze(1)
        embs_inputs = embs_inputs.expand_as(embs_all)

        dists = self.dist().apply(embs_inputs, embs_all).squeeze(-1)       

        if self.lossfnname == 'kl':
            if self.Qdist == 'laplace':
                return self.lsm(-self.gamma * dists)
            elif self.Qdist == 'gaussian':
                return self.lsm(-self.gamma * dists.pow(2))
            elif self.Qdist == 'student':
                return self.lsm(-torch.log(1 + self.gamma * dists))
            else:
                raise NotImplementedError
        elif self.lossfnname == 'klSym':
            if self.Qdist == 'laplace':
                return self.sm(-self.gamma * dists)
            elif self.Qdist == 'gaussian':
                return self.sm(-self.gamma * dists.pow(2))
            elif self.Qdist == 'student':
                return self.sm(-torch.log(1 + self.gamma * dists))
            else:
                raise NotImplementedError
        elif self.lossfnname == 'mse':
            return self.sm(-self.gamma * dists)
        else:
            raise NotImplementedError
        

def poincare_grad(p, d_p):
    r"""
    Function to compute Riemannian gradient from the
    Euclidean gradient in the Poincaré ball.

    Args:
        p (Tensor): Current point in the ball
        d_p (Tensor): Euclidean gradient at p
    """
    if d_p.is_sparse:
        p_sqnorm = th.sum(
            p.data[d_p._indices()[0].squeeze()] ** 2, dim=1,
            keepdim=True
        ).expand_as(d_p._values())
        n_vals = d_p._values() * ((1 - p_sqnorm) ** 2) / 4
        d_p = spten_t(d_p._indices(), n_vals, d_p.size())
    else:
        p_sqnorm = th.sum(p.data ** 2, dim=-1, keepdim=True)
        d_p = d_p * ((1 - p_sqnorm) ** 2 / 4).expand_as(d_p)
    return d_p


def euclidean_grad(p, d_p):
    return d_p


def euclidean_retraction(p, d_p, lr):
    p.data.add_(-lr, d_p)


class RiemannianSGD(Optimizer):
    r"""Riemannian stochastic gradient descent.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        rgrad (Function): Function to compute the Riemannian gradient from
            an Euclidean gradient
        retraction (Function): Function to update the parameters via a
            retraction of the Riemannian gradient
        lr (float): learning rate
    """

    def __init__(self,
                 params,
                 lr=1e-3,
                 rgrad=poincare_grad,
                 retraction=euclidean_retraction):
        defaults = dict(lr=lr, rgrad=rgrad, retraction=retraction)
        super(RiemannianSGD, self).__init__(params, defaults)

    def step(self, lr=None):
        """Performs a single optimization step.

        Arguments:
            lr (float, optional): learning rate for the current update.
        """
        loss = None

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if lr is None:
                    lr = group['lr']
                d_p = group['rgrad'](p, d_p)
                group['retraction'](p, d_p, lr)

        return loss


def connect_knn(KNN, distances, n_components, labels):
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


def compute_rfa(features, mode='features', k_neighbours=15, distfn='sym', 
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
			KNN = connect_knn(KNN, distances, n_components, labels)
	else:
		KNN = features

	if distlocal == 'minkowski':
		# sigma = np.mean(features)
		S = np.exp(-KNN / (sigma*features.size(1)))
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


class PoincareOptions:
    def __init__(self, debugplot=False, epochs=500, batchsize=-1, lr=0.1, burnin=500, lrm=1.0, earlystop=0.0001, cuda=0):
        self.debugplot = debugplot
        self.batchsize = batchsize
        self.epochs = epochs
        self.lr =lr
        self.lrm =lrm
        self.burnin = burnin
        self.debugplot = debugplot


def train(model, data, optimizer, args, fout=None, labels=None, earlystop=0.0, color_dict=None):
    loader = DataLoader(data, batch_size=args.batchsize, shuffle=True)

    pbar = tqdm(range(args.epochs), ncols=80)

    n_iter = 0
    epoch_loss = []
    t_start = timeit.default_timer()
    earlystop_count = 0
    for epoch in pbar:
        grad_norm = []

        # determine learning rate
        lr = args.lr
        if epoch < args.burnin:
            lr = lr * args.lrm

        epoch_error = 0
        for inputs, targets in loader:
            loss = model.lossfn(model(inputs), targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step(lr=lr)

            epoch_error += loss.item()
            
            grad_norm.append(model.lt.weight.grad.data.norm().item())

            n_iter += 1

        epoch_error /= len(loader)
        epoch_loss.append(epoch_error)
        pbar.set_description("loss: {:.5f}".format(epoch_error))

        if epoch > 10:
            delta = abs(epoch_loss[epoch] - epoch_loss[epoch-1])            
            if (delta < earlystop):                
                earlystop_count += 1
            if earlystop_count > 50:
                print(f'\nStopped at epoch {epoch}')
                break

    return model.lt.weight.cpu().detach().numpy(), epoch_error, epoch


def compute_poincare_maps(features, labels, fout,
                        mode='features', k_neighbours=15, 
                        distlocal='minkowski', sigma=1.0, gamma=2.0,
                        epochs = 300,
                        color_dict=None, debugplot=False,
                        batchsize=-1, lr=0.1, burnin=500, lrm=1.0, earlystop=0.0001, cuda=0):

    RFA = compute_rfa(features, mode=mode,
                                k_neighbours=k_neighbours,
                                distlocal= distlocal,
                                distfn='MFIsym',
                                connected=True,
                                sigma=sigma)

    if batchsize < 0:
        batchsize = min(512, int(len(RFA)/10))
        print('batchsize = ', batchsize)
    lr = batchsize / 16 * lr

    indices = torch.arange(len(RFA))
    if cuda:
        indices = indices.cuda()
        RFA = RFA.cuda()

    dataset = TensorDataset(indices, RFA)

    # instantiate our Embedding predictor
    predictor = PoincareEmbedding(len(dataset), 2,
                                                dist=PoincareDistance,
                                                max_norm=1,
                                                Qdist='laplace', 
                                                lossfn = 'klSym',
                                                gamma=gamma,
                                                cuda=cuda)

    t_start = timeit.default_timer()
    optimizer = RiemannianSGD(predictor.parameters(), lr=lr)

    opt = PoincareOptions(debugplot=debugplot, batchsize=batchsize, lr=lr, 
        burnin=burnin, lrm=lrm, earlystop=earlystop, cuda=cuda, epochs=epochs)

    # train predictor
    print('Starting training...')
    embeddings, loss, epoch = train(predictor,
                                     dataset,
                                     optimizer,
                                     opt,
                                     fout=fout,
                                     labels=labels,
                                     earlystop=earlystop,
                                     color_dict=color_dict)

    t = timeit.default_timer() - t_start
    titlename = f"loss = {loss:.3e}\ntime = {t/60:.3f} min"
    print(titlename)

    return embeddings, titlename