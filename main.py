import functools
import os
import numpy as np
import pytorch_lightning as pl
import torch
import Loss.dmt_loss_aug2 as dmt_loss_aug

from torch.nn import functional as F
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torch import nn
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torchvision import transforms
from aug.aug import aug_near_feautee_change, aug_near_mix, aug_randn
from dataloader import data_base
from manifolds.hyperbolic_project import ToPoincare
from layers.hyp_layers import HypLinear, HypAct
import manifolds
from eval.eval import Eval_all, Eval_all_sample

torch.set_num_threads(2)


def gpu2np(a):
    return a.cpu().detach().numpy()


class NN_FCBNRL_MM(nn.Module):
    def __init__(self, in_dim, out_dim, channel=8, use_RL=True):
        super(NN_FCBNRL_MM, self).__init__()
        m_l = []
        m_l.append(
            nn.Linear(
                in_dim,
                out_dim,
            )
        )
        m_l.append(nn.BatchNorm1d(out_dim))
        if use_RL:
            m_l.append(nn.LeakyReLU(0.1))
        self.block = nn.Sequential(*m_l)

    def forward(self, x):
        return self.block(x)

class HNN_FCBNRL_MM(nn.Module):
    def __init__(self, in_dim, out_dim, channel=8, use_RL=True):
        super(HNN_FCBNRL_MM, self).__init__()
        act = getattr(F, 'leaky_relu')
        manifold = getattr(manifolds, "PoincareBall")()
        m_l = []
        m_l.append(
            HypLinear(manifold, in_dim, out_dim, 1, 0.0, 0)
        )
        if use_RL:
            m_l.append(HypAct(manifold, 1, 1, act))
        self.block = nn.Sequential(*m_l)

    def forward(self, x):
        return self.block(x)

class LitPatNN(LightningModule):
    def __init__(
        self,
        dataname,
        **kwargs,
    ):

        super().__init__()

        self.dataname = dataname
        self.save_hyperparameters()
        self.t = 0.1
        self.alpha = None
        self.stop = False
        self.detaalpha = self.hparams.detaalpha
        self.bestval = 0
        self.aim_cluster = None
        self.importance = None
        self.mse = torch.nn.CrossEntropyLoss()
        self.setup()

        self.hparams.num_pat = min(self.data_train.data.shape[1], self.hparams.num_pat)

        self.model_pat, self.model_b = self.InitNetworkMLP(
            self.hparams.NetworkStructure_1,
            self.hparams.NetworkStructure_2,
        )

        if self.data_train.data.shape[0] > 10000:
            self.scatter_size = 3
        else:
            self.scatter_size = 7

        if self.hparams.num_fea_aim < 1:
            self.hparams.num_fea_aim = int(
                self.data_train.data.shape[1]*self.hparams.num_fea_aim)
        else:
            self.hparams.num_fea_aim = int(
                self.hparams.num_fea_aim
            )
        self.hparams.num_fea_aim = min(
            self.hparams.num_fea_aim, self.data_train.data.shape[1]
        )

        self.rie_pro_latent = ToPoincare(c=1, manifold="PoincareBall")

        self.Loss = dmt_loss_aug.MyLoss(
            v_input=100,
            metric=self.hparams.metric,
            augNearRate=self.hparams.augNearRate,
        )

        if len(self.data_train.data.shape) > 2:
            self.transforms = transforms.AutoAugment(
                transforms.AutoAugmentPolicy.CIFAR10
            )

        self.fea_num = 1
        for i in range(len(self.data_train.data.shape) - 1):
            self.fea_num = self.fea_num * self.data_train.data.shape[i + 1]

        print("fea_num", self.fea_num)
        self.PM_root = nn.Linear(self.fea_num, 1)
        self.PM_root.weight.data = torch.ones_like(self.PM_root.weight.data) / 5

    def forward_fea(self, x):
        self.mask = self.PM_root.weight.reshape(-1) > 0.1
        if self.alpha is not None:
            lat = x * ((self.PM_root.weight.reshape(-1)) * self.mask)
        else:
            lat = x * ((self.PM_root.weight.reshape(-1)) * self.mask).detach()
        lat1 = self.model_pat(lat)
        lat2 = self.rie_pro_latent(lat1)
        lat3 = lat2
        for i, m in enumerate(self.model_b):
            lat3 = m(lat3)
        return lat1, lat1, lat3

    def forward(self, x):
        return self.forward_fea(x)

    def training_step(self, batch, batch_idx):
        index = batch.to(self.device)
        index_cpu = index.cpu()
        data_index = self.data_train.data[index]
        data_index = data_index.to(self.device)
        data_aug, random_select_near_index = self.augmentation_warper(index, data_index)
        data_neighbor = self.data_train.data[random_select_near_index]
        index_cpu_all = torch.cat([index_cpu, random_select_near_index.cpu()])
        data_rfa = self.data_train.data_rfa[index_cpu_all].T[index_cpu_all].T.to(self.device)
        data = torch.cat([data_index, data_neighbor, data_aug])
        data = data.reshape(data.shape[0], -1)

        pat, mid, lat = self(data)
        loss_rfa, loss_manifold = self.Loss(
            data_rfa = data_rfa,
            input_data=mid.reshape(mid.shape[0], -1),
            latent_data=lat.reshape(lat.shape[0], -1),
            v_latent=self.hparams.nu,
            v_latent_rfa = self.hparams.nu_rfa,
            metric="euclidean",
        )

        if self.current_epoch >= 200:
            loss_topo = loss_rfa + self.hparams.eta * loss_manifold
        else:
            loss_topo = loss_rfa
        
        return loss_topo

    def validation_step(self, batch, batch_idx):
        if (self.current_epoch + 1) % self.hparams.log_interval == 0:
            index = batch.to(self.device)
            data = self.data_train.data[index]
            data = data.reshape(data.shape[0], -1)
            pat, mid, lat = self(data)

            return (
                gpu2np(data),
                gpu2np(pat),
                gpu2np(lat),
                np.array(self.data_train.label.cpu())[gpu2np(index)],
                gpu2np(index),
            )

    def validation_epoch_end(self, outputs):
        if not self.stop:
            self.log("es_monitor", self.current_epoch)
        else:
            self.log("es_monitor", 0)

        if (self.current_epoch + 1) % self.hparams.log_interval == 0:
            print("self.current_epoch", self.current_epoch)
            data = np.concatenate([data_item[0] for data_item in outputs])
            mid_old = np.concatenate([data_item[1] for data_item in outputs])
            ins_emb = np.concatenate([data_item[2] for data_item in outputs])
            label = np.concatenate([data_item[3] for data_item in outputs])
            index = np.concatenate([data_item[4] for data_item in outputs])

            self.data = data
            self.mid_old = mid_old
            self.ins_emb = ins_emb
            self.label = label
            self.index = index

            Eval_all(data, ins_emb, label, metric_e='poin_dist_mobiusm_v2')
            # result_index = Eval_all_sample(data, ins_emb, label, metric_e='poin_dist_mobiusm_v2', data_name=self.hparams.data_name)
            
            data_test = self.data_test.data
            label_test = self.data_test.label
            _, _, lat_test = self(data_test)

            data_test = gpu2np(data_test)
            lat_test = gpu2np(lat_test)
            label_test = gpu2np(label_test)

            if self.hparams.save_checkpoint:
                np.save(
                    "save_checkpoint/"
                    + self.hparams.data_name
                    + "={}".format(self.current_epoch),
                    gpu2np(self.PM_root.weight.data),
                )

        else:
            self.log("SVC", 0)

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.hparams.lr, weight_decay=1e-9
        )
        self.scheduler = StepLR(
            optimizer, step_size=self.hparams.epochs // 10, gamma=0.8
        )
        return [optimizer], [self.scheduler]

    def setup(self, stage=None):
        dataset_f = getattr(data_base, self.dataname + "Dataset")
        self.data_train = dataset_f(
            data_name=self.hparams.data_name,
            knn = self.hparams.knn,
            sigma = self.hparams.sigma,
            n_components = self.hparams.n_components,
            train=True,
            datapath=self.hparams.data_path,
        )
        if len(self.data_train.data.shape) == 2:
            self.data_train.cal_near_index(
                device=self.device,
                k=self.hparams.K,
                n_components = self.hparams.n_components,
                uselabel=bool(self.hparams.uselabel),
            )
        self.data_train.to_device_("cuda")

        self.data_test = dataset_f(
            data_name=self.hparams.data_name,
            knn = self.hparams.knn,
            sigma = self.hparams.sigma,
            n_components = self.hparams.n_components,
            train=False,
            datapath=self.hparams.data_path,
        )
        self.data_test.to_device_("cuda")

        self.dims = self.data_train.get_dim()

    def train_dataloader(self):
        return DataLoader(
            self.data_train,
            drop_last=True,
            shuffle=True,
            batch_size=min(self.hparams.batch_size, self.data_train.data.shape[0]),
            num_workers=1,
            pin_memory=True,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.data_train,
            batch_size=min(self.hparams.batch_size, self.data_train.data.shape[0]),
            num_workers=1,
            pin_memory=True,
            persistent_workers=True,
        )

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.hparams.batch_size)

    def InitNetworkMLP(self, NetworkStructure_1, NetworkStructure_2):
        num_fea_per_pat = self.hparams.num_fea_per_pat
        struc_model_pat = (
            [functools.reduce(lambda x, y: x * y, self.dims)]
            + NetworkStructure_1[1:]
            + [num_fea_per_pat]
        )
        struc_model_b = NetworkStructure_2 + [self.hparams.num_latent_dim]
        struc_model_b[0] = num_fea_per_pat

        m_l = []
        for i in range(len(struc_model_pat) - 1):
            m_l.append(
                NN_FCBNRL_MM(
                    struc_model_pat[i],
                    struc_model_pat[i + 1],
                )
            )
        model_pat = nn.Sequential(*m_l)

        model_b = nn.ModuleList()
        for i in range(len(struc_model_b) - 1):
            if i != len(struc_model_b) - 2:
                model_b.append(HNN_FCBNRL_MM(struc_model_b[i], struc_model_b[i + 1]))
            else:
                model_b.append(
                    HNN_FCBNRL_MM(struc_model_b[i], struc_model_b[i + 1], use_RL=False)
                )

        print(model_pat)
        print(model_b)
        return model_pat, model_b

    def augmentation_warper(self, index, data1):
        return self.augmentation(index, data1)

    def augmentation(self, index, data1):
        data2_list = []
        if self.hparams.Uniform_t > 0:
            data_new, random_select_near_index = aug_near_mix(
                index,
                self.data_train,
                k=self.hparams.K,
                random_t=self.hparams.Uniform_t,
                device=self.device,
            )
            data2_list.append(data_new)
        if self.hparams.Bernoulli_t > 0:
            data_new = aug_near_feautee_change(
                index,
                self.data_train,
                k=self.hparams.K,
                t=self.hparams.Bernoulli_t,
                device=self.device,
            )
            data2_list.append(data_new)
        if self.hparams.Normal_t > 0:
            data_new = aug_randn(
                index,
                self.data_train,
                k=self.hparams.K,
                t=self.hparams.Normal_t,
                device=self.device,
            )
            data2_list.append(data_new)
        if (
            max(
                [
                    self.hparams.Uniform_t,
                    self.hparams.Normal_t,
                    self.hparams.Bernoulli_t,
                ]
            )
            < 0
        ):
            data_new = data1
            data2_list.append(data_new)

        if len(data2_list) == 1:
            data2 = data2_list[0]
        elif len(data2_list) == 2:
            data2 = (data2_list[0] + data2_list[1]) / 2
        elif len(data2_list) == 3:
            data2 = (data2_list[0] + data2_list[1] + data2_list[2]) / 3

        return data2, random_select_near_index


def main(args):

    pl.utilities.seed.seed_everything(1)
    callbacks_list = []

    if args.save_checkpoint:
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            dirpath="save_checkpoint/",
            every_n_epochs=args.log_interval,
            filename=args.data_name + "{epoch}",
        )
        callbacks_list.append(checkpoint_callback)

    model = LitPatNN(
        dataname=args.data_name,
        **args.__dict__,
    )
    early_stop = EarlyStopping(
        monitor="es_monitor", patience=1, verbose=False, mode="max"
    )
    callbacks_list.append(early_stop)

    trainer = Trainer(
        gpus=1,
        max_epochs=args.epochs,
        callbacks=callbacks_list,
    )
    print("start fit")
    trainer.fit(model)
    trainer.save_checkpoint("model_save/model.ckpt")
    print("end fit")

    model.eval()


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description="*** author")
    parser.add_argument('--name', type=str, default='digits_T', )
    parser.add_argument("--offline", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1, metavar="S")
    parser.add_argument("--data_path", type=str, default="./data")
    parser.add_argument("--log_interval", type=int, default=400)
    parser.add_argument("--project_name", type=str, default="test")
    parser.add_argument("--method", type=str, default="Ours")
    parser.add_argument(
        "--computer", type=str,
        default=os.popen("git config user.name").read()[:-1]
    )

    # dataset param
    parser.add_argument(
        "--data_name",
        type=str,
        default="Olsson",
        choices=[
            "Olsson",
        ],
    )
    parser.add_argument(
        "--n_point",
        type=int,
        default=60000000,
    )

    # model param
    parser.add_argument(
        "--metric",
        type=str,
        default="euclidean",
    )
    parser.add_argument("--detaalpha", type=float, default=1.005)
    parser.add_argument("--l2alpha", type=float, default=10)
    parser.add_argument("--nu", type=float, default=0.03)
    parser.add_argument("--nu_rfa", type=float, default=0.01)
    parser.add_argument("--num_link_aim", type=float, default=0.2)
    parser.add_argument("--num_fea_aim", type=float, default=2000)
    parser.add_argument("--K_plot", type=int, default=40)
    parser.add_argument("--save_checkpoint", type=int, default=0)

    parser.add_argument("--num_fea_per_pat", type=int, default=80)
    parser.add_argument("--K", type=int, default=10)
    parser.add_argument("--Uniform_t", type=float, default=1)
    parser.add_argument("--Bernoulli_t", type=float, default=-1)
    parser.add_argument("--Normal_t", type=float, default=-1)
    parser.add_argument("--uselabel", type=int, default=0)
    parser.add_argument("--showmainfig", type=int, default=1)

    # train param
    parser.add_argument(
        "--NetworkStructure_1", type=list, default=[-1, 200] + [200] * 5
    )
    parser.add_argument("--NetworkStructure_2", type=list, default=[-1, 500, 80])
    parser.add_argument("--num_pat", type=int, default=8)
    parser.add_argument("--num_latent_dim", type=int, default=2)
    parser.add_argument("--augNearRate", type=float, default=1000)
    parser.add_argument("--eta", type=float, default=10)
    parser.add_argument("--knn", type=int, default=5)
    parser.add_argument("--n_components", type=int, default=50)
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--explevel", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=38,)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--lr", type=float, default=1e-3, metavar="LR")

    args = pl.Trainer.add_argparse_args(parser)
    args = args.parse_args()

    main(args)
