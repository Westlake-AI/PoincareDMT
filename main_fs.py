import pytorch_lightning as pl
import os
import matplotlib.pyplot as plt
import numpy as np
import shap

from dataloader import data_base
from main import LitPatNN

def main(args):
    path = 'model_Moignard2015.ckpt'
    
    model = LitPatNN.load_from_checkpoint(path)
    model.eval()
    
    print('load the data')
    dataset_f = getattr(data_base, 'Moignard2015' + "Dataset")
    data_train = dataset_f(
                data_name=args.data_name,
                knn = args.knn,
                sigma = args.sigma,
                n_components = args.n_components,
                train=True,
                datapath=args.data_path,
            )
    
    vis_2d = model(data_train.data)
    feature_names = data_train.col_names
    label = data_train.label.detach().numpy()

    ### visualization
    x = vis_2d[:, 0]
    y = vis_2d[:, 1]
    fig = plt.figure()
    plt.scatter(x, y, c=label, s=1)
    plt.title('2D Visualization of Data')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.savefig('benchmarks/exp_update/vis.png')
    plt.close()

    #### new label
    label_str = np.load('benchmarks/exp_update/new_true_labels.npy', allow_pickle=True)
    label_str_set = ['PS', 'HF', 'NP', '4SG', '4SFG', 'Meso']
    
    center = []
    for i in range(len(label_str_set)):
        center.append(vis_2d[label_str == label_str_set[i]].mean(0))
    model.center = np.array(center)
    print('center', model.center)
    data_input_exp = data_train.data.detach().numpy()
    e = shap.KernelExplainer(model.forward_muticlass, data_input_exp,)
    shap_values = e.shap_values(data_input_exp[:100], nsamples=100)
    np.save('benchmarks/exp_update/new_shap_value.npy', shap_values)
    # shap_values = np.load('benchmarks/exp_update/new_shap_value.npy')

    #### analysis
    #### PS
    print('PS genes')
    shap_values_PS = shap_values[0]
    shap_values_PS = np.abs(np.array(shap_values_PS)).mean(axis=0)
    indices_of_top_6_shap_values = np.argsort(shap_values_PS)[-6:][::-1] # 根据SHAP值从大到小排序，获取前10个最大SHAP值的索引
    top_6_feature_names = [feature_names[i] for i in indices_of_top_6_shap_values] # 根据索引找到对应的特征名称和SHAP值
    top_6_shap_values = [shap_values_PS[i] for i in indices_of_top_6_shap_values]
    print(top_6_feature_names)

    plt.figure(figsize=(17, 7))
    plt.bar(feature_names, shap_values_PS, color='skyblue')
    plt.xlabel('Genes')
    plt.ylabel('Mean |SHAP Value|')
    plt.title('Mean SHAP Values of Genes')
    plt.xticks(rotation=90)  # 将横坐标文字旋转 90 度
    plt.savefig('benchmarks/exp_update/PS_new.png')
    plt.close()

    #### NP
    print('NP genes')
    shap_values_NP = shap_values[2]
    shap_values_NP = np.abs(np.array(shap_values_NP)).mean(axis=0)
    indices_of_top_6_shap_values = np.argsort(shap_values_NP)[-6:][::-1] # 根据SHAP值从大到小排序，获取前10个最大SHAP值的索引
    top_6_feature_names = [feature_names[i] for i in indices_of_top_6_shap_values] # 根据索引找到对应的特征名称和SHAP值
    top_6_shap_values = [shap_values_NP[i] for i in indices_of_top_6_shap_values]
    print(top_6_feature_names)

    plt.figure(figsize=(17, 7))
    plt.bar(feature_names, shap_values_NP, color='skyblue')
    plt.xlabel('Genes')
    plt.ylabel('Mean |SHAP Value|')
    plt.title('Mean SHAP Values of Genes')
    plt.xticks(rotation=90)  # 将横坐标文字旋转 90 度
    plt.savefig('benchmarks/exp_update/NP_new.png')
    plt.close()

    #### HF
    print('HF genes')
    shap_values_HF = shap_values[1]
    shap_values_HF = np.abs(np.array(shap_values_HF)).mean(axis=0)
    indices_of_top_6_shap_values = np.argsort(shap_values_HF)[-6:][::-1] # 根据SHAP值从大到小排序，获取前10个最大SHAP值的索引
    top_6_feature_names = [feature_names[i] for i in indices_of_top_6_shap_values] # 根据索引找到对应的特征名称和SHAP值
    top_6_shap_values = [shap_values_HF[i] for i in indices_of_top_6_shap_values]
    print(top_6_feature_names)

    plt.figure(figsize=(17, 7))
    plt.bar(feature_names, shap_values_HF, color='skyblue')
    plt.xlabel('Genes')
    plt.ylabel('Mean |SHAP Value|')
    plt.title('Mean SHAP Values of Genes')
    plt.xticks(rotation=90)  # 将横坐标文字旋转 90 度
    plt.savefig('benchmarks/exp_update/HF_new.png')
    plt.close()

    #### 4SG
    print('4SG genes')
    shap_values_4SG = shap_values[3]
    shap_values_4SG = np.abs(np.array(shap_values_4SG)).mean(axis=0)
    indices_of_top_6_shap_values = np.argsort(shap_values_4SG)[-6:][::-1] # 根据SHAP值从大到小排序，获取前10个最大SHAP值的索引
    top_6_feature_names = [feature_names[i] for i in indices_of_top_6_shap_values] # 根据索引找到对应的特征名称和SHAP值
    top_6_shap_values = [shap_values_4SG[i] for i in indices_of_top_6_shap_values]
    print(top_6_feature_names)

    plt.figure(figsize=(17, 7))
    plt.bar(feature_names, shap_values_4SG, color='skyblue')
    plt.xlabel('Genes')
    plt.ylabel('Mean |SHAP Value|')
    plt.title('Mean SHAP Values of Genes')
    plt.xticks(rotation=90)  # 将横坐标文字旋转 90 度
    plt.savefig('benchmarks/exp_update/4SG_new.png')
    plt.close()

    #### 4SFG
    print('4SFG genes')
    shap_values_4SFG = shap_values[4]
    shap_values_4SFG = np.abs(np.array(shap_values_4SFG)).mean(axis=0)
    indices_of_top_6_shap_values = np.argsort(shap_values_4SFG)[-6:][::-1] # 根据SHAP值从大到小排序，获取前10个最大SHAP值的索引
    top_6_feature_names = [feature_names[i] for i in indices_of_top_6_shap_values] # 根据索引找到对应的特征名称和SHAP值
    top_6_shap_values = [shap_values_4SFG[i] for i in indices_of_top_6_shap_values]
    print(top_6_feature_names)

    plt.figure(figsize=(17, 7))
    plt.bar(feature_names, shap_values_4SFG, color='skyblue')
    plt.xlabel('Genes')
    plt.ylabel('Mean |SHAP Value|')
    plt.title('Mean SHAP Values of Genes')
    plt.xticks(rotation=90)  # 将横坐标文字旋转 90 度
    plt.savefig('benchmarks/exp_update/4SFG_new.png')
    plt.close()

    #### NP to HF
    print('NP to HF genes')
    shap_values_NP = shap_values[2]
    shap_values_HF = shap_values[1]
    shap_values_NP_HF = np.abs(np.array(shap_values_NP) - np.array(shap_values_HF)).mean(axis=0)
    indices_of_top_6_shap_values = np.argsort(shap_values_NP_HF)[-6:][::-1] # 根据SHAP值从大到小排序，获取前10个最大SHAP值的索引
    top_6_feature_names = [feature_names[i] for i in indices_of_top_6_shap_values] # 根据索引找到对应的特征名称和SHAP值
    top_6_shap_values = [shap_values_NP_HF[i] for i in indices_of_top_6_shap_values]
    print(top_6_feature_names)

    # 绘制柱状图
    plt.figure(figsize=(17, 7))
    plt.bar(feature_names, shap_values_NP_HF, color='skyblue')
    plt.xlabel('Genes')
    plt.ylabel('Mean |SHAP Value|')
    plt.title('Mean SHAP Values of Genes')
    plt.xticks(rotation=90)  # 将横坐标文字旋转 90 度
    plt.savefig('benchmarks/exp_update/NP_HF_new.png')
    plt.close()

    #### HF to 4SFG
    print('HF to 4SFG genes')
    shap_values_HF = shap_values[1]
    shap_values_4SFG = shap_values[4]
    shap_values_HF_4SFG = np.abs(np.array(shap_values_HF) - np.array(shap_values_4SFG)).mean(axis=0)
    indices_of_top_6_shap_values = np.argsort(shap_values_HF_4SFG)[-6:][::-1] # 根据SHAP值从大到小排序，获取前10个最大SHAP值的索引
    top_6_feature_names = [feature_names[i] for i in indices_of_top_6_shap_values] # 根据索引找到对应的特征名称和SHAP值
    top_6_shap_values = [shap_values_HF_4SFG[i] for i in indices_of_top_6_shap_values]
    print(top_6_feature_names)

    # 绘制柱状图
    plt.figure(figsize=(17, 7))
    plt.bar(feature_names, shap_values_HF_4SFG, color='skyblue')
    plt.xlabel('Genes')
    plt.ylabel('Mean |SHAP Value|')
    plt.title('Mean SHAP Values of Genes')
    plt.xticks(rotation=90)  # 将横坐标文字旋转 90 度
    plt.savefig('benchmarks/exp_update/HF_4SFG_new.png')
    plt.close()

    #### HF to 4SG
    print('HF to 4SG genes')
    shap_values_HF = shap_values[1]
    shap_values_4SG = shap_values[3]
    shap_values_HF_4SG = np.abs(np.array(shap_values_HF) - np.array(shap_values_4SG)).mean(axis=0)
    indices_of_top_6_shap_values = np.argsort(shap_values_HF_4SG)[-6:][::-1] # 根据SHAP值从大到小排序，获取前10个最大SHAP值的索引
    top_6_feature_names = [feature_names[i] for i in indices_of_top_6_shap_values] # 根据索引找到对应的特征名称和SHAP值
    top_6_shap_values = [shap_values_HF_4SG[i] for i in indices_of_top_6_shap_values]
    print(top_6_feature_names)

    # 绘制柱状图
    plt.figure(figsize=(17, 7))
    plt.bar(feature_names, shap_values_HF_4SG, color='skyblue')
    plt.xlabel('Genes')
    plt.ylabel('Mean |SHAP Value|')
    plt.title('Mean SHAP Values of Genes')
    plt.xticks(rotation=90)  # 将横坐标文字旋转 90 度
    plt.savefig('benchmarks/exp_update/HF_4SG_new.png')
    plt.close()

    print(data_train)
    
if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description="*** author")
    parser.add_argument('--name', type=str, default='digits_T',)
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

    # data set param
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
    parser.add_argument("--nu", type=float, default=5e-3)
    parser.add_argument("--nu_rfa", type=float, default=5e-3)
    parser.add_argument("--num_link_aim", type=float, default=0.2)
    parser.add_argument("--num_fea_aim", type=float, default=42)
    parser.add_argument("--K_plot", type=int, default=40)
    parser.add_argument("--save_checkpoint", type=int, default=0)

    parser.add_argument("--num_fea_per_pat", type=int, default=80)  # 0.5
    # parser.add_argument("--K", type=int, default=3)
    parser.add_argument("--K", type=int, default=10)
    parser.add_argument("--Uniform_t", type=float, default=1)  # 0.3
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
    parser.add_argument("--eta1", type=float, default=1)
    parser.add_argument("--knn", type=int, default=5)
    parser.add_argument("--n_components", type=int, default=10)
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--explevel", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=50000)
    parser.add_argument("--lr", type=float, default=1e-3, metavar="LR")

    args = pl.Trainer.add_argparse_args(parser)
    args = args.parse_args()

    main(args)