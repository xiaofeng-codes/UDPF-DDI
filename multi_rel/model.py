import pickle

import math
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool, LayerNorm, GCNConv, HeteroConv, global_add_pool, global_max_pool
from torch_geometric.utils import to_dense_batch
from torch_scatter import scatter_softmax

from FuzzyLayer import FuzzyLayer
from Diff_Transformer.multihead_diffattn import MultiheadDiffAttn  # 导入Diff Attention模块

# from KAN import KAN
# from torch_sparse import SparseTensor


class FuzzyGCN(nn.Module):
    def __init__(self, d_hidden, d_out, mem_num=5):
        super().__init__()
        # self.output_norm = LayerNorm(d_hidden)
        self.first_norm = LayerNorm(d_hidden)
        self.first_linear = nn.Linear(d_hidden, d_hidden)
        self.output_norm = LayerNorm(d_hidden)

        self.FuzzyLayers = nn.ModuleList([
            FuzzyLayer(d_hidden, d_hidden, 20)
            for _ in range(mem_num)      # 假设你想要创建10个FusedFuzzyDeepNet实例
        ])

        self.GCNConv = GCNConv(d_hidden, d_hidden)
        # 添加dropout层
        self.dropout = nn.Dropout(0.5)

    def forward(self, drug, edge_index, batch, edge_attr=None):

        first_out = drug

        if edge_attr is None:
            gcn_out = self.GCNConv(first_out, edge_index)
        else:
            gcn_out = self.GCNConv(first_out, edge_index, edge_attr)

        midnorm_out = self.first_norm(gcn_out, batch)

        # 计算多模糊隶属度
        layer_outputs = [layer(midnorm_out) for layer in self.FuzzyLayers]

        fuzzy_out = torch.stack(layer_outputs, dim=-1)

        outnorm_out = fuzzy_out.sum(dim=-1)

        final_out = outnorm_out * gcn_out

        # final_out = self.dropout(final_out)
        if torch.isnan(final_out).any() or torch.isinf(final_out).any():
            print(final_out)

        # final_out = final_out + drug        # 使用残差连接大法

        return final_out

class DDIPredictor(nn.Module):
    def __init__(self, d_atom=33, d_edge=6, d_hidden=256, d_unimol=1536, atom_gcn_layers=3, mol_gcn_layers=3, diff_layers=1, multi_rel=65, atom_mem_num=5, subgraph_mem_num=3):
        super().__init__()

        self.init_norm = LayerNorm(d_atom)
        self.init_linear = nn.Linear(d_atom, d_hidden)
        self.atom_edge_emb = nn.Sequential(
            nn.Linear(d_edge, d_edge * 2),
            nn.SiLU(),
            nn.Linear(d_edge * 2, 1),
            nn.Sigmoid()  # 确保输出在0-1之间
        )
        # 定义多层FuzzyGCN
        self.FGCN = nn.ModuleList([
            FuzzyGCN(d_hidden, d_hidden, mem_num=atom_mem_num) for _ in range(atom_gcn_layers)  # 3层FuzzyGCN
        ])
        self.classifier = nn.Sequential(
            nn.Linear(d_hidden * 2, 4 * d_hidden),
            nn.SiLU(),
            nn.Dropout(0.5),
            nn.Linear(4 * d_hidden, multi_rel)
        )


        # self.edge_multi_rel_emb = nn.Embedding(num_embeddings=multi_rel, embedding_dim=1)
        self.edge_multi_rel_emb = nn.Sequential(
            nn.Embedding(num_embeddings=multi_rel, embedding_dim=1),
            nn.Sigmoid()  # 确保输出在0-1之间
        )
        self.subgraph_norm = LayerNorm(d_unimol)
        # self.subgraph_norm2 = LayerNorm(d_hidden)
        self.subgraph_linear = nn.Linear(d_unimol, d_hidden)
        self.subgraph_FGCN = nn.ModuleList([
            FuzzyGCN(d_hidden, d_hidden, mem_num=subgraph_mem_num) for _ in range(mol_gcn_layers)  # 2层FuzzyGCN
        ])
        self.subgraph_mlp = nn.Sequential(
            nn.Linear(d_hidden * 2, d_hidden * 4),
            nn.SiLU(),
            nn.Dropout(0.5),
            nn.Linear(d_hidden * 4, multi_rel),
        )


        self.uni_mol_norm = LayerNorm(d_unimol)
        self.uni_mol_linear = nn.Linear(d_unimol, d_hidden)
        # self.inter_ex = InterExtractor(d_hidden, 20)
        self.attention = nn.Linear(d_unimol, 1)
        self.diff_transformer = nn.Sequential(*[
            MultiheadDiffAttn(embed_dim=d_hidden, depth=i, num_heads=8, num_kv_heads=4)
            for i in range(diff_layers)
        ])

        self.uni_mol_mlp = nn.Sequential(
            nn.Linear(d_hidden * 2, d_hidden * 4),
            nn.SiLU(),
            nn.Dropout(0.5),
            nn.Linear(d_hidden * 4, multi_rel),
        )

        self.multi_rel_emb = nn.Embedding(num_embeddings=multi_rel, embedding_dim=d_hidden)
        self.multi_rel_norm = nn.LayerNorm(d_hidden)
        # self.multi_rel_kan = KAN((d_hidden, 1))

        # 添加动态融合权重
        self.fusion_weights = nn.Parameter(torch.tensor([1.0, 1.0, 1.0]))  # 初始化为0.5, 0.2, 0.3
        # self.fusion_softmax = nn.Softmax(dim=0)  # 用于归一化权重

    def diff_encoder(self, x, batch):
        uni_mol = self.uni_mol_norm(x, batch)

        # 1. 多模式池化
        sum_pool = global_add_pool(uni_mol, batch)  # (batch_size, feat_dim)
        mean_pool = global_mean_pool(uni_mol, batch)  # (batch_size, feat_dim)
        max_pool = global_max_pool(uni_mol, batch)  # (batch_size, feat_dim)

        # 2. 注意力池化（可选，扩展序列长度）
        scores = self.attention(uni_mol).squeeze(-1)  # (num_nodes,)
        weights = scatter_softmax(scores, batch, dim=0)
        # weights = torch.softmax(scores, dim=0)  # 归一化权重
        attn_pool = global_add_pool(uni_mol * weights.unsqueeze(-1), batch)  # (batch_size, feat_dim)

        # 3. 拼接为序列 (batch_size, seq, feat_dim)
        graph_features = torch.stack(
            [sum_pool, mean_pool, max_pool, attn_pool],
            dim=1  # 新增序列维度
        )  # 若 seq_length=4，则形状为 (batch_size, 4, feat_dim)

        graph_features = self.uni_mol_linear(graph_features)

        # 扩展维度
        # h_mask = h_mask.unsqueeze(1).unsqueeze(2)  # 变为 (256, 1, 1, 71)
        #
        # # 创建数值型掩码
        # h_mask = h_mask.to(torch.float)
        # h_mask = h_mask.masked_fill(h_mask == 0, float('-inf'))  # 不可见位置
        # h_mask = h_mask.masked_fill(h_mask == 1, 0)  # 可见位置

        # 多层 Diff Attention 前向传播
        # diff_uni_mol = self.diff_transformer1(batch_uni_mol, h_mask)

        diff_uni_mol = self.diff_transformer(graph_features)

        diff_uni_mol = torch.mean(diff_uni_mol, dim=1)
        # diff_uni_mol, last_feat = self.gru(diff_uni_mol)

        return diff_uni_mol


    def forward(self, batch):
        h_drug, t_drug, rel_lable, uni_atom_h, uni_atom_t, h_subgraph, t_subgraph = batch

        # one_hot编码原子特征处理过程
        h_drug.x = self.init_norm(h_drug.x, h_drug.batch)  # t药物原子特征归一化
        t_drug.x = self.init_norm(t_drug.x, t_drug.batch)
        h_drug.x = self.init_linear(h_drug.x)
        t_drug.x = self.init_linear(t_drug.x)
        h_drug.edge_attr = self.atom_edge_emb(h_drug.edge_attr)
        t_drug.edge_attr = self.atom_edge_emb(t_drug.edge_attr)
        # 逐层处理FuzzyGCN
        for layer in self.FGCN:
            h_drug.x = layer(h_drug.x, h_drug.edge_index, h_drug.batch, h_drug.edge_attr.float())
            t_drug.x = layer(t_drug.x, t_drug.edge_index, t_drug.batch, t_drug.edge_attr.float())
        z1 = global_mean_pool(h_drug.x, h_drug.batch)
        z2 = global_mean_pool(t_drug.x, t_drug.batch)
        scores1 = self.classifier(torch.cat([z1, z2], dim=-1))


        # 子图处理过程
        h_subgraph.edge_attr = self.edge_multi_rel_emb(h_subgraph.edge_attr)
        t_subgraph.edge_attr = self.edge_multi_rel_emb(t_subgraph.edge_attr)
        h_subgraph.x = self.subgraph_norm(h_subgraph.x, h_subgraph.batch)
        t_subgraph.x = self.subgraph_norm(t_subgraph.x, t_subgraph.batch)
        h_subgraph.x = self.subgraph_linear(h_subgraph.x)
        t_subgraph.x = self.subgraph_linear(t_subgraph.x)
        # 逐层处理FuzzyGCN
        for layer in self.subgraph_FGCN:
            h_subgraph.x = layer(h_subgraph.x, h_subgraph.edge_index, h_subgraph.batch, h_subgraph.edge_attr.float())
            t_subgraph.x = layer(t_subgraph.x, t_subgraph.edge_index, t_subgraph.batch, t_subgraph.edge_attr.float())
        h_sub_pool = global_mean_pool(h_subgraph.x, h_subgraph.batch)
        t_sub_pool = global_mean_pool(t_subgraph.x, t_subgraph.batch)
        scores2 = self.subgraph_mlp(torch.cat([h_sub_pool, t_sub_pool], dim=-1))      #使用MLP


        # unimoi原子特征处理过程
        h_uni_mol = self.diff_encoder(uni_atom_h.x, uni_atom_h.batch)
        t_uni_mol = self.diff_encoder(uni_atom_t.x, uni_atom_t.batch)
        scores3 = self.uni_mol_mlp(torch.cat([h_uni_mol, t_uni_mol], dim=-1))      #使用MLP
        # scores = self.uni_mol_kan(torch.cat([h_uni_mol, t_uni_mol], dim=-1))      #使用KAN


        # 多关系处理过程
        # emb_lable = self.multi_rel_emb((rel_lable.reshape(-1, 1).squeeze(-1)))
        # # norm_lable = self.multi_rel_norm(emb_lable)
        # kan_lable = self.multi_rel_kan(emb_lable)


        # 动态融合
        weight1, weight2, weight3 = self.fusion_weights
        scores = weight1 * scores1 + weight2 * scores2 + weight3 * scores3
        # scores = scores3

        if torch.isnan(scores).any() or torch.isinf(scores).any() or torch.isnan(torch.sigmoid(scores.detach())).any() or torch.isinf(torch.sigmoid(scores.detach())).any():
            print(scores)

        rel_lable = rel_lable.reshape(-1, 1).long()
        # 创建one-hot编码
        one_hot = torch.zeros(len(rel_lable), 65, device=rel_lable.device)
        one_hot.scatter_(1, rel_lable, 1)

        num_n = 0

        # # 保存模型预测分数scores和真实值one_hot到npy文件
        np.save(f'deng_pred{num_n}_case.npy', scores.detach().cpu().numpy())
        np.save(f'deng_target{num_n}_case.npy', one_hot.detach().cpu().numpy())


        return scores, one_hot





