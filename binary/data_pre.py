import json
import os
import pickle
import random
from collections import defaultdict

import math
import networkx as nx

import torch
from sklearn.model_selection import StratifiedShuffleSplit
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch
from rdkit import Chem
from rdkit.Chem import AllChem
import pandas as pd
import numpy as np

from tqdm import tqdm
import time
import ast
from collections import deque

from torch_geometric.utils import subgraph, degree, get_laplacian


def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def atom_features(atom, atom_feat_list,
                  explicit_H=True,
                  use_chirality=True):
    results = one_of_k_encoding_unk(
        atom.GetSymbol(),
        atom_feat_list) + [atom.GetDegree() / 10, atom.GetFormalCharge(), atom.GetNumRadicalElectrons()] + \
              one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6]) + \
              one_of_k_encoding_unk(atom.GetHybridization(), [
                  Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
                  Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.
                                    SP3D, Chem.rdchem.HybridizationType.SP3D2
              ]) + [atom.GetIsAromatic()]

    if explicit_H:
        results = results + [atom.GetTotalNumHs()]

    if use_chirality:
        try:
            results = results + one_of_k_encoding_unk(
                atom.GetProp('_CIPCode'),
                ['R', 'S']) + [atom.HasProp('_ChiralityPossible')]
        except:
            results = results + [False, False
                                 ] + [atom.HasProp('_ChiralityPossible')]

    results = np.array(results).astype(np.float32)

    return torch.from_numpy(results)


def edge_features(bond):
    '''
    Get bond features
    '''
    bond_type = bond.GetBondType()
    return torch.tensor([
        bond_type == Chem.rdchem.BondType.SINGLE,
        bond_type == Chem.rdchem.BondType.DOUBLE,
        bond_type == Chem.rdchem.BondType.TRIPLE,
        bond_type == Chem.rdchem.BondType.AROMATIC,
        bond.GetIsConjugated(),
        bond.IsInRing()]).long()


def get_mol_edge_list_and_feat_mtx(mol_graph, atom_feat_list):
    n_features = [(atom.GetIdx(), atom_features(atom, atom_feat_list)) for atom in mol_graph.GetAtoms()]
    n_features.sort()
    _, n_features = zip(*n_features)
    n_features = torch.stack(n_features)

    edge_list = torch.LongTensor(
        [(b.GetBeginAtomIdx(), b.GetEndAtomIdx(), *edge_features(b)) for b in mol_graph.GetBonds()])
    edge_list, edge_feats = (edge_list[:, :2], edge_list[:, 2:].float()) if len(edge_list) else (
        torch.LongTensor([]), torch.FloatTensor([]))

    edge_list = torch.cat([edge_list, edge_list[:, [1, 0]]], dim=0) if len(edge_list) else edge_list
    edge_feats = torch.cat([edge_feats] * 2, dim=0) if len(edge_feats) else edge_feats

    new_edge_index = edge_list.T

    z = torch.LongTensor([torch.nonzero(row[:len(atom_feat_list)] == 1).squeeze().item() if torch.sum(
        row[:len(atom_feat_list)] == 1) == 1 else None for row in n_features])

    return [new_edge_index, n_features, edge_feats, z]


def read_network(drug_list, current_set):
    edge_index = []
    rel_index = []

    id_list = drug_list['drug_id'].tolist()

    for line in current_set:
        head, tail, rel = line
        if rel == 0:
            continue
        else:
            edge_index.append([int(id_list.index(head)), int(id_list.index(tail))])
            rel_index.append(int(rel))

    num_node = np.max((np.array(edge_index)))
    num_rel = max(rel_index) + 1
    print(f'Multi-relations:' + str(len(list(set(rel_index)))))

    return num_node, edge_index, rel_index, num_rel


def generate_k_subgraphs(drug_list, mol_feat, current_set, args, data_type):
    id_list = drug_list['drug_id'].tolist()
    all_contained_drgus = [index for index, _ in enumerate(id_list)]

    num_node, network_edge_index, network_rel_index, num_rel = read_network(drug_list, current_set)

    subgraphs, max_degree, max_rel_num = generate_node_subgraphs(args, data_type, mol_feat, id_list,
                                                                 all_contained_drgus, network_edge_index,
                                                                 network_rel_index,
                                                                 num_rel)

    return subgraphs, max_degree, max_rel_num


def generate_node_subgraphs(args, data_type, uni_mol_feat, id_list, drug_id, network_edge_index, network_rel_index,
                            num_rel):
    edge_index = torch.from_numpy(np.array(network_edge_index).T)
    rel_index = torch.from_numpy(np.array(network_rel_index))

    row, col = edge_index
    reverse_edge_index = torch.stack((col, row), 0)
    undirected_edge_index = torch.cat((edge_index, reverse_edge_index), 1)

    paths = "dataset/" + str(args.dataset) + "/" + 'k_hop_subgraph' + "/"
    if not os.path.exists(paths):
        os.mkdir(paths)

    subgraphs, max_degree, max_rel_num = subtreeExtractor(data_type, uni_mol_feat, id_list, drug_id,
                                                          undirected_edge_index, rel_index, paths, num_rel,
                                                          fixed_num=args.fixed_num, khop=args.khop, fold=args.fold)

    return subgraphs, max_degree, max_rel_num


def subtreeExtractor(data_type, uni_mol_feat, id_list, drug_id, edge_index, rel_index, shortest_paths, num_rel,
                     fixed_num, khop, fold):
    all_degree = []
    num_rel_update = []
    subgraphs = {}

    data_path = shortest_paths + f"{data_type}_fold{fold}_subtree_fixed_" + str(fixed_num) + "_hop_" + str(
        khop) + "sp.pkl"

    if os.path.exists(data_path):
        with open(data_path, 'rb') as f:
            print(f'Reading {data_type} sub_graph data!')
            subgraphs = pickle.load(f)
            max_rel = 0
            max_degree = 0
            for s in subgraphs.keys():
                if len(subgraphs[s][6]) > 0:
                    max_rel = max(subgraphs[s][6]) if max(subgraphs[s][6]) > max_rel else max_rel
                    max_degree = subgraphs[s][7] if subgraphs[s][7] > max_degree else max_degree

        return subgraphs, max_degree, max_rel

    undirected_rel_index = torch.cat((rel_index, rel_index), 0)

    for d in tqdm(drug_id, desc=f"Generating {data_type} sub_graph", ncols=150):
        subset, sub_edge_index, sub_rel_index, mapping_list = k_hop_subgraph(int(d), khop, edge_index,
                                                                             undirected_rel_index, fixed_num,
                                                                             relabel_nodes=True, num_nodes=len(id_list))
        row, col = sub_edge_index
        if col.numel() > 0:
            all_degree.append(torch.max(degree(col)).item())
            max_degree = torch.max(degree(col)).item()
        else:
            all_degree.append(0)
            max_degree = 0

        new_s_edge_index = sub_edge_index.transpose(1, 0).numpy().tolist()
        new_s_value = [1 for _ in range(len(new_s_edge_index))]
        new_s_rel = sub_rel_index.numpy().tolist()
        node_idx = subset.numpy().tolist()

        s_edge_index = new_s_edge_index.copy()
        s_value = new_s_value.copy()
        s_rel = new_s_rel.copy()

        edge_index_value = calculate_shortest_path(sub_edge_index.transpose(1, 0).numpy())
        if edge_index_value.ndim == 1:
            sp_edge_index = []
            sp_value = []
        else:
            sp_edge_index = edge_index_value[:, :2]
            sp_value = edge_index_value[:, 2]

        for i in range(len(sp_edge_index)):
            if sp_value[i] == 1:
                continue
            else:
                s_edge_index.append(sp_edge_index[i].tolist())
                s_value.append(sp_value[i])
                s_rel.append(sp_value[i] + num_rel)

        assert len(s_edge_index) == len(s_value)
        assert len(s_edge_index) == len(s_rel)

        if len(s_rel) > 0:
            num_rel_update.append(np.max(s_rel))
        else:
            num_rel_update.append(0)

        if len(new_s_edge_index) == 0:
            new_s_edge_index = [[0, 0]]
            new_s_rel = [1]

        node_feat = torch.tensor([uni_mol_feat[id_list[id]] for id in node_idx]).squeeze(1)
        subgraphs[d] = node_feat, new_s_edge_index, new_s_rel, mapping_list, s_edge_index, s_value, s_rel, max_degree

    with open(data_path, 'wb') as f:
        pickle.dump(subgraphs, f)

    return subgraphs, max(all_degree), max(num_rel_update)


def k_hop_subgraph(node_idx, num_hops, edge_index, rel_index, fixed_num, relabel_nodes=False,
                   num_nodes=None, flow='source_to_target'):
    np.random.seed(42)
    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    assert flow in ['source_to_target', 'target_to_source']
    if flow == 'target_to_source':
        row, col = edge_index
    else:
        col, row = edge_index

    node_mask = row.new_empty(num_nodes, dtype=torch.bool)
    edge_mask = row.new_empty(row.size(0), dtype=torch.bool)

    if isinstance(node_idx, (int, list, tuple)):
        node_idx = torch.tensor([node_idx], device=row.device).flatten()
    else:
        node_idx = node_idx.to(row.device)

    subsets = [node_idx]

    for _ in range(num_hops):
        node_mask.fill_(False)
        node_mask[subsets[-1]] = True
        torch.index_select(node_mask, 0, row, out=edge_mask)

        if fixed_num == None:
            subsets.append(col[edge_mask])
        elif col[edge_mask].size(0) > fixed_num:
            neighbors = np.random.choice(a=col[edge_mask].numpy(), size=fixed_num, replace=False)
            subsets.append(torch.LongTensor(neighbors))
        else:
            subsets.append(col[edge_mask])

    subset, inv = torch.cat(subsets).unique(return_inverse=True)
    inv = inv[:node_idx.numel()]

    node_mask.fill_(False)
    node_mask[subset] = True
    edge_mask = node_mask[row] & node_mask[col]

    edge_index = edge_index[:, edge_mask]

    if relabel_nodes:
        node_idx = row.new_full((num_nodes,), -1)
        node_idx[subset] = torch.arange(subset.size(0), device=row.device)
        edge_index = node_idx[edge_index]

    rel_index = rel_index[edge_mask] if rel_index is not None else None

    mapping_mask = [False for _ in range(len(subset))]
    mapping_mask[inv] = True

    return subset, edge_index, rel_index, mapping_mask


def maybe_num_nodes(edge_index, num_nodes=None):
    if num_nodes is not None:
        return num_nodes
    elif isinstance(edge_index, Tensor):

        return int(edge_index.max()) + 1 if edge_index.numel() > 0 else 0
    else:
        return max(edge_index.size(0), edge_index.size(1))


def calculate_shortest_path(edge_index):
    s_edge_index_value = []

    g = nx.DiGraph()
    g.add_edges_from(edge_index.tolist())

    paths = nx.all_pairs_shortest_path_length(g)
    for node_i, node_ij in paths:
        for node_j, length_ij in node_ij.items():
            s_edge_index_value.append([node_i, node_j, length_ij])

    s_edge_index_value.sort()

    return np.array(s_edge_index_value)


def convert(o):
    if isinstance(o, np.int64): return int(o)
    raise TypeError


class DrugDataset(Dataset):
    def __init__(self, drug_list, all_molfeature, uni_mol_dict, uni_atom_dict, ddi_subgraph, tri_list, ratio=1.0,
                 shuffle=False):
        ''''disjoint_split: Consider whether entities should appear in one and only one split of the data
        '''
        self.tri_list = []
        self.ratio = ratio
        self.all_molfeature = all_molfeature

        self.uni_mol_dict = uni_mol_dict
        self.uni_atom_dict = uni_atom_dict

        self.ddi_subgraph = ddi_subgraph
        self.drug_list = drug_list
        self.id_list = drug_list['drug_id'].tolist()

        for h, t, r in tri_list:
            if ((h in all_molfeature) and (t in all_molfeature)):
                self.tri_list.append((h, t, r))

        if shuffle:
            random.seed(6)
            random.shuffle(self.tri_list)
        limit = math.ceil(len(self.tri_list) * ratio)
        self.tri_list = self.tri_list[:limit]

    def __len__(self):
        return len(self.tri_list)

    def __getitem__(self, index):
        return self.tri_list[index]

    def collate_fn(self, batch):
        pos_rels = []
        pos_h_samples = []
        pos_t_samples = []

        pos_h_uni_atom_samples = []
        pos_t_uni_atom_samples = []

        h_subgraph = []
        t_subgraph = []

        for h, t, r in batch:
            pos_rels.append(r)
            h_data = self.__create_graph_data(h)
            t_data = self.__create_graph_data(t)

            pos_h_samples.append(h_data)
            pos_t_samples.append(t_data)

            pos_h_uni_atom_samples.append(self.get_uni_atom_graph(h))
            pos_t_uni_atom_samples.append(self.get_uni_atom_graph(t))

            h_subgraph.append(self.creat_subgraph_data(self.ddi_subgraph[self.id_list.index(h)]))
            t_subgraph.append(self.creat_subgraph_data(self.ddi_subgraph[self.id_list.index(t)]))

        pos_h_graph = Batch.from_data_list(pos_h_samples)
        pos_t_graph = Batch.from_data_list(pos_t_samples)
        pos_rels = torch.LongTensor(pos_rels).unsqueeze(0)
        pos_h_uni_atom_samples = Batch.from_data_list(pos_h_uni_atom_samples)
        pos_t_uni_atom_samples = Batch.from_data_list(pos_t_uni_atom_samples)

        h_subgraph = Batch.from_data_list(h_subgraph)
        t_subgraph = Batch.from_data_list(t_subgraph)

        pos_tri = (
            pos_h_graph, pos_t_graph, pos_rels, pos_h_uni_atom_samples, pos_t_uni_atom_samples, h_subgraph, t_subgraph)
        return pos_tri

    def __create_graph_data(self, id):
        edge_index = self.all_molfeature[id][0]
        n_features = self.all_molfeature[id][1]
        edge_feature = self.all_molfeature[id][2]

        z = self.all_molfeature[id][3]

        return Data(x=n_features, edge_index=edge_index, edge_attr=edge_feature, z=z)

    def creat_subgraph_data(self, subgraph_data):
        n_features = torch.tensor(subgraph_data[0])
        edge_index = torch.tensor(subgraph_data[1]).T
        edge_feature = torch.tensor(subgraph_data[2])

        return Data(x=n_features, edge_index=edge_index, edge_attr=edge_feature)

    def get_uni_atom_graph(self, id):
        n_features = torch.tensor(self.uni_atom_dict[id].squeeze(0))
        edge_index = self.all_molfeature[id][0]
        edge_feature = self.all_molfeature[id][2]

        return Data(x=n_features, edge_index=edge_index, edge_attr=edge_feature)


class DrugDataLoader(DataLoader):
    def __init__(self, data, **kwargs):
        super().__init__(data, collate_fn=data.collate_fn, **kwargs)


def split_train_valid(data, val_ratio=0.125):
    data = np.array(data)
    cv_split = StratifiedShuffleSplit(n_splits=2, test_size=val_ratio, random_state=1)
    train_index, val_index = next(iter(cv_split.split(X=data, y=data[:, 2])))
    train_tup = data[train_index]
    val_tup = data[val_index]
    train_tup = [(tup[0], tup[1], int(tup[2])) for tup in train_tup]
    val_tup = [(tup[0], tup[1], int(tup[2])) for tup in val_tup]

    return train_tup, val_tup


def data_unzip(data):
    data_tup = [(h, t, r, n) for h, t, r, n in zip(data['Drug1_ID'], data['Drug2_ID'], data['Y'], data['Neg samples'])]

    data_set = []
    for h, t, r, n in data_tup:

        Neg_ID, Ntype = n.split('$')
        data_set.append((h, t, 1))

        if Ntype == 'h':
            data_set.append((Neg_ID, t, 0))

        else:
            data_set.append((h, Neg_ID, 0))

    return data_set


def load_data(args, batch_size, fold_i):
    dataset = args.dataset

    df_drugs_smiles = pd.read_csv(f'dataset/{dataset}/drug_smiles.csv', dtype=str)

    drug_id_mol_graph_tup = [(id, Chem.MolFromSmiles(smiles.strip())) for id, smiles in
                             zip(df_drugs_smiles['drug_id'], df_drugs_smiles['smiles'])]

    with open(f'unimol_feature/{dataset}_molecular_features.pkl', 'rb') as f:
        uni_mol_dict = pickle.load(f)

    with open(f'unimol_feature/{dataset}_atomic_features.pkl', 'rb') as f:
        uni_atom_dict = pickle.load(f)

    atom_feat_list = set()
    for line in drug_id_mol_graph_tup:
        for atom in line[1].GetAtoms():
            atom_feat_list.add(atom.GetSymbol())
    atom_feat_list = sorted(atom_feat_list)

    drug_data_path = f'dataset/{dataset}/drug_data.pkl'
    if os.path.exists(drug_data_path):
        with open(drug_data_path, 'rb') as f:
            print('Reading drug_data!')
            MOL_EDGE_LIST_FEAT_MTX = pickle.load(f)
    else:

        MOL_EDGE_LIST_FEAT_MTX = {drug_id: get_mol_edge_list_and_feat_mtx(mol, atom_feat_list)
                                  for drug_id, mol in
                                  tqdm(drug_id_mol_graph_tup, desc="Generating mol_feature", ncols=150)}

        MOL_EDGE_LIST_FEAT_MTX = {drug_id: mol for drug_id, mol in
                                  tqdm(MOL_EDGE_LIST_FEAT_MTX.items(), desc="Generating mol_feature", ncols=150) if
                                  mol is not None}

        with open(drug_data_path, 'wb') as f:
            pickle.dump(MOL_EDGE_LIST_FEAT_MTX, f)

    df_ddi_train = pd.read_csv(f'dataset/{dataset}/train_fold{fold_i}.csv', dtype=str)
    df_ddi_test = pd.read_csv(f'dataset/{dataset}/test_fold{fold_i}.csv', dtype=str)

    train_tup = data_unzip(df_ddi_train)

    train_tup, val_tup = split_train_valid(train_tup, val_ratio=0.125)

    test_tup = data_unzip(df_ddi_test)

    train_subgraphs, train_max_degree, train_max_rel_num = generate_k_subgraphs(df_drugs_smiles, uni_mol_dict,
                                                                                train_tup, args, 'train')

    train_data = DrugDataset(df_drugs_smiles, MOL_EDGE_LIST_FEAT_MTX, uni_mol_dict, uni_atom_dict, train_subgraphs,
                             train_tup)
    val_data = DrugDataset(df_drugs_smiles, MOL_EDGE_LIST_FEAT_MTX, uni_mol_dict, uni_atom_dict, train_subgraphs,
                           val_tup)
    test_data = DrugDataset(df_drugs_smiles, MOL_EDGE_LIST_FEAT_MTX, uni_mol_dict, uni_atom_dict, train_subgraphs,
                            test_tup)

    print(
        f"Fold_{fold_i} Training with {len(train_data)} samples,and valing with {len(val_data)}, and testing with {len(test_data)}")

    train_data_loader = DrugDataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4)
    val_data_loader = DrugDataLoader(val_data, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4)
    test_data_loader = DrugDataLoader(test_data, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4)

    return train_data_loader, val_data_loader, test_data_loader
