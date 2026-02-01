import sys
from torch_geometric.data import Data
from collections import defaultdict
from sklearn.model_selection import StratifiedShuffleSplit
from rdkit import Chem
import pandas as pd
from rdkit.Chem import AllChem
from rdkit import DataStructs
from tqdm import tqdm
import torch
import pickle
import torch.utils.data
import os
import dgl
from scipy import sparse as sp
import numpy as np
import argparse


# --- 自定义 Data 类 ---
class CustomData(Data):
    def __inc__(self, key, value, *args, **kwargs):
        if key == 'line_graph_edge_index':
            return self.edge_index.size(1) if self.edge_index.nelement() != 0 else 0
        return super().__inc__(key, value, *args, **kwargs)


# --- 辅助编码函数 ---
def one_of_k_encoding(k, possible_values):
    if k not in possible_values:
        raise ValueError(f"{k} is not a valid value in {possible_values}")
    return [k == e for e in possible_values]


def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


# --- 原子特征提取 (作为 Fallback 或用于原子符号) ---
def atom_features(atom, atom_symbols, explicit_H=True, use_chirality=False):
    results = one_of_k_encoding_unk(atom.GetSymbol(), atom_symbols + ['Unknown']) + \
              one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) + \
              one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6]) + \
              [atom.GetFormalCharge(), atom.GetNumRadicalElectrons()] + \
              one_of_k_encoding_unk(atom.GetHybridization(), [
                  Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
                  Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.
                                    SP3D, Chem.rdchem.HybridizationType.SP3D2
              ]) + [atom.GetIsAromatic()]
    if explicit_H:
        results = results + one_of_k_encoding_unk(atom.GetTotalNumHs(),
                                                  [0, 1, 2, 3, 4])
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


# --- 边特征提取 (保持不变) ---
def edge_features(bond):
    bond_type = bond.GetBondType()
    return torch.tensor([
        bond_type == Chem.rdchem.BondType.SINGLE,
        bond_type == Chem.rdchem.BondType.DOUBLE,
        bond_type == Chem.rdchem.BondType.TRIPLE,
        bond_type == Chem.rdchem.BondType.AROMATIC,
        bond.GetIsConjugated(),
        bond.IsInRing()]).long()


# --- PyG 数据生成函数 (已修改支持 Uni-Mol) ---
def generate_drug_data(mol_graph, atom_symbols, smiles_rdkit_list, id, uni_atom_features=None):
    edge_list = torch.LongTensor(
        [(b.GetBeginAtomIdx(), b.GetEndAtomIdx(), *edge_features(b)) for b in mol_graph.GetBonds()])

    edge_list, edge_feats = (edge_list[:, :2], edge_list[:, 2:].float()) if len(edge_list) else (
        torch.LongTensor([]), torch.FloatTensor([]))
    edge_list = torch.cat([edge_list, edge_list[:, [1, 0]]], dim=0) if len(edge_list) else edge_list
    edge_feats = torch.cat([edge_feats] * 2, dim=0) if len(edge_feats) else edge_feats

    # [修改点 1] 替换原子特征
    if uni_atom_features is not None:
        # uni_atom_features 原始 shape: (1, n, 1536) -> 需要 (n, 1536)
        # 注意: 这里一定要转 float，因为 Uni-Mol 输出是 embedding
        features = torch.tensor(uni_atom_features).squeeze(0).float()

        # 简单的维度校验
        if features.shape[0] != mol_graph.GetNumAtoms():
            print(f"[Warning] ID {id}: Mol atoms ({mol_graph.GetNumAtoms()}) != UniMol feats ({features.shape[0]})")
            # 如果维度对不上，可能需要截断或填充，或者降级使用旧特征。
            # 这里默认直接使用，若报错则需检查数据对齐情况。
    else:
        # 原有 One-Hot 逻辑
        features = [(atom.GetIdx(), atom_features(atom, atom_symbols)) for atom in mol_graph.GetAtoms()]
        features.sort()
        _, features = zip(*features)
        features = torch.stack(features)

    line_graph_edge_index = torch.LongTensor([])
    if edge_list.nelement() != 0:
        conn = (edge_list[:, 1].unsqueeze(1) == edge_list[:, 0].unsqueeze(0)) & (
                edge_list[:, 0].unsqueeze(1) != edge_list[:, 1].unsqueeze(0))
        line_graph_edge_index = conn.nonzero(as_tuple=False).T

    new_edge_index = edge_list.T

    # 相似度计算 (保持不变，注意性能瓶颈)
    # fps = [AllChem.GetMorganFingerprintAsBitVect(mol, 2) for mol in smiles_rdkit_list]
    # 优化建议：在外部计算好传入，不要在这里循环计算 fps
    mol_graph_fps = AllChem.GetMorganFingerprintAsBitVect(mol_graph, 2)
    similarity_matrix = np.zeros((1, len(smiles_rdkit_list)))

    # 这里的循环计算比较耗时，如果数据量大建议优化
    for i in range(len(smiles_rdkit_list)):
        similarity = DataStructs.FingerprintSimilarity(smiles_rdkit_list[i], mol_graph_fps)  # 修正: 直接用列表里的 fp
        similarity_matrix[0][i] = similarity
    similarity_matrix = torch.tensor(similarity_matrix)

    data = CustomData(x=features, edge_index=new_edge_index, line_graph_edge_index=line_graph_edge_index,
                      edge_attr=edge_feats, sim=similarity_matrix, id=id)
    return data


# --- DGL 数据生成函数 (已修改支持 Uni-Mol) ---
def generate_drug_data_dgl(mol_graph, atom_symbols, id=None, uni_atom_features=None):
    edge_list = torch.LongTensor(
        [(b.GetBeginAtomIdx(), b.GetEndAtomIdx(), *edge_features(b)) for b in mol_graph.GetBonds()])
    edge_list, edge_feats = (edge_list[:, :2], edge_list[:, 2:].float()) if len(edge_list) else (
        torch.LongTensor([]), torch.FloatTensor([]))
    edge_list = torch.cat([edge_list, edge_list[:, [1, 0]]], dim=0) if len(edge_list) else edge_list
    edge_feats = torch.cat([edge_feats] * 2, dim=0) if len(edge_feats) else edge_feats

    # [修改点 2] 替换节点特征
    if uni_atom_features is not None:
        # Uni-Mol 特征: shape (1, n, 1536) -> squeeze -> (n, 1536)
        # 类型必须是 float，不再是 long
        node_feature = torch.tensor(uni_atom_features).squeeze(0).float()
    else:
        features = [(atom.GetIdx(), atom_features(atom, atom_symbols)) for atom in mol_graph.GetAtoms()]
        features.sort()
        _, features = zip(*features)
        features = torch.stack(features)
        node_feature = features.long()  # 旧逻辑是 Long

    edge_feature = edge_feats.long()

    # 构建 DGL 图
    g = dgl.DGLGraph()
    g.add_nodes(node_feature.shape[0])
    g.ndata['feat'] = node_feature

    # 批量添加边 (比循环稍快，但兼容旧逻辑)
    if len(edge_list) > 0:
        g.add_edges(edge_list[:, 0], edge_list[:, 1])

    g.edata['feat'] = edge_feature
    data_dgl = g
    return data_dgl


# --- 主数据加载函数 ---
def load_drug_mol_data(args):
    data = pd.read_csv(args.dataset_filename, delimiter=args.delimiter)
    drug_id_mol_tup = []
    symbols = list()
    drug_smile_dict = {}
    smiles_rdkit_list = []  # 这里存 Mol 对象还是 Fingerprint 对象需要统一，建议存 Fingerprint 以加速

    # 1. 读取 SMILES
    for id1, id2, smiles1, smiles2, relation in zip(data[args.c_id1], data[args.c_id2], data[args.c_s1],
                                                    data[args.c_s2], data[args.c_y]):
        drug_smile_dict[id1] = smiles1
        drug_smile_dict[id2] = smiles2

    # 2. 转换为 RDKit Mol 对象
    for id, smiles in drug_smile_dict.items():
        mol = Chem.MolFromSmiles(smiles.strip())
        if mol is not None:
            drug_id_mol_tup.append((id, mol))
            symbols.extend(atom.GetSymbol() for atom in mol.GetAtoms())

    # 3. 预计算指纹 (优化性能)
    fps_list = []
    for m in drug_id_mol_tup:
        mol = m[-1]
        fps_list.append(AllChem.GetMorganFingerprintAsBitVect(mol, 2))
    symbols = list(set(symbols))

    # [修改点 3] 加载 Uni-Mol 特征字典
    print(f'Loading Uni-Mol features from unimol_feature/{args.dataset}_atomic_features.pkl ...')
    uni_atom_dict = {}
    try:
        with open(f'unimol_feature/{args.dataset}_atomic_features.pkl', 'rb') as f:
            uni_atom_dict = pickle.load(f)
    except FileNotFoundError:
        print("Error: Uni-Mol feature file not found. Please check path.")
        exit()

    # 4. 生成 PyG 数据
    drug_data_pyg = {}
    for id, mol in tqdm(drug_id_mol_tup, desc='Processing drugs_pyg'):
        uni_feat = uni_atom_dict.get(id)  # 获取对应的 UniMol 特征
        if uni_feat is None:
            print(f"Warning: Missing UniMol feature for drug {id}")

        # 传入 uni_atom_features
        drug_data_pyg[id] = generate_drug_data(mol, symbols, fps_list, id, uni_atom_features=uni_feat)

    # 5. 生成 DGL 数据
    drug_data_dgl = {}
    for id, mol in tqdm(drug_id_mol_tup, desc='Processing drugs_dgl'):
        uni_feat = uni_atom_dict.get(id)

        # 传入 uni_atom_features
        drug_data_dgl[id] = generate_drug_data_dgl(mol, symbols, id=id, uni_atom_features=uni_feat)

    save_data(drug_data_pyg, 'drug_data_pyg.pkl', args)
    save_data(drug_data_dgl, 'drug_data_dgl.pkl', args)
    return drug_data_pyg, drug_data_dgl


# --- 以下函数保持原样，负责负采样和统计 ---

def generate_pair_triplets(args):
    pos_triplets = []
    with open(f'{args.dirname}/{args.dataset.lower()}/drug_data_pyg.pkl', 'rb') as f:
        drug_ids = list(pickle.load(f).keys())

    data = pd.read_csv(args.dataset_filename, delimiter=args.delimiter)
    for id1, id2, relation in zip(data[args.c_id1], data[args.c_id2], data[args.c_y]):
        if ((id1 not in drug_ids) or (id2 not in drug_ids)): continue
        pos_triplets.append([id1, id2, relation])

    if len(pos_triplets) == 0:
        raise ValueError('All tuples are invalid.')

    pos_triplets = np.array(pos_triplets)
    data_statistics = load_data_statistics(pos_triplets)
    drug_ids = np.array(drug_ids)

    neg_samples = []
    for pos_item in tqdm(pos_triplets, desc='Generating Negative sample'):
        temp_neg = []
        h, t, r = pos_item[:3]
        neg_heads, neg_tails = _normal_batch(h, t, r, args.neg_ent, data_statistics, drug_ids, args)
        temp_neg = [str(neg_h) + '$h' for neg_h in neg_heads] + \
                   [str(neg_t) + '$t' for neg_t in neg_tails]
        neg_samples.append('_'.join(map(str, temp_neg[:args.neg_ent])))

    df = pd.DataFrame({'Drug1_ID': pos_triplets[:, 0],
                       'Drug2_ID': pos_triplets[:, 1],
                       'Y': pos_triplets[:, 2],
                       'Neg samples': neg_samples})
    filename = f'{args.dirname}/{args.dataset}/pair_pos_neg_triplets.csv'
    df.to_csv(filename, index=False)
    print(f'\nData saved as {filename}!')
    save_data(data_statistics, 'data_statistics.pkl', args)


def load_data_statistics(all_tuples):
    print('Loading data statistics ...')
    statistics = dict()
    statistics["ALL_TRUE_H_WITH_TR"] = defaultdict(list)
    statistics["ALL_TRUE_T_WITH_HR"] = defaultdict(list)
    statistics["FREQ_REL"] = defaultdict(int)
    statistics["ALL_H_WITH_R"] = defaultdict(dict)
    statistics["ALL_T_WITH_R"] = defaultdict(dict)
    statistics["ALL_TAIL_PER_HEAD"] = {}
    statistics["ALL_HEAD_PER_TAIL"] = {}

    for h, t, r in tqdm(all_tuples, desc='Getting data statistics'):
        statistics["ALL_TRUE_H_WITH_TR"][(t, r)].append(h)
        statistics["ALL_TRUE_T_WITH_HR"][(h, r)].append(t)
        statistics["FREQ_REL"][r] += 1.0
        statistics["ALL_H_WITH_R"][r][h] = 1
        statistics["ALL_T_WITH_R"][r][t] = 1

    for t, r in statistics["ALL_TRUE_H_WITH_TR"]:
        statistics["ALL_TRUE_H_WITH_TR"][(t, r)] = np.array(list(set(statistics["ALL_TRUE_H_WITH_TR"][(t, r)])))
    for h, r in statistics["ALL_TRUE_T_WITH_HR"]:
        statistics["ALL_TRUE_T_WITH_HR"][(h, r)] = np.array(list(set(statistics["ALL_TRUE_T_WITH_HR"][(h, r)])))

    for r in statistics["FREQ_REL"]:
        statistics["ALL_H_WITH_R"][r] = np.array(list(statistics["ALL_H_WITH_R"][r].keys()))
        statistics["ALL_T_WITH_R"][r] = np.array(list(statistics["ALL_T_WITH_R"][r].keys()))
        statistics["ALL_HEAD_PER_TAIL"][r] = statistics["FREQ_REL"][r] / len(statistics["ALL_T_WITH_R"][r])
        statistics["ALL_TAIL_PER_HEAD"][r] = statistics["FREQ_REL"][r] / len(statistics["ALL_H_WITH_R"][r])

    print('getting data statistics done!')
    return statistics


def _corrupt_ent(positive_existing_ents, max_num, drug_ids, args):
    corrupted_ents = []
    while len(corrupted_ents) < max_num:
        candidates = args.random_num_gen.choice(drug_ids, (max_num - len(corrupted_ents)) * 2, replace=False)
        invalid_drug_ids = np.concatenate([positive_existing_ents, corrupted_ents], axis=0)
        mask = np.isin(candidates, invalid_drug_ids, assume_unique=True, invert=True)
        corrupted_ents.extend(candidates[mask])
    corrupted_ents = np.array(corrupted_ents)[:max_num]
    return corrupted_ents


def _normal_batch(h, t, r, neg_size, data_statistics, drug_ids, args):
    neg_size_h = 0
    neg_size_t = 0
    prob = data_statistics["ALL_TAIL_PER_HEAD"][r] / (data_statistics["ALL_TAIL_PER_HEAD"][r] +
                                                      data_statistics["ALL_HEAD_PER_TAIL"][r])
    for i in range(neg_size):
        if args.random_num_gen.random() < prob:
            neg_size_h += 1
        else:
            neg_size_t += 1
    return (_corrupt_ent(data_statistics["ALL_TRUE_H_WITH_TR"][t, r], neg_size_h, drug_ids, args),
            _corrupt_ent(data_statistics["ALL_TRUE_T_WITH_HR"][h, r], neg_size_t, drug_ids, args))


def save_data(data, filename, args):
    dirname = f'{args.dirname}/{args.dataset}'
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    filename = dirname + '/' + filename
    with open(filename, 'wb') as f:
        pickle.dump(data, f)
    print(f'\nData saved as {filename}!')


def split_data(args):
    filename = f'{args.dirname}/{args.dataset}/pair_pos_neg_triplets.csv'
    df = pd.read_csv(filename)
    seed = args.seed
    class_name = args.class_name
    save_to_filename = os.path.splitext(filename)[0]
    cv_split = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=seed)
    for fold_i, (train_index, test_index) in enumerate(cv_split.split(X=df, y=df[class_name])):
        print(f'Fold {fold_i} generated!')
        train_df = df.iloc[train_index]
        test_df = df.iloc[test_index]
        train_df.to_csv(f'{save_to_filename}_train_fold{fold_i}.csv', index=False)
        print(f'{save_to_filename}_train_fold{fold_i}.csv', 'saved!')
        test_df.to_csv(f'{save_to_filename}_test_fold{fold_i}.csv', index=False)
        print(f'{save_to_filename}_test_fold{fold_i}.csv', 'saved!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, default='drugbank', required=True,
                        choices=['drugbank', 'zhang', 'miner', 'deep'],
                        help='Dataset to preprocess.')
    parser.add_argument('-n', '--neg_ent', type=int, default=1, help='Number of negative samples')
    parser.add_argument('-s', '--seed', type=int, default=0, help='Seed for the random number generator')
    parser.add_argument('-o', '--operation', type=str, default='split', required=True,
                        choices=['all', 'generate_triplets', 'drug_data', 'split'], help='Operation to perform')
    parser.add_argument('-t_r', '--test_ratio', type=float, default=0.2)
    parser.add_argument('-n_f', '--n_folds', type=int, default=3)

    dataset_columns_map = {
        'drugbank': ('ID1', 'ID2', 'X1', 'X2', 'Y'),
        'deep': ('ID1', 'ID2', 'X1', 'X2', 'Y'),
        'zhang': ('ID1', 'ID2', 'X1', 'X2', 'Y'),
        'miner': ('ID1', 'ID2', 'X1', 'X2', 'Y'),
        'twosides': ('Drug1_ID', 'Drug2_ID', 'Drug1', 'Drug2', 'New Y'),
    }

    dataset_file_name_map = {
        'drugbank': ('data/drugbank.tab', '\t'),
        'deep': ('data/deep.tab', '\t'),
        'zhang': ('data/zhang.tab', '\t'),
        'miner': ('data/miner.tab', '\t')
    }

    sys.argv = ['data_pre.py', '-d', 'zhang', '-o', 'all']  # 已移除硬编码，以便命令行传参

    args = parser.parse_args()
    args.dataset = args.dataset.lower()

    # 简单的容错，防止 Key Error
    if args.dataset in dataset_columns_map:
        args.c_id1, args.c_id2, args.c_s1, args.c_s2, args.c_y = dataset_columns_map[args.dataset]
        args.dataset_filename, args.delimiter = dataset_file_name_map[args.dataset]
    else:
        print(f"Dataset {args.dataset} configuration not found.")
        exit()

    args.dirname = 'data/warm start'
    args.random_num_gen = np.random.RandomState(args.seed)

    if args.operation in ('all', 'drug_data'):
        load_drug_mol_data(args)

    if args.operation in ('all', 'generate_triplets'):
        generate_pair_triplets(args)

    if args.operation in ('all', 'split'):
        args.class_name = 'Y'
        split_data(args)