import os
import sys
from datetime import datetime
import time
import argparse
import torch

from torch import optim
import torch.nn as nn
from sklearn import metrics
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
import custom_loss

from data_pre import load_data
# from data_preprocessing import load_data
import warnings

# from ETDDI import ETDDI
from tqdm import tqdm
import matplotlib.pyplot as plt
from tools import AverageMeter
# import torch.nn.functional as F

# from DDPM import DDIPredictor
# from DDIM import DDIModel
from model import DDIPredictor

warnings.filterwarnings('ignore', category=UserWarning)

######################### Parameters ######################
parser = argparse.ArgumentParser()
parser.add_argument('--n_atom_feats', type=int, default=64, help='num of input features')
# parser.add_argument('--n_channel', type=int, default=1, help='num of n_channel')
parser.add_argument('--n_layers', type=int, default=3, help='num of n_layers')
parser.add_argument('--d_edge', type=int, default=6, help='num of d_edge')
parser.add_argument('--n_head', type=int, default=4, help='num of n_head')
parser.add_argument('--rel_total', type=int, default=86, help='num of interaction types')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--dropout', type=float, default=0.5, help='dropout')
parser.add_argument('--n_epochs', type=int, default=150, help='num of epochs')
parser.add_argument('--batch_size', type=int, default=256, help='batch size')

parser.add_argument('--fixed_num', type=int, default=32, help='fixed_num')
parser.add_argument('--khop', type=int, default=2, help='khop')

parser.add_argument('--weight_decay', type=float, default=1e-4)
parser.add_argument('--data_size_ratio', type=int, default=1)
parser.add_argument('--use_cuda', type=bool, default=True, choices=[0, 1])
parser.add_argument('--pkl_name', type=str, default='BestModel_drugbank.pkl')
parser.add_argument('--dataset', type=str, default='miner', choices=['miner', 'zhang'])

parser.add_argument('--fold', type=int, default=0)

num_node_feats_dict = {"drugbank": 36, "kegg": 33, "zhang": 31, "miner": 33, "deep": 41}


############################################################

def do_compute(batch, device, model):
    batch = [tensor.to(device=device) for tensor in batch]
    pre_score = model(batch)
    return pre_score, torch.sigmoid(pre_score.detach()).cpu(), batch[2].detach().cpu()


def do_compute_metrics(probas_pred, target):
    pred = (probas_pred >= 0.5).astype(int)
    acc = metrics.accuracy_score(target, pred)
    auroc = metrics.roc_auc_score(target, probas_pred)
    f1_score = metrics.f1_score(target, pred)
    precision = metrics.precision_score(target, pred)
    recall = metrics.recall_score(target, pred)
    p, r, t = metrics.precision_recall_curve(target, probas_pred)
    int_ap = metrics.auc(r, p)
    ap = metrics.average_precision_score(target, probas_pred)

    return acc, auroc, f1_score, precision, recall, int_ap, ap


def compute_metrics(probas_pred, target):
    target = np.array(target).reshape(-1)

    if probas_pred.ndim == 1 or probas_pred.shape[1] == 2:
        pred = (probas_pred >= 0.5).astype(int)
        auroc = metrics.roc_auc_score(target, probas_pred)

    else:
        pred = np.argmax(probas_pred, axis=1)
        auroc = metrics.roc_auc_score(target, probas_pred, multi_class='ovr')

    acc = metrics.accuracy_score(target, pred)
    f1 = metrics.f1_score(target, pred, average='macro')
    precision = metrics.precision_score(target, pred, average='macro')
    recall = metrics.recall_score(target, pred, average='macro')
    p, r, _ = metrics.precision_recall_curve(target, probas_pred if probas_pred.ndim == 1 else probas_pred[:, 1])
    int_ap = metrics.auc(r, p)
    ap = metrics.average_precision_score(target, probas_pred,
                                         average='binary' if probas_pred.ndim == 1 or (
                                                     probas_pred.shape[1] == 2) else 'macro')
    return acc, auroc, f1, precision, recall, int_ap, ap


def train(model, train_data_loader, val_data_loader, test_data_loader, loss_fn, optimizer, n_epochs, device, fold_i,
          result_name, pkl_name, scheduler=None):
    max_acc = 0

    print(f'Fold_{fold_i} Starting training at', datetime.today())
    for i in range(1, n_epochs + 1):
        start = time.time()

        train_probas_pred = []
        train_ground_truth = []
        val_probas_pred = []
        val_ground_truth = []

        rows = []
        run_tloss = AverageMeter()
        run_vloss = AverageMeter()

        for batch in tqdm(train_data_loader, desc=f"Epoch_{i} Train_Batch Processing items", unit="item",
                          ncols=150):  # 调用collate_fn 函数合并batch中的数据

            model.train()

            batch = [tensor.to(device=device) for tensor in batch]
            p_score, rel_lable = model(batch)
            total_loss = loss_fn(p_score, rel_lable)
            train_probas_pred.append(torch.sigmoid(p_score.detach()).cpu())
            train_ground_truth.append(rel_lable.detach().cpu())

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            run_tloss.update(total_loss, len(p_score))

        train_loss = run_tloss.get_average()
        run_tloss.reset()

        with torch.no_grad():
            train_probas_pred = np.concatenate(train_probas_pred)
            train_ground_truth = np.concatenate(train_ground_truth)

            train_acc, train_auc_roc, train_f1, train_precision, train_recall, train_int_ap, train_ap = do_compute_metrics(
                train_probas_pred, train_ground_truth)

            for batch in tqdm(val_data_loader, desc=f"Epoch_{i} Val_Batch Processing items", unit="item", ncols=150):
                model.eval()
                batch = [tensor.to(device=device) for tensor in batch]
                p_score, rel_lable = model(batch)
                total_loss = loss_fn(p_score, rel_lable)

                val_probas_pred.append(torch.sigmoid(p_score.detach()).cpu())
                val_ground_truth.append(rel_lable.detach().cpu())
                run_vloss.update(total_loss, len(p_score))

            val_loss = run_vloss.get_average()
            run_vloss.reset()
            val_probas_pred = np.concatenate(val_probas_pred)
            val_ground_truth = np.concatenate(val_ground_truth)
            val_acc, val_auc_roc, val_f1, val_precision, val_recall, val_int_ap, val_ap = do_compute_metrics(
                val_probas_pred, val_ground_truth)

            if val_acc > max_acc:
                max_acc = val_acc
                torch.save(model, pkl_name)

        if scheduler:
            scheduler.step()

        # 填充数据
        rows.append([
            i,
            round(train_loss, 4), round(val_loss, 4),
            round(train_acc, 4), round(val_acc, 4),
            round(train_auc_roc, 4), round(val_auc_roc, 4),
            round(train_precision, 4), round(val_precision, 4),
            round(train_f1, 4), round(val_f1, 4),
            round(train_recall, 4), round(val_recall, 4),
            round(train_int_ap, 4), round(val_int_ap, 4),
            round(train_ap, 4), round(val_ap, 4)
        ])

        # 创建 DataFrame
        columns = [
            'Epoch',
            'Train Loss', 'Val Loss',
            'Train Accuracy', 'Val Accuracy',
            'Train AUC ROC', 'Val AUC ROC',
            'Train PR', 'Val PR',
            'Train F1', 'Val F1',
            'Train Recall', 'Val Recall',
            'Train Int AP', 'Val Int AP',
            'Train AP', 'Val AP',
        ]
        df = pd.DataFrame(rows, columns=columns)

        df.to_csv(result_name, mode='a', index=False, header=(i == 1))
        print(f"Results saved to {result_name}")

        print(f'Epoch {i} ({time.time() - start:.4f}s): ')
        print(f'{"Metric":<15} {"Loss":<20} {"Accuracy":<20} {"AUC ROC":<20} {"PR":<20}')
        print(f'{"Train":<15}: {train_loss:<20.4f} {train_acc:<20.4f} {train_auc_roc:<20.4f} {train_precision:<20.4f}')
        print(f'{"Validation":<15}: {val_loss:<20.4f} {val_acc:<20.4f} {val_auc_roc:<20.4f} {val_precision:<20.4f}')


def test(test_data_loader, model, loss_fn):
    test_probas_pred = []
    test_ground_truth = []
    run_tloss = AverageMeter()
    with torch.no_grad():
        for batch in tqdm(test_data_loader, desc="Test_Batch Processing items", unit="item", ncols=150):
            model.eval()

            model.eval()
            batch = [tensor.to(device=device) for tensor in batch]
            p_score, rel_lable = model(batch)
            total_loss = loss_fn(p_score, rel_lable)

            test_probas_pred.append(torch.sigmoid(p_score.detach()).cpu())
            test_ground_truth.append(rel_lable.detach().cpu())
            run_tloss.update(total_loss, len(p_score))

        test_loss = run_tloss.get_average()
        run_tloss.reset()
        test_probas_pred = np.concatenate(test_probas_pred)
        test_ground_truth = np.concatenate(test_ground_truth)
        test_acc, test_auc_roc, test_f1, test_precision, test_recall, test_int_ap, test_ap = do_compute_metrics(
            test_probas_pred, test_ground_truth)
    print('\n')
    print('============================== Test Result ==============================')
    print(
        f'\t\ttest_acc: {test_acc:.4f}, test_auc_roc: {test_auc_roc:.4f},test_f1: {test_f1:.4f},test_precision:{test_precision:.4f}')
    print(f'\t\ttest_recall: {test_recall:.4f}, test_int_ap: {test_int_ap:.4f},test_ap: {test_ap:.4f}')

    test_rows = []
    test_rows.append([
        n_epochs,
        round(test_loss, 4),
        round(test_acc, 4),
        round(test_auc_roc, 4),
        round(test_precision, 4),
        round(test_f1, 4),
        round(test_recall, 4),
        round(test_int_ap, 4),
        round(test_ap, 4)
    ])

    columns = [
        'Epoch',
        'Test Loss', 'Test Accuracy', 'Test AUC ROC',
        'Test PR', 'Test F1', 'Test Recall', 'Test Int AP', 'Test AP'
    ]
    df = pd.DataFrame(test_rows, columns=columns)
    df.to_csv(result_name, mode='a', index=False)
    print(f'Test_result have saved in {result_name} !')


if __name__ == '__main__':

    args = parser.parse_args()

    fold_i = args.fold

    if len(sys.argv) > 2:
        # argument = sys.argv[1]
        fold_i = sys.argv[2]
        print(f"Received fold_i: {fold_i}")
    else:
        print("No fold_i provided.")

    d_atom = num_node_feats_dict[args.dataset]
    d_hidden = args.n_atom_feats * 2
    n_layers = args.n_layers
    d_edge = args.d_edge
    n_head = args.n_head
    n_rbf = args.n_rbf
    rel_total = args.rel_total
    lr = args.lr
    n_epochs = args.n_epochs
    batch_size = args.batch_size

    weight_decay = args.weight_decay
    data_size_ratio = args.data_size_ratio
    device = 'cuda:0' if torch.cuda.is_available() and args.use_cuda else 'cpu'
    dropout = args.dropout
    print(args)

    save_dir = f'results/{args.dataset}/'
    pkl_name = f'{save_dir}fold{fold_i}_best_model.pkl'

    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_name = f'{save_dir}fold{fold_i}_result_{current_time}.csv'

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    train_data_loader, val_data_loader, test_data_loader = load_data(args, batch_size, fold_i)

    model = DDIPredictor(d_atom=d_atom)

    loss = nn.BCEWithLogitsLoss()

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.5)
    model.to(device=device)

    train(model, train_data_loader, val_data_loader, test_data_loader, loss, optimizer, n_epochs, device, fold_i,
          result_name, pkl_name, scheduler=scheduler)
    test_model = torch.load(pkl_name)
    test(test_data_loader, test_model, loss)
