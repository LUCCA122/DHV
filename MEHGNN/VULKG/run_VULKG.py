import time
import argparse

import os
import sys

# 获取当前脚本文件所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))

# 将上一级目录添加到 sys.path 中
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)

import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

from utils.pytorchtools import EarlyStopping
from utils.data import load_VULKG_data
from utils.tools import index_generator, parse_minibatch_LastFM
from model.MEHGNN_lp import MEHGNN_lp
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, top_k_accuracy_score





# Params
num_ntype = 2
dropout_rate = 0.5
lr = 0.005
weight_decay = 0.001
etypes_lists = [[[0, 1]],
                [[1, 0]]]
use_masks = [[True],
             [True]]
no_masks = [[False], [False]]
num_user = 152
num_artist = 201
'''expected_metapaths = [
    [(0, 1, 0)],
    [(1, 0, 1)]
]
'''

expected_metapaths = [
    [(0, 1, 0)],
    [(1, 0, 1)]
]


def run_model_LastFM(feats_type, hidden_dim, num_heads, attn_vec_dim, rnn_type,
                     num_epochs, patience, batch_size, neighbor_samples, repeat, save_postfix):
    adjlists_ua, edge_metapath_indices_list_ua, _, type_mask, train_val_test_pos_user_artist, train_val_test_neg_user_artist = load_VULKG_data()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
    torch.cuda.synchronize()
    features_list = []
    in_dims = []
    if feats_type == 0:
        for i in range(num_ntype):
            dim = (type_mask == i).sum()
            in_dims.append(dim)
            indices = np.vstack((np.arange(dim), np.arange(dim)))
            indices = torch.LongTensor(indices)
            indices = indices.to(device)
            values = torch.FloatTensor(np.ones(dim))
            values = values.to(device)
            features_list.append(torch.sparse.FloatTensor(indices, values, torch.Size([dim, dim])).to(device))
        #print(features_list[0].dtype,features_list[1].shape)    
    elif feats_type == 1:
        for i in range(num_ntype):
            dim = 10
            num_nodes = (type_mask == i).sum()
            in_dims.append(dim)
            features_list.append(torch.zeros((num_nodes, 10)))
        features_list = torch.tensor(features_list)
        features_list = features_list.to(device)       
    train_pos_user_artist = train_val_test_pos_user_artist['train_pos_user_artist']
    val_pos_user_artist = train_val_test_pos_user_artist['val_pos_user_artist']
    test_pos_user_artist = train_val_test_pos_user_artist['test_pos_user_artist']
    train_neg_user_artist = train_val_test_neg_user_artist['train_neg_user_artist']
    val_neg_user_artist = train_val_test_neg_user_artist['val_neg_user_artist']
    test_neg_user_artist = train_val_test_neg_user_artist['test_neg_user_artist']
    y_true_test = np.array([1] * len(test_pos_user_artist) + [0] * len(test_neg_user_artist))

    auc_list = []
    ap_list = []
    for _ in range(repeat):
        net = MEHGNN_lp(
            [1, 1], 2, etypes_lists, in_dims, hidden_dim, hidden_dim, num_heads, attn_vec_dim, rnn_type, dropout_rate)
        net.to(device)
        optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)

        # training loop
        net.train()
        early_stopping = EarlyStopping(patience=patience, verbose=True, save_path='../checkpoint/checkpoint_{}.pt'.format(save_postfix))
        dur1 = []
        dur2 = []
        dur3 = []
        train_pos_idx_generator = index_generator(batch_size=batch_size, num_data=len(train_pos_user_artist))
        val_idx_generator = index_generator(batch_size=batch_size, num_data=len(val_pos_user_artist), shuffle=False)
        for epoch in range(num_epochs):
            t_start = time.time()
            # training
            net.train()
            for iteration in range(train_pos_idx_generator.num_iterations()):
                # forward
                #记录开始时间
                t0 = time.time()
                #获取训练正样本的批次索引
                train_pos_idx_batch = train_pos_idx_generator.next()
                #对索引进行排序
                train_pos_idx_batch.sort()
                
                #获取用户（CVE）和艺术家（Product Version）对的索引
                train_pos_user_artist_batch = train_pos_user_artist[train_pos_idx_batch].tolist()
                
                #获取负样本索引
                train_neg_idx_batch = np.random.choice(len(train_neg_user_artist), len(train_pos_idx_batch)) #从负样本中随机选择与正样本数量相等的索引
                train_neg_idx_batch.sort() #对负样本索引进行排序
                train_neg_user_artist_batch = train_neg_user_artist[train_neg_idx_batch].tolist() #获取负样本的用户（CVE）-艺术家（Product Version）对
                
                #解析最小批次 调用parse_minibatch_LastFM（）函数将正负样本分批次解析，同时获取元路径邻接列表、索引列表和映射索引列表
                train_pos_g_lists, train_pos_indices_lists, train_pos_idx_batch_mapped_lists = parse_minibatch_LastFM(
                    adjlists_ua, edge_metapath_indices_list_ua, train_pos_user_artist_batch, device, neighbor_samples, use_masks, num_user)
                
                #获取负样本用户（CVE）和艺术家（Product Version）的索引
                train_neg_g_lists, train_neg_indices_lists, train_neg_idx_batch_mapped_lists = parse_minibatch_LastFM(
                    adjlists_ua, edge_metapath_indices_list_ua, train_neg_user_artist_batch, device, neighbor_samples, no_masks, num_user)

                t1 = time.time()
                dur1.append(t1 - t0)
                #前向传播
                [pos_embedding_user, pos_embedding_artist], _ = net(
                    (train_pos_g_lists, features_list, type_mask, train_pos_indices_lists, train_pos_idx_batch_mapped_lists))
                [neg_embedding_user, neg_embedding_artist], _ = net(
                    (train_neg_g_lists, features_list, type_mask, train_neg_indices_lists, train_neg_idx_batch_mapped_lists))
               
                #计算损失
                #将正负样本的用户和艺术家嵌入向量进行reshape，用于批量矩阵乘法
                #分别计算正样本和负样本的输出
                #使用F.logsigmoid计算损失并求均值
                pos_embedding_user = pos_embedding_user.view(-1, 1, pos_embedding_user.shape[1])
                pos_embedding_artist = pos_embedding_artist.view(-1, pos_embedding_artist.shape[1], 1)
                neg_embedding_user = neg_embedding_user.view(-1, 1, neg_embedding_user.shape[1])
                neg_embedding_artist = neg_embedding_artist.view(-1, neg_embedding_artist.shape[1], 1)
                pos_out = torch.bmm(pos_embedding_user, pos_embedding_artist)
                neg_out = -torch.bmm(neg_embedding_user, neg_embedding_artist)
                train_loss = -torch.mean(F.logsigmoid(pos_out) + F.logsigmoid(neg_out))
                
                #记录计算损失所消耗的时间并添加到dur2中
                t2 = time.time()
                dur2.append(t2 - t1)
                
                #反向传播和优化
                # autograd
                optimizer.zero_grad() #清空梯度
                train_loss.backward() #计算梯度
                optimizer.step() #更新模型参数
                
                #记录反向传播和优化所消耗的时间并添加到dur3
                t3 = time.time()
                dur3.append(t3 - t2)

                # print training info
                if iteration % 100 == 0:
                    print(
                        'Epoch {:05d} | Iteration {:05d} | Train_Loss {:.4f} | Time1(s) {:.4f} | Time2(s) {:.4f} | Time3(s) {:.4f}'.format(
                            epoch, iteration, train_loss.item(), np.mean(dur1), np.mean(dur2), np.mean(dur3)))
            # validation
            net.eval()
            val_loss = []
            with torch.no_grad():
                for iteration in range(val_idx_generator.num_iterations()):
                    # forward
                    val_idx_batch = val_idx_generator.next()
                    val_pos_user_artist_batch = val_pos_user_artist[val_idx_batch].tolist()
                    val_neg_user_artist_batch = val_neg_user_artist[val_idx_batch].tolist()
                    val_pos_g_lists, val_pos_indices_lists, val_pos_idx_batch_mapped_lists = parse_minibatch_LastFM(
                        adjlists_ua, edge_metapath_indices_list_ua, val_pos_user_artist_batch, device, neighbor_samples, no_masks, num_user)
                    val_neg_g_lists, val_neg_indices_lists, val_neg_idx_batch_mapped_lists = parse_minibatch_LastFM(
                        adjlists_ua, edge_metapath_indices_list_ua, val_neg_user_artist_batch, device, neighbor_samples, no_masks, num_user)

                    [pos_embedding_user, pos_embedding_artist], _ = net(
                        (val_pos_g_lists, features_list, type_mask, val_pos_indices_lists, val_pos_idx_batch_mapped_lists))
                    [neg_embedding_user, neg_embedding_artist], _ = net(
                        (val_neg_g_lists, features_list, type_mask, val_neg_indices_lists, val_neg_idx_batch_mapped_lists))
                    pos_embedding_user = pos_embedding_user.view(-1, 1, pos_embedding_user.shape[1])
                    pos_embedding_artist = pos_embedding_artist.view(-1, pos_embedding_artist.shape[1], 1)
                    neg_embedding_user = neg_embedding_user.view(-1, 1, neg_embedding_user.shape[1])
                    neg_embedding_artist = neg_embedding_artist.view(-1, neg_embedding_artist.shape[1], 1)

                    pos_out = torch.bmm(pos_embedding_user, pos_embedding_artist)
                    neg_out = -torch.bmm(neg_embedding_user, neg_embedding_artist)
                    val_loss.append(-torch.mean(F.logsigmoid(pos_out) + F.logsigmoid(neg_out)))
                val_loss = torch.mean(torch.tensor(val_loss))
            t_end = time.time()
            # print validation info
            print('Epoch {:05d} | Val_Loss {:.4f} | Time(s) {:.4f}'.format(
                epoch, val_loss.item(), t_end - t_start))
            # early stopping
            early_stopping(val_loss, net)
            if early_stopping.early_stop:
                print('Early stopping!')
                break

        test_idx_generator = index_generator(batch_size=batch_size, num_data=len(test_pos_user_artist), shuffle=False)
        net.load_state_dict(torch.load('../checkpoint/checkpoint_{}.pt'.format(save_postfix)))
        net.eval()
        pos_proba_list = []
        neg_proba_list = []
        with torch.no_grad():
            for iteration in range(test_idx_generator.num_iterations()):
                # forward
                test_idx_batch = test_idx_generator.next()
                test_pos_user_artist_batch = test_pos_user_artist[test_idx_batch].tolist()
                test_neg_user_artist_batch = test_neg_user_artist[test_idx_batch].tolist()
                test_pos_g_lists, test_pos_indices_lists, test_pos_idx_batch_mapped_lists = parse_minibatch_LastFM(
                    adjlists_ua, edge_metapath_indices_list_ua, test_pos_user_artist_batch, device, neighbor_samples, no_masks, num_user)
                test_neg_g_lists, test_neg_indices_lists, test_neg_idx_batch_mapped_lists = parse_minibatch_LastFM(
                    adjlists_ua, edge_metapath_indices_list_ua, test_neg_user_artist_batch, device, neighbor_samples, no_masks, num_user)

                [pos_embedding_user, pos_embedding_artist], _ = net(
                    (test_pos_g_lists, features_list, type_mask, test_pos_indices_lists, test_pos_idx_batch_mapped_lists))
                [neg_embedding_user, neg_embedding_artist], _ = net(
                    (test_neg_g_lists, features_list, type_mask, test_neg_indices_lists, test_neg_idx_batch_mapped_lists))
                pos_embedding_user = pos_embedding_user.view(-1, 1, pos_embedding_user.shape[1])
                pos_embedding_artist = pos_embedding_artist.view(-1, pos_embedding_artist.shape[1], 1)
                neg_embedding_user = neg_embedding_user.view(-1, 1, neg_embedding_user.shape[1])
                neg_embedding_artist = neg_embedding_artist.view(-1, neg_embedding_artist.shape[1], 1)

                pos_out = torch.bmm(pos_embedding_user, pos_embedding_artist).flatten()
                neg_out = torch.bmm(neg_embedding_user, neg_embedding_artist).flatten()
                pos_proba_list.append(torch.sigmoid(pos_out))
                neg_proba_list.append(torch.sigmoid(neg_out))
            y_proba_test = torch.cat(pos_proba_list + neg_proba_list)
            y_proba_test = y_proba_test.cpu().numpy()
        auc = roc_auc_score(y_true_test, y_proba_test)
        ap = average_precision_score(y_true_test, y_proba_test)
        
        
        top_1_accuracy = top_k_accuracy_score(y_true_test, y_proba_test, k=1)
        top_3_accuracy = top_k_accuracy_score(y_true_test, y_proba_test, k=3)
        top_10_accuracy = top_k_accuracy_score(y_true_test, y_proba_test, k=10)
        
        print('Link Prediction Test')
        print('AUC = {}'.format(auc))
        print('AP = {}'.format(ap))
        
        
        print("hit@1: ", top_1_accuracy)
        print("hit@3: ", top_3_accuracy)
        print("hit@10: ", top_10_accuracy)
        
        #分界阈值为0.2，如果大于0.2模型则认为
        y_pred_bull = [1 if x > 0.2 else 0 for x in y_proba_test]
        
        # 计算准确率、精确度和召回率
        acc = accuracy_score(y_true_test, y_pred_bull)
        pre = precision_score(y_true_test, y_pred_bull)
        recall = recall_score(y_true_test, y_pred_bull)
        f1 = f1_score(y_true_test, y_pred_bull)
        # 输出评价指标
        print("Accuracy: ", acc)
        print("Precision: ", pre)
        print("Recall: ", recall)
        print("F1: ", f1)


        auc_list.append(auc)
        ap_list.append(ap)

    print('----------------------------------------------------------------')
    print('Link Prediction Tests Summary')
    print('AUC_mean = {}, AUC_std = {}'.format(np.mean(auc_list), np.std(auc_list)))
    print('AP_mean = {}, AP_std = {}'.format(np.mean(ap_list), np.std(ap_list)))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='MRGNN testing for the recommendation dataset')
    ap.add_argument('--feats-type', type=int, default=0,
                    help='Type of the node features used. ' +
                         '0 - all id vectors; ' +
                         '1 - all zero vector. Default is 0.')
    ap.add_argument('--hidden-dim', type=int, default=64, help='Dimension of the node hidden state. Default is 64.')
    ap.add_argument('--num-heads', type=int, default=8, help='Number of the attention heads. Default is 8.')
    ap.add_argument('--attn-vec-dim', type=int, default=128, help='Dimension of the attention vector. Default is 128.')
    ap.add_argument('--rnn-type', default='RotatE0', help='Type of the aggregator. Default is RotatE0.')
    ap.add_argument('--epoch', type=int, default=100, help='Number of epochs. Default is 100.')
    ap.add_argument('--patience', type=int, default=5, help='Patience. Default is 5.')
    ap.add_argument('--batch-size', type=int, default=64, help='Batch size. Default is 8.')
    ap.add_argument('--samples', type=int, default=100, help='Number of neighbors sampled. Default is 100.')
    ap.add_argument('--repeat', type=int, default=1, help='Repeat the training and testing for N times. Default is 1.')
    ap.add_argument('--save-postfix', default='LastFM', help='Postfix for the saved model and result. Default is LastFM.')

    args = ap.parse_args()
    run_model_LastFM(args.feats_type, args.hidden_dim, args.num_heads, args.attn_vec_dim, args.rnn_type, args.epoch,
                     args.patience, args.batch_size, args.samples, args.repeat, args.save_postfix)

