import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch_scatter import scatter_sum, scatter_softmax
from collections import defaultdict
from tqdm import tqdm

# 初始化模型的参数
init = nn.init.xavier_uniform_

class RGAT(nn.Module):
    """
    基于关系的图注意力网络类。
    
    参数:
    - latdim: 嵌入向量的维度。
    - n_hops: 信息传递的轮数。
    - mess_dropout_rate: 消息传递过程中dropout的概率。
    """
    
    def __init__(self, latdim, n_hops, entity_n,relation_n,mess_dropout_rate=0.4):
        super(RGAT, self).__init__()
        self.mess_dropout_rate = mess_dropout_rate
        self.W = nn.Parameter(init(torch.empty(size=(2*latdim, latdim)), gain=nn.init.calculate_gain('relu')))
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.n_hops = n_hops
        self.dropout = nn.Dropout(p=mess_dropout_rate)
        self.entity_emb = nn.Parameter(init(torch.empty(entity_n, latdim))).cuda()
        self.relation_emb = nn.Parameter(init(torch.empty(relation_n, latdim))).cuda()

    def agg(self, kg):
        """
        对实体嵌入向量进行聚合。
        
        参数:
        - entity_emb: 实体嵌入向量。
        - relation_emb: 关系嵌入向量。
        - kg: 知识图谱数据，包含边索引和类型。
        
        返回:
        - 聚合后的实体嵌入向量。
        """
        edge_index, edge_type = kg
        head, tail = edge_index
        a_input = torch.cat([self.entity_emb[head], self.entity_emb[tail]], dim=-1)
        e_input = torch.multiply(torch.mm(a_input, self.W), self.relation_emb[edge_type]).sum(-1)
        e = self.leakyrelu(e_input)
        e = scatter_softmax(e, head, dim=0, dim_size=self.entity_emb.shape[0])
        agg_emb = self.entity_emb[tail] * e.view(-1, 1)
        agg_emb = scatter_sum(agg_emb, head, dim=0, dim_size=self.entity_emb.shape[0])
        agg_emb = agg_emb + self.entity_emb
        return agg_emb
        
    def forward(self,  kg, mess_dropout=True):
        """
        基于关系的图注意力网络的前向传播函数。
        
        参数:
        - entity_emb: 实体嵌入向量。
        - relation_emb: 关系嵌入向量。
        - kg: 知识图谱数据，包含边索引和类型。
        - mess_dropout: 是否在消息传递过程中使用dropout。
        
        返回:
        - 经过信息传递更新后的实体嵌入向量。
        """
        entity_res_emb = self.entity_emb
        for _ in range(self.n_hops):
            self.entity_emb = self.agg(kg)
            if mess_dropout:
                self.entity_emb = self.dropout(self.entity_emb)
            self.entity_emb = F.normalize(self.entity_emb)
            entity_res_emb = 0.5 * entity_res_emb + self.entity_emb
        return entity_res_emb
class KGHandler:
    def __init__(self, kg_file):
        self.kg_file = kg_file
        self.kg_dict = defaultdict(list)
        self.kg_edges = []
        self.entity_n = 0
        self.relation_n = 0

    def read_kg(self):
        triplets = np.loadtxt(self.kg_file, dtype=np.int32)
        triplets = np.unique(triplets, axis=0)
        inv_triplets = triplets.copy()
        inv_triplets[:, 0] = triplets[:, 2]
        inv_triplets[:, 2] = triplets[:, 0]
        inv_triplets[:, 1] = triplets[:, 1] + triplets[:, 1].max() + 1
        all_triplets = np.concatenate((triplets, inv_triplets), axis=0)

        self.entity_n = max(max(all_triplets[:, 0]), max(all_triplets[:, 2])) + 1
        self.relation_n = max(all_triplets[:, 1]) + 1

        for h, r, t in all_triplets:
            self.kg_dict[h].append((r, t))
            self.kg_edges.append([h, t, r])

    def get_edges(self):
        edges = torch.tensor(self.kg_edges)
        edge_index = edges[:, :-1].t().long().cuda()
        edge_type = edges[:, -1].long().cuda()
        return edge_index, edge_type

