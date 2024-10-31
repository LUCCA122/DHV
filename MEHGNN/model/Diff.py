import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch_scatter import scatter_sum, scatter_softmax
import scipy.sparse as sp
from collections import defaultdict
import math
import pickle
import torch.utils.data as data
from scipy.sparse import coo_matrix
import scipy.sparse as sp
import torch.utils.data as dataloader

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
    
    def __init__(self, latdim, n_hops, mess_dropout_rate=0.4):
        super(RGAT, self).__init__()
        self.mess_dropout_rate = mess_dropout_rate
        self.W = nn.Parameter(init(torch.empty(size=(2*latdim, latdim)), gain=nn.init.calculate_gain('relu')))
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.n_hops = n_hops
        self.dropout = nn.Dropout(p=mess_dropout_rate)

    def agg(self, entity_emb, relation_emb, kg):
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
        a_input = torch.cat([entity_emb[head], entity_emb[tail]], dim=-1)
        e_input = torch.mul(torch.mm(a_input, self.W), relation_emb[edge_type]).sum(-1)
        e = self.leakyrelu(e_input)
        e = scatter_softmax(e, head, dim=0, dim_size=entity_emb.shape[0])
        agg_emb = entity_emb[tail] * e.view(-1, 1)
        agg_emb = scatter_sum(agg_emb, head, dim=0, dim_size=entity_emb.shape[0])
        agg_emb = agg_emb + entity_emb
        return agg_emb
        
    def forward(self, entity_emb, relation_emb, kg, mess_dropout=True):
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
        entity_res_emb = entity_emb
        for _ in range(self.n_hops):
            entity_emb = self.agg(entity_emb, relation_emb, kg)
            if mess_dropout:
                entity_emb = self.dropout(entity_emb)
            entity_emb = F.normalize(entity_emb)
            entity_res_emb = 0.5 * entity_res_emb + entity_emb
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


        kg_counter_dict = {}    

        for h_id, r_id, t_id in all_triplets:
            if h_id not in kg_counter_dict.keys():
                kg_counter_dict[h_id] = set()
            if t_id not in kg_counter_dict[h_id]:
                kg_counter_dict[h_id].add(t_id)
            else:
                continue
            self.kg_edges.append([h_id, t_id, r_id])
            self.kg_dict[h_id].append((r_id, t_id))    

    def get_edges(self):
        edges = torch.tensor(self.kg_edges)
        edge_index = edges[:, :-1].t().long().cuda()
        edge_type = edges[:, -1].long().cuda()
        return edge_index, edge_type

    
    def buildKGMatrix(self, kg_edges):
        edge_list = []
        for h_id, t_id, r_id in kg_edges:
            edge_list.append((h_id, t_id))
        edge_list = np.array(edge_list)

        kgMatrix = sp.csr_matrix((np.ones_like(edge_list[:,0]), (edge_list[:,0], edge_list[:,1])), dtype='float64', shape=(self.entity_n, self.entity_n))

        return kgMatrix

    def normalizeAdj(self, mat): 
        degree = np.array(mat.sum(axis=-1))
        dInvSqrt = np.reshape(np.power(degree, -0.5), [-1])
        dInvSqrt[np.isinf(dInvSqrt)] = 0.0
        dInvSqrtMat = sp.diags(dInvSqrt)
        return mat.dot(dInvSqrtMat).transpose().dot(dInvSqrtMat).tocoo()

    def makeTorchAdj(self, mat):
        # make ui adj
        user,item = mat.shape
        a = sp.csr_matrix((user, user))
        b = sp.csr_matrix((item, item))
        mat = sp.vstack([sp.hstack([a, mat]), sp.hstack([mat.transpose(), b])])
        mat = (mat != 0) * 1.0
        mat = (mat + sp.eye(mat.shape[0])) * 1.0
        mat = self.normalizeAdj(mat)

        # make cuda tensor
        idxs = torch.from_numpy(np.vstack([mat.row, mat.col]).astype(np.int64))
        vals = torch.from_numpy(mat.data.astype(np.float32))
        shape = torch.Size(mat.shape)
        return torch.sparse.FloatTensor(idxs, vals, shape).cuda()

    def RelationDictBuild(self):
        relation_dict = {}
        for head in self.kg_dict:
            relation_dict[head] = {}
            for (relation, tail) in self.kg_dict[head]:
                relation_dict[head][tail] = relation
        return relation_dict

    def buildUIMatrix(self, mat):
        idxs = torch.from_numpy(np.vstack([mat.row, mat.col]).astype(np.int64))
        vals = torch.from_numpy(mat.data.astype(np.float32))
        shape = torch.Size(mat.shape)
        return torch.sparse.FloatTensor(idxs, vals, shape).cuda()

    def LoadData(self,trnMat,batch_size):
        self.read_kg()
        trnMat = trnMat.tocoo()

        self.torchBiAdj = self.makeTorchAdj(trnMat)

        self.ui_matrix = self.buildUIMatrix(trnMat)# 构建pytorch的用户-物品交互稀疏矩阵，并存储到gpu上
        
        
        '''
        构建知识图谱的稀疏矩阵，并存储到gpu上，形状为(entity_n, entity_n)
        每个三元组中的头实体和尾实体的id作为矩阵的行和列，值为1
        '''
        self.kg_matrix = self.buildKGMatrix(self.kg_edges)
        print("kg shape: ", self.kg_matrix.shape)
        print("number of edges in KG: ", len(self.kg_edges))
        
        self.diffusionData = DiffusionData(torch.FloatTensor(self.kg_matrix.A))# kg矩阵作为扩散矩阵
        self.diffusionLoader = dataloader.DataLoader(self.diffusionData, batch_size=batch_size, shuffle=True, num_workers=0)

        self.relation_dict = self.RelationDictBuild()




'''
# 参数设置
latdim = 64  # 嵌入向量维度
n_hops = 2   # 信息传播的轮数
kg_file = 'kg.txt'  # 知识图谱文件路径

# 加载和处理知识图谱
handler = KGHandler(kg_file)
handler.read_kg()

# 初始化模型
rgat = RGAT(latdim=latdim, n_hops=n_hops).cuda()

# 创建实体和关系的嵌入向量
entity_emb = nn.Parameter(init(torch.empty(handler.entity_n, latdim))).cuda()
relation_emb = nn.Parameter(init(torch.empty(handler.relation_n, latdim))).cuda()

# 获取边索引和关系类型
kg = handler.get_edges()

# 使用RGAT模型进行传播
entity_embeddings = rgat(entity_emb, relation_emb, kg)

print("实体嵌入向量：", entity_embeddings)
'''


class Denoise(nn.Module):
    def __init__(self, in_dims, out_dims, emb_size, norm=False, dropout=0.5):
        super(Denoise, self).__init__()
        self.in_dims = in_dims
        self.out_dims = out_dims
        self.time_emb_dim = emb_size
        self.norm = norm

        self.emb_layer = nn.Linear(self.time_emb_dim, self.time_emb_dim)

        in_dims_temp = [self.in_dims[0] + self.time_emb_dim] + self.in_dims[1:]

        out_dims_temp = self.out_dims

        self.in_layers = nn.ModuleList([nn.Linear(d_in, d_out) for d_in, d_out in zip(in_dims_temp[:-1], in_dims_temp[1:])])
        self.out_layers = nn.ModuleList([nn.Linear(d_in, d_out) for d_in, d_out in zip(out_dims_temp[:-1], out_dims_temp[1:])])

        self.drop = nn.Dropout(dropout)
        self.init_weights()

    def init_weights(self):
        for layer in self.in_layers:
            size = layer.weight.size()
            std = np.sqrt(2.0 / (size[0] + size[1]))
            layer.weight.data.normal_(0.0, std)
            layer.bias.data.normal_(0.0, 0.001)
        
        for layer in self.out_layers:
            size = layer.weight.size()
            std = np.sqrt(2.0 / (size[0] + size[1]))
            layer.weight.data.normal_(0.0, std)
            layer.bias.data.normal_(0.0, 0.001)

        size = self.emb_layer.weight.size()
        std = np.sqrt(2.0 / (size[0] + size[1]))
        self.emb_layer.weight.data.normal_(0.0, std)
        self.emb_layer.bias.data.normal_(0.0, 0.001)

    def forward(self, x, timesteps, mess_dropout=True):
        freqs = torch.exp(-math.log(10000) * torch.arange(start=0, end=self.time_emb_dim//2, dtype=torch.float32) / (self.time_emb_dim//2)).cuda()
        temp = timesteps[:, None].float() * freqs[None]
        time_emb = torch.cat([torch.cos(temp), torch.sin(temp)], dim=-1)
        if self.time_emb_dim % 2:
            time_emb = torch.cat([time_emb, torch.zeros_like(time_emb[:, :1])], dim=-1)
        emb = self.emb_layer(time_emb)
        if self.norm:
            x = F.normalize(x)
        if mess_dropout:
            x = self.drop(x)
        h = torch.cat([x, emb], dim=-1)
        for i, layer in enumerate(self.in_layers):
            h = layer(h)
            h = torch.tanh(h)
        for i, layer in enumerate(self.out_layers):
            h = layer(h)
            if i != len(self.out_layers) - 1:
                h = torch.tanh(h)

        return h

class GaussianDiffusion(nn.Module):
    def __init__(self, noise_scale, noise_min, noise_max, steps, beta_fixed=True):
        super(GaussianDiffusion, self).__init__()

        self.noise_scale = noise_scale
        self.noise_min = noise_min
        self.noise_max = noise_max
        self.steps = steps

        if noise_scale != 0:
            self.betas = torch.tensor(self.get_betas(), dtype=torch.float64).cuda()
            if beta_fixed:
                self.betas[0] = 0.0001

            self.calculate_for_diffusion()

    def get_betas(self):
        start = self.noise_scale * self.noise_min
        end = self.noise_scale * self.noise_max
        variance = np.linspace(start, end, self.steps, dtype=np.float64)
        alpha_bar = 1 - variance
        betas = []
        betas.append(1 - alpha_bar[0])
        for i in range(1, self.steps):
            betas.append(min(1 - alpha_bar[i] / alpha_bar[i-1], 0.999))
        return np.array(betas)
    
    def calculate_for_diffusion(self):
        alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(alphas, axis=0).cuda()
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0]).cuda(), self.alphas_cumprod[:-1]]).cuda()
        self.alphas_cumprod_next = torch.cat([self.alphas_cumprod[1:], torch.tensor([0.0]).cuda()]).cuda()

        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = torch.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)

        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_log_variance_clipped = torch.log(torch.cat([self.posterior_variance[1].unsqueeze(0), self.posterior_variance[1:]]))
        self.posterior_mean_coef1 = (self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod))
        self.posterior_mean_coef2 = ((1.0 - self.alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - self.alphas_cumprod))

    def p_sample(self, model, x_start, steps):
        if steps == 0:
            x_t = x_start
        else:
            t = torch.tensor([steps-1] * x_start.shape[0]).cuda()
            x_t = self.q_sample(x_start, t)
        
        indices = list(range(self.steps))[::-1]

        for i in indices:
            t = torch.tensor([i] * x_t.shape[0]).cuda()
            model_mean, model_log_variance = self.p_mean_variance(model, x_t, t)
            x_t = model_mean
        return x_t
            
    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        return self._extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start + self._extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
    
    def _extract_into_tensor(self, arr, timesteps, broadcast_shape):
        arr = arr.cuda()
        res = arr[timesteps].float()
        while len(res.shape) < len(broadcast_shape):
            res = res[..., None]
        return res.expand(broadcast_shape)
    
    def p_mean_variance(self, model, x, t):
        model_output = model(x, t, False)

        model_variance = self.posterior_variance
        model_log_variance = self.posterior_log_variance_clipped

        model_variance = self._extract_into_tensor(model_variance, t, x.shape)
        model_log_variance = self._extract_into_tensor(model_log_variance, t, x.shape)

        model_mean = (self._extract_into_tensor(self.posterior_mean_coef1, t, x.shape) * model_output + self._extract_into_tensor(self.posterior_mean_coef2, t, x.shape) * x)
        
        return model_mean, model_log_variance

    def training_losses(self, model, x_start, ui_matrix, userEmbeds, itmEmbeds, batch_index):
        item =ui_matrix.shape[1]
        batch_size = x_start.size(0)
        ts = torch.randint(0, self.steps, (batch_size,)).long().cuda()
        noise = torch.randn_like(x_start)
        if self.noise_scale != 0:
            x_t = self.q_sample(x_start, ts, noise)
        else:
            x_t = x_start

        model_output = model(x_t, ts)
        mse = self.mean_flat((x_start - model_output) ** 2)

        weight = self.SNR(ts - 1) - self.SNR(ts)
        weight = torch.where((ts == 0), 1.0, weight)

        diff_loss = weight * mse
        
        item_user_matrix = torch.spmm(ui_matrix, model_output[:, :item].t()).t()
        itmEmbeds_kg = torch.mm(item_user_matrix, userEmbeds)
        ukgc_loss = self.mean_flat((itmEmbeds_kg - itmEmbeds[batch_index]) ** 2)

        return diff_loss, ukgc_loss
        
    def mean_flat(self, tensor):
        return tensor.mean(dim=list(range(1, len(tensor.shape))))
    
    def SNR(self, t):
        self.alphas_cumprod = self.alphas_cumprod.cuda()
        return self.alphas_cumprod[t] / (1 - self.alphas_cumprod[t])
    
class DiffusionData(data.Dataset):
	def __init__(self, data):
		self.data = data
	
	def __getitem__(self, index):
		item = self.data[index]
		return item, index
	
	def __len__(self):
		return len(self.data)    