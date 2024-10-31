import torch
import torch.nn as nn
import numpy as np
from model.rgat import RGAT
from model.base_MEHGNN import MEHGNN_ctr_ntype_specific
init = nn.init.xavier_uniform_


# for link prediction task
class MEHGNN_lp_layer(nn.Module):
    def __init__(self,
                 num_metapaths_list,
                 num_edge_type,
                 etypes_lists,
                 in_dim,
                 out_dim,
                 num_heads,
                 attn_vec_dim,
                 rnn_type='gru',
                 attn_drop=0.5):
        super(MEHGNN_lp_layer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads

        # etype-specific parameters
        r_vec = None
        if rnn_type == 'TransE0':
            r_vec = nn.Parameter(torch.empty(size=(num_edge_type // 2, in_dim)))
        elif rnn_type == 'TransE1':
            r_vec = nn.Parameter(torch.empty(size=(num_edge_type, in_dim)))
        elif rnn_type == 'RotatE0':
            r_vec = nn.Parameter(torch.empty(size=(num_edge_type // 2, in_dim // 2, 2)))
        elif rnn_type == 'RotatE1':
            r_vec = nn.Parameter(torch.empty(size=(num_edge_type, in_dim // 2, 2)))
        if r_vec is not None:
            nn.init.xavier_normal_(r_vec.data, gain=1.414)

        # ctr_ntype-specific layers
        self.user_layer = MEHGNN_ctr_ntype_specific(num_metapaths_list[0],
                                                   etypes_lists[0],
                                                   in_dim,
                                                   num_heads,
                                                   attn_vec_dim,
                                                   rnn_type,
                                                   r_vec,
                                                   attn_drop,
                                                   use_minibatch=True)
        self.item_layer = MEHGNN_ctr_ntype_specific(num_metapaths_list[1],
                                                   etypes_lists[1],
                                                   in_dim,
                                                   num_heads,
                                                   attn_vec_dim,
                                                   rnn_type,
                                                   r_vec,
                                                   attn_drop,
                                                   use_minibatch=True)

        # note that the acutal input dimension should consider the number of heads
        # as multiple head outputs are concatenated together
        self.fc_user = nn.Linear(in_dim * num_heads, out_dim, bias=True)
        self.fc_item = nn.Linear(in_dim * num_heads, out_dim, bias=True)
        nn.init.xavier_normal_(self.fc_user.weight, gain=1.414)
        nn.init.xavier_normal_(self.fc_item.weight, gain=1.414)

    def forward(self, inputs):
        g_lists, features, type_mask, edge_metapath_indices_lists, target_idx_lists = inputs

        # ctr_ntype-specific layers
        h_user = self.user_layer(
            (g_lists[0], features, type_mask, edge_metapath_indices_lists[0], target_idx_lists[0]))
        h_item = self.item_layer(
            (g_lists[1], features, type_mask, edge_metapath_indices_lists[1], target_idx_lists[1]))

        logits_user = self.fc_user(h_user)
        logits_item = self.fc_item(h_item)
        return [logits_user, logits_item], [h_user, h_item]


class MEHGNN_lp1(nn.Module):
    def __init__(self,
                 num_metapaths_list,
                 num_edge_type,
                 etypes_lists,
                 feats_dim_list,
                 hidden_dim,
                 out_dim,
                 num_heads,
                 attn_vec_dim,
                 n_hops,
                 entity_n,
                 relation_n,
                 rnn_type='gru',
                 dropout_rate=0.5):
        super(MEHGNN_lp1, self).__init__()
        self.hidden_dim = hidden_dim

        # ntype-specific transformation
        self.fc_list = nn.ModuleList([nn.Linear(feats_dim, hidden_dim, bias=True) for feats_dim in feats_dim_list])
        # feature dropout after trainsformation
        if dropout_rate > 0:
            self.feat_drop = nn.Dropout(dropout_rate)
        else:
            self.feat_drop = lambda x: x
        # initialization of fc layers
        for fc in self.fc_list:
            nn.init.xavier_normal_(fc.weight, gain=1.414)

        # MEHGNN_lp layers
        self.layer1 = MEHGNN_lp_layer(num_metapaths_list,
                                     num_edge_type,
                                     etypes_lists,
                                     hidden_dim,
                                     out_dim,
                                     num_heads,
                                     attn_vec_dim,
                                     rnn_type,
                                     attn_drop=dropout_rate)
        
        self.rgat = RGAT(latdim=hidden_dim, n_hops=n_hops)    
        self.entity_emb = nn.Parameter(init(torch.empty(entity_n, hidden_dim))) #实体嵌入
        self.relation_emb = nn.Parameter(init(torch.empty(relation_n, hidden_dim))) #关系嵌入     
        self.pvs_embeddings = nn.Parameter(init(torch.empty(feats_dim_list[1], hidden_dim))) # 组件嵌入    
        
                  

    def forward(self, inputs,kg):
        g_lists, features_list, type_mask, edge_metapath_indices_lists, target_idx_lists = inputs

        # ntype-specific transformation
        '''
        transformed_features = torch.zeros(type_mask.shape[0], self.hidden_dim, device=features_list[0].device)
        for i, fc in enumerate(self.fc_list):
            node_indices = np.where(type_mask == i)[0]
            transformed_features[node_indices] = fc(features_list[i])
        transformed_features = self.feat_drop(transformed_features)
        '''
        
        
        num_cves = (type_mask==0).sum()
        entity_embeddings = self.rgat(self.entity_emb, self.relation_emb, kg) 
        cves_embeddings = entity_embeddings[:num_cves]
        transformed_features = torch.cat([cves_embeddings, self.pvs_embeddings], dim=0)
        transformed_features = self.feat_drop(transformed_features)
        
        
        
        
        # hidden layers
        [logits_user, logits_item], [h_user, h_item] = self.layer1(
            (g_lists, transformed_features, type_mask, edge_metapath_indices_lists, target_idx_lists))

        return [logits_user, logits_item], [h_user, h_item]
    
    def getEntityEmbeds(self):
        return self.entity_emb

    def getUserEmbeds(self):
        return self.pvs_embeddings
