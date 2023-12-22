import torch
import itertools
import torch.nn as nn
from torch_geometric.nn import GINConv
import torch.nn.functional as F

def log_sinkhorn_norm(log_alpha: torch.Tensor, n_iter: int =20):
    '''
    Note: sinkhorn算子的行列做归一化操作 
    Para:   
        log_alpha         目标矩阵
        n_iter            归一化迭代次数
    Return: 返回一个双随机矩阵 取log每个数除sum就变成log(a) - log(sum)
    '''
    for _ in range(n_iter):
        log_alpha = log_alpha - torch.logsumexp(log_alpha, -1, keepdim=True)
        log_alpha = log_alpha - torch.logsumexp(log_alpha, -2, keepdim=True)
    return log_alpha.exp()

def gumbel_sinkhorn(log_alpha: torch.Tensor, tau: float = 0.001, n_iter: int = 20, noise: bool = True):
    '''
    Note: gumbel_sinkhorn算法
    Para:   
        log_alpha         目标矩阵
        tau               论文中temperature 超参数 越小越精准
        n_iter            归一化迭代次数
        noise             添加噪声使数据不一致 保证收敛
    Return: 返回一个双随机矩阵 取log
    '''
    if noise:
        uniform_noise = torch.rand_like(log_alpha)
        gumbel_noise = -torch.log(-torch.log(uniform_noise+1e-20)+1e-20)
        log_alpha = (log_alpha + gumbel_noise)/tau         #原文公式4
    sampled_perm_mat = log_sinkhorn_norm(log_alpha, n_iter)
    return sampled_perm_mat

class GSModel(nn.Module):
    def __init__(self, config, n_feat):
        super(GSModel, self).__init__()
        self.config = config
        self.n_feat = n_feat
        self.setup_layers()
        
    def setup_layers(self):
        gnn_enc = self.config['gnn_encoder']
        self.filters = self.config['gnn_filters']
        self.num_filter = len(self.filters)

        self.tau = self.config['tau']
        self.n_sink_iter = self.config['n_sink_iter']
        self.n_samples = self.config['n_samples']
        self.use_multi_p = True if self.n_samples > 1 else False

        self.max_set_size  = self.config['max_set_size']
        self.trans_out_dim = self.filters[self.num_filter - 1]
        self.fc_transform1 = nn.Linear(self.trans_out_dim, self.trans_out_dim)
        self.relu1 = nn.ReLU()
        self.fc_transform2 = nn.Linear(self.trans_out_dim, self.trans_out_dim)
        self.graph_size_to_mask_map = [torch.cat((torch.tensor([1]).repeat(x,1).repeat(1, self.trans_out_dim), \
        torch.tensor([0]).repeat(self.max_set_size-x,1).repeat(1, self.trans_out_dim))) for x in range(0,self.max_set_size+1)]

        self.gnn_list = nn.ModuleList()

        self.W = nn.Parameter(torch.Tensor(self.trans_out_dim, self.trans_out_dim))
        self.reset_parameters()

        self.input_mlp1  = nn.Linear(self.max_set_size, self.max_set_size)
        self.input_relu  = nn.ReLU()
        self.input_mlp2  = nn.Linear(self.max_set_size, self.max_set_size) 
        

        if gnn_enc == 'GIN':
            self.gnn_list.append(GINConv(torch.nn.Sequential(
                torch.nn.Linear(self.n_feat, self.filters[0]),
                torch.nn.ReLU(),
                torch.nn.Linear(self.filters[0], self.filters[0]),
                torch.nn.BatchNorm1d(self.filters[0]),
            ), eps=True))

            for i in range(self.num_filter - 1):
                self.gnn_list.append(GINConv(torch.nn.Sequential(
                    torch.nn.Linear(self.filters[i], self.filters[i + 1]),
                    torch.nn.ReLU(),
                    torch.nn.Linear(self.filters[i + 1], self.filters[i + 1]),
                    torch.nn.BatchNorm1d(self.filters[i + 1]),
                ), eps=True))
        else:
            raise NotImplementedError("Unknown GNN-Operator.")

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W) 

    def convolutional_pass_level(self, enc, edge_index, x):
        feat = enc(x, edge_index)
        feat = F.relu(feat)
        feat = F.dropout(feat, p=self.config['dropout'], training=self.training)
        return feat

    def forward(self, data):
        features_1    = data["g1"].x.cuda()
        features_2    = data["g2"].x.cuda()
        edge_index_1  = data['g1'].edge_index.cuda()
        edge_index_2  = data['g2'].edge_index.cuda()

        batch_1_size  = [len(list(group)) for _, group in itertools.groupby(data["g1"].batch)]
        batch_2_size  = [len(list(group)) for _, group in itertools.groupby(data["g2"].batch)]

        conv_source_1 = torch.clone(features_1)
        conv_source_2 = torch.clone(features_2)

        for i in range(self.num_filter):
            conv_source_1 = self.convolutional_pass_level(self.gnn_list[i], edge_index_1, conv_source_1)
            conv_source_2 = self.convolutional_pass_level(self.gnn_list[i], edge_index_2, conv_source_2)

        permuted_score   = 0
        features_split_1 = torch.split(conv_source_1, batch_1_size, dim=0)
        features_split_2 = torch.split(conv_source_2, batch_2_size, dim=0)

        stacked_g1_emb   = torch.stack([F.pad(x, pad=(0, 0, 0, self.max_set_size-x.shape[0])) for x in features_split_1])
        stacked_g2_emb   = torch.stack([F.pad(x, pad=(0, 0, 0, self.max_set_size-x.shape[0])) for x in features_split_2])

        transformed_g1_emb = self.fc_transform2(self.relu1(self.fc_transform1(stacked_g1_emb)))
        transformed_g2_emb = self.fc_transform2(self.relu1(self.fc_transform1(stacked_g2_emb)))  

        g1_mask = torch.stack([self.graph_size_to_mask_map[i] for i in batch_1_size]).cuda()
        g2_mask = torch.stack([self.graph_size_to_mask_map[i] for i in batch_2_size]).cuda()

        masked_g1_emb = torch.mul(g1_mask,transformed_g1_emb)
        masked_g2_emb = torch.mul(g2_mask,transformed_g2_emb)

        sinkhorn_input = torch.matmul(torch.matmul(masked_g1_emb, self.W), masked_g2_emb.permute(0,2,1))
        
        if self.use_multi_p:
            multi_head_input  = [
                self.input_mlp2(self.input_relu(self.input_mlp1(sinkhorn_input)))
                for _ in range(self.n_samples)
            ]

            gumbel_sinkhorn_mat = torch.stack([
                gumbel_sinkhorn(input_mat, self.tau, self.n_sink_iter)
                for input_mat in multi_head_input
            ]) 
            #  Multi-Head
            permuted_list = []
            for gs_perm in gumbel_sinkhorn_mat:
                p_score= torch.sum(stacked_g1_emb - gs_perm@stacked_g2_emb,dim=(1,2))
                permuted_score = (p_score - p_score.min()) / (p_score.max() - p_score.min())
                permuted_list.append(torch.exp(-permuted_score))
            permuted_score = torch.stack(permuted_list, dim=0)
            prediction = torch.mean(permuted_score, dim=0, keepdim=True)[0]
            return prediction
        else:
            transport_plan = gumbel_sinkhorn(sinkhorn_input, self.tau, self.n_sink_iter)

            p_score= torch.sum(stacked_g1_emb - transport_plan@stacked_g2_emb,dim=(1,2))
            permuted_score = (p_score - p_score.min()) / (p_score.max() - p_score.min())
            prediction = torch.exp(-permuted_score)
            return prediction

if __name__ == "__main__":
    pass