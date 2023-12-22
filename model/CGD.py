import torch
import torch.nn as nn
from torch_geometric.nn import GINConv
import torch.nn.functional as F
from torch_geometric.nn.glob import global_add_pool, global_mean_pool
from model.layers import MLPLayers, TensorNetworkModule
from model.GSModel import GSModel

class CGD(nn.Module):
    def __init__(self, config, n_feat):
        super(CGD, self).__init__()
        self.config = config
        self.n_feat = n_feat
        self.x_augment = config.get('x_augment', False)
        if self.x_augment:
            self.max_node = config.get('max_set_size', 0)
        self.setup_layers()
        self.setup_score_layer()
        self.scale_init()
        
    def setup_layers(self):
        gnn_enc = self.config['gnn_encoder']
        self.filters = self.config['gnn_filters']
        self.num_filter = len(self.filters)
        self.use_gs  = self.config.get('use_gs', False)

        self.trans_out_dim = self.filters[self.num_filter - 1]
        self.fc_transform1 = torch.nn.Linear(self.trans_out_dim, self.trans_out_dim)
        self.relu1 = torch.nn.ReLU()
        self.fc_transform2 = torch.nn.Linear(self.trans_out_dim, self.trans_out_dim)

        self.gnn_list = nn.ModuleList()
        self.mlp_list_inner = nn.ModuleList()
        self.mlp_list_outer = nn.ModuleList()
        self.NTN_list = nn.ModuleList()

        self.core_mlp_list = nn.ModuleList()
        self.diff_mlp_list = nn.ModuleList()

        if gnn_enc == 'GIN':
            self.gnn_list.append(GINConv(torch.nn.Sequential(
                # n_feat - max_node 是原feature + k-core过一次mlp的dim
                torch.nn.Linear(self.n_feat - self.max_node + self.filters[0], self.filters[0]) if self.x_augment else
                torch.nn.Linear(self.n_feat, self.filters[0]),
                torch.nn.ReLU(),
                torch.nn.Linear(self.filters[0], self.filters[0]),
                torch.nn.BatchNorm1d(self.filters[0]),
            ), eps=True))

            isAugment = 2 if self.x_augment else 1
            for i in range(self.num_filter - 1):
                self.gnn_list.append(GINConv(torch.nn.Sequential(
                    torch.nn.Linear(self.filters[i] * isAugment, self.filters[i + 1]),
                    torch.nn.ReLU(),
                    torch.nn.Linear(self.filters[i + 1], self.filters[i + 1]),
                    torch.nn.BatchNorm1d(self.filters[i + 1]),
                ), eps=True))
        else:
            raise NotImplementedError("Unknown GNN-Operator.")
        
        if self.config['deepsets']:   #保证permutation-invariant
            for i in range(self.num_filter):
                if self.config['inner_mlp']:
                    if self.config.get('inner_mlp_layers', 1) == 1:
                        self.mlp_list_inner.append(
                            MLPLayers(self.filters[i], self.filters[i], None, num_layers=1, use_bn=False))
                    else:
                        self.mlp_list_inner.append(MLPLayers(self.filters[i], self.filters[i], self.filters[i],
                                                             num_layers=self.config['inner_mlp_layers'], use_bn=False))
                if self.config.get('outer_mlp_layers', 1) == 1:
                    self.mlp_list_outer.append(
                        MLPLayers(self.filters[i], self.filters[i], None, num_layers=1, use_bn=False))
                else:
                    self.mlp_list_outer.append(MLPLayers(self.filters[i], self.filters[i], self.filters[i],
                                                         num_layers=self.config['outer_mlp_layers'], use_bn=False))
                self.act_inner = getattr(F, self.config.get('deepsets_inner_act', 'relu'))
                self.act_outer = getattr(F, self.config.get('deepsets_outer_act', 'relu'))
                if self.config['use_sim'] and self.config['NTN_layers'] != 1:
                    self.NTN_list.append(TensorNetworkModule(self.config, self.filters[i]))
                if self.config['fuse_type'] == 'diff':
                    self.diff_mlp_list.append(MLPLayers(self.filters[i], self.filters[i], self.filters[i],
                                                        num_layers=2, use_bn=False))
            
            if self.config['use_sim'] and self.config['NTN_layers'] == 1:
                self.NTN = TensorNetworkModule(self.config, self.filters[self.num_filter - 1])

            if self.config['fuse_type'] == 'cat':
                self.channel_dim = sum(self.filters)
                self.reduction = self.config['reduction']
                self.conv_stack = nn.Sequential(
                    nn.Linear(self.channel_dim, self.channel_dim // self.reduction),
                    nn.ReLU(),
                    nn.Dropout(p=self.config['dropout']),
                    nn.Linear(self.channel_dim // self.reduction, (self.channel_dim // self.reduction)),
                    nn.Dropout(p=self.config['dropout']),
                    nn.Tanh(),
                )
            elif self.config['fuse_type'] == 'diff':
                self.channel_dim = sum(self.filters) * 2
                self.reduction = self.config['reduction']
                self.diff_stack = nn.Sequential(
                    nn.Linear(self.channel_dim, self.channel_dim // self.reduction),
                    nn.ReLU(),
                    nn.Dropout(p=self.config['dropout']),
                    nn.Linear(self.channel_dim // self.reduction, (self.channel_dim // self.reduction)),
                    nn.Dropout(p=self.config['dropout']),
                    nn.Tanh(),
                )
            else:
                raise RuntimeError(
                    'unsupported fuse type')


        if self.x_augment:
            '''
                AIDS700nef
                core_filter	gin_filter   
                [10 64]		[93 64]	
                [64 64]		[64*2 32]
                [64 32]		[32*2 16]		
                [32 16]	
            '''
            self.core_mlp_list.append(MLPLayers(self.max_node, self.filters[0], None, num_layers=1, use_bn=False))
            self.core_mlp_list.append(MLPLayers(self.filters[0], self.filters[0], None, num_layers=1, use_bn=False))
            for i in range(self.num_filter-1):
                if self.config.get('core_mlp_layers', 1) == 1:
                    self.core_mlp_list.append(
                        MLPLayers(self.filters[i], self.filters[i+1], None, num_layers=1, use_bn=False))
                else:
                    self.core_mlp_list.append(MLPLayers(self.filters[i], self.filters[i+1], self.filters[i+1],
                                                            num_layers=self.config['core_mlp_layers'], use_bn=False))


        # usin Gumbel-sinhorn permutation
        if self.use_gs:
            self.GSModel = GSModel(self.config, sum(self.filters))

    def setup_score_layer(self):
        if self.config['deepsets']:
            if self.config['fuse_type'] == 'cat':
                self.score_layer = nn.Sequential(nn.Linear((self.channel_dim // self.reduction), 16),
                                                 nn.ReLU(),
                                                 nn.Linear(16, 1))
            elif self.config['fuse_type'] == 'stack':
                self.score_layer = nn.Linear(self.filters[0], 1)
            elif self.config['fuse_type'] == 'diff':
                self.score_layer = nn.Sequential(nn.Linear((self.channel_dim // self.reduction), 16),nn.ReLU(),nn.Linear(16, 1))
            
            if self.config['use_sim']:
                if self.config['NTN_layers'] != 1:
                    self.score_sim_layer = nn.Sequential(
                        nn.Linear(self.config['tensor_neurons'] * self.num_filter, self.config['tensor_neurons']),
                        nn.ReLU(),
                        nn.Linear(self.config['tensor_neurons'], 1))
                else:
                    self.score_sim_layer = nn.Sequential(
                        nn.Linear(self.config['tensor_neurons'], self.config['tensor_neurons']),
                        nn.ReLU(),
                        nn.Linear(self.config['tensor_neurons'], 1))

        if self.config.get('output_comb', False):
            self.alpha = nn.Parameter(torch.Tensor(1))
            self.beta  = nn.Parameter(torch.Tensor(1))
            self.delta = nn.Parameter(torch.Tensor(1))

    def scale_init(self):
        nn.init.zeros_(self.alpha)
        nn.init.zeros_(self.beta)

    def convolutional_pass_level(self, enc, edge_index, x):
        feat = enc(x, edge_index)
        feat = F.relu(feat)
        feat = F.dropout(feat, p=self.config['dropout'], training=self.training)
        return feat

    def deepsets_outer(self, batch, feat, filter_idx, size=None):
        """
            mlp φ 采用gloabl_sum_pool输出
        """
        size = (batch[-1].item() + 1 if size is None else size)  # 一个batch中的图数

        pool = global_add_pool(feat, batch, size=size) if self.config['pooling'] == 'add' else global_mean_pool(feat,batch,size=size)
        return self.act_outer(self.mlp_list_outer[filter_idx](pool))

    def collect_embeddings(self, all_graphs):
        node_embs_dict = dict()
        for g in all_graphs:
            feat = g.x.cuda()
            edge_index = g.edge_index.cuda()
            for i, gnn in enumerate(self.gnn_list):
                if i not in node_embs_dict.keys():
                    node_embs_dict[i] = dict()
                feat = gnn(feat, edge_index)
                feat = F.relu(feat)
                node_embs_dict[i][int(g['i'])] = feat
        return node_embs_dict

    def collect_graph_embeddings(self, all_graphs):
        node_embs_dict = self.collect_embeddings(all_graphs)
        graph_embs_dicts = dict()
        for i in node_embs_dict.keys():
            if i not in graph_embs_dicts.keys():
                graph_embs_dicts[i] = dict()
            for k, v in node_embs_dict[i].items():
                deepsets_inner = self.act_inner(self.mlp_list_inner[i](v))
                g_emb = torch.sum(deepsets_inner, dim=0)
                graph_embs_dicts[i][k] = g_emb

        return graph_embs_dicts

    def forward(self, data):
        edge_index_1 = data['g1'].edge_index.cuda()
        edge_index_2 = data['g2'].edge_index.cuda()
        features_1 = data["g1"].x.cuda()
        features_2 = data["g2"].x.cuda()
        batch_1 = (
            data["g1"].batch.cuda()
            if hasattr(data["g1"], "batch")
            else torch.tensor((), dtype=torch.long).new_zeros(data["g1"].num_nodes).cuda()
        )
        batch_2 = (
            data["g2"].batch.cuda()
            if hasattr(data["g2"], "batch")
            else torch.tensor((), dtype=torch.long).new_zeros(data["g2"].num_nodes).cuda()
        )

        conv_source_1 = torch.clone(features_1)
        conv_source_2 = torch.clone(features_2)
        if self.x_augment:
            core_feature_1, conv_source_1 = torch.split(conv_source_1, [self.max_node, len(conv_source_1[0]) - self.max_node], dim=1)
            core_feature_2, conv_source_2 = torch.split(conv_source_2, [self.max_node, len(conv_source_2[0]) - self.max_node], dim=1)

        for i in range(self.num_filter):
            if self.x_augment:
                core_feature_1 = F.relu(self.core_mlp_list[i](core_feature_1))
                core_feature_2 = F.relu(self.core_mlp_list[i](core_feature_2))
                conv_source_1  = torch.cat((core_feature_1, conv_source_1), dim=1)
                conv_source_2  = torch.cat((core_feature_2, conv_source_2), dim=1)
            conv_source_1  = self.convolutional_pass_level(self.gnn_list[i], edge_index_1, conv_source_1)
            conv_source_2  = self.convolutional_pass_level(self.gnn_list[i], edge_index_2, conv_source_2)

            if self.config['deepsets']:
                if self.config.get('inner_mlp', True):
                    deepsets_inner_1 = self.act_inner(self.mlp_list_inner[i](conv_source_1))
                    deepsets_inner_2 = self.act_inner(self.mlp_list_inner[i](conv_source_2))
                else:
                    deepsets_inner_1 = self.act_inner(conv_source_1)
                    deepsets_inner_2 = self.act_inner(conv_source_2)

                deepsets_outer_1 = self.deepsets_outer(batch_1, deepsets_inner_1, i)
                deepsets_outer_2 = self.deepsets_outer(batch_2, deepsets_inner_2, i)

                if self.config['fuse_type'] == 'cat':
                    diff_rep = torch.exp(-torch.pow(deepsets_outer_1 - deepsets_outer_2, 2)) if i == 0 else torch.cat(
                        (diff_rep, torch.exp(-torch.pow(deepsets_outer_1 - deepsets_outer_2, 2))), dim=1)
                elif self.config['fuse_type'] == 'stack':
                    diff_rep = torch.abs(deepsets_outer_1 - deepsets_outer_2).unsqueeze(1) if i == 0 else torch.cat(
                        (diff_rep, torch.abs(deepsets_outer_1 - deepsets_outer_2).unsqueeze(1)), dim=1)
                elif self.config['fuse_type'] == 'diff':
                    cat_zij  = torch.abs(deepsets_outer_1 - deepsets_outer_2)
                    diff_atte= torch.softmax(self.diff_mlp_list[i](cat_zij) * (1/5), 1)
                    diff_rep_1 = diff_atte * deepsets_outer_1 if i == 0 else torch.cat((diff_rep_1, diff_atte * deepsets_outer_1), dim=1)
                    diff_rep_2 = diff_atte * deepsets_outer_2 if i == 0 else torch.cat((diff_rep_2, diff_atte * deepsets_outer_2), dim=1)

                if self.config['use_sim'] and self.config['NTN_layers'] != 1:
                    sim_rep = self.NTN_list[i](deepsets_outer_1, deepsets_outer_2) if i == 0 else torch.cat(
                        (sim_rep, self.NTN_list[i](deepsets_outer_1, deepsets_outer_2)), dim=1)
        if self.config['fuse_type'] == 'diff':
            score_rep = self.diff_stack(torch.cat((diff_rep_1, diff_rep_2), dim=1)).squeeze()
        else:
            score_rep = self.conv_stack(diff_rep).squeeze()

        if self.config['use_sim'] and self.config['NTN_layers'] == 1:
            sim_rep = self.NTN(deepsets_outer_1, deepsets_outer_2)

        if self.config['use_sim']:
            sim_score = torch.sigmoid(self.score_sim_layer(sim_rep).squeeze())      #s_{NTN} in paper

        score = torch.sigmoid(self.score_layer(score_rep)).view(-1)                 #s_{p} in paper


        if self.config.get('use_sim', False):
            if self.config.get('output_comb', False):
                comb_score = self.alpha * score + self.beta * sim_score
                return comb_score
            else:
                return (score + sim_score) / 2
        else:
            return score

if __name__ == "__main__":
    pass