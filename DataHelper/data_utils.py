import torch
from torch_geometric.data import Data
from torch_geometric.utils import erdos_renyi_graph, to_undirected, to_networkx
import random

def to_directed(edge_index):
    '''
    无向图变为有向图
    '''
    row, col = edge_index
    mask = row < col
    row, col = row[mask], col[mask]
    return torch.stack([row, col], dim=0)

def gen_pair(g, kl=None, ku=2):
    '''
    g: 输入一个图
    kl: edit下限为0
    ku: edit上限为2
    随机将g中0~2条边删除，并添加0~2条边， 返回ged
    '''
    if kl is None:
        kl = ku

    directed_edge_index = to_directed(g.edge_index)

    n = g.num_nodes
    num_edges = directed_edge_index.size()[1]
    to_remove = random.randint(kl, ku)  # 要移除的节点数 0,1,2

    edge_index_n = directed_edge_index[:, torch.randperm(num_edges)[to_remove:]]   # 随机移除to_remove条边后的edge_index
    if edge_index_n.size(1) != 0:   # 转为无向图
        edge_index_n = to_undirected(edge_index_n)

    row, col = g.edge_index
    adj = torch.ones((n, n), dtype=torch.uint8)  # 提取所有不在图中的边
    adj[row, col] = 0   # 存在的边设为0， 不在的边设为1
    non_edge_index = adj.nonzero().t()  # 

    directed_non_edge_index = to_directed(non_edge_index)
    num_edges = directed_non_edge_index.size()[1]

    to_add = random.randint(kl, ku)

    edge_index_p = directed_non_edge_index[:, torch.randperm(num_edges)[:to_add]]  # 随机添加0~2条边
    if edge_index_p.size(1):
        edge_index_p = to_undirected(edge_index_p)
    edge_index_p = torch.cat((edge_index_n, edge_index_p), 1)

    if hasattr(g, "i"):
        g2 = Data(x=g.x, edge_index=edge_index_p, i=g.i)
    else:
        g2 = Data(x=g.x, edge_index=edge_index_p)

    g2.num_nodes = g.num_nodes
    return g2, to_remove + to_add


def gen_pairs(graphs, kl=None, ku=2):
    '''
    graphs: 训练集前500个图
    kl: ged下限   0
    ku: ged上限   2
    '''
    
    gen_graphs_1 = []  # 存放原图
    gen_graphs_2 = []  # 存放edit后的图

    count = len(graphs)
    mat = torch.full((count, count), float("inf"))  # 500*500 对角线元素存放500对 图的ged
    norm_mat = torch.full((count, count), float("inf"))

    for i, g in enumerate(graphs):
        g = g.clone()
        g.i = torch.tensor([i])  # 图的id
        g2, ged = gen_pair(g, kl, ku)
        gen_graphs_1.append(g)
        gen_graphs_2.append(g2)
        mat[i, i] = ged
        norm_mat[i, i] = ged / (0.5 * (g.num_nodes + g2.num_nodes))

    return gen_graphs_1, gen_graphs_2, mat, norm_mat


    