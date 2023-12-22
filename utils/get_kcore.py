import torch
import networkx as nx  
import numpy as np  
import matplotlib.pyplot as plt  

def kcore(G):  
    # 计算最大度数  
    max_degree = max(dict(nx.degree(G)).values())  
  
    # 初始化结果矩阵  
    result = np.zeros((len(G), max_degree + 1), dtype=int)  
  
    # 计算 k-core  
    for k in range(1, max_degree + 1):  
        # 移除度数小于 k 的节点  
        while True:
            remove_nodes = list(filter(lambda node: G.degree(node) < k, G.nodes()))
            if len(remove_nodes):
                G.remove_nodes_from(remove_nodes)
            else:
                 break 
        # 将当前 k-core 的节点标记为 1
        for node in G.nodes():
            result[node, k-1] = 1
    return result  


def test():
    edge_index = torch.tensor([[0, 1, 2, 3, 4, 5],[1, 2, 3, 4, 5, 2]])
    edge_index = edge_index.t().contiguous().tolist()
    G = nx.Graph()  
    G.add_edges_from(tuple(edge_index))
    nx.draw(G, with_labels=True)
    plt.savefig("test.jpg")
    # 计算 k-core 并返回结果矩阵  
    kcore_matrix = kcore(G)  
    print(kcore_matrix)


if __name__ == "__main__":
    test()