import torch
import numpy as np 
import networkx as nx  
from .data_utils import *
from .dataset_helper import dataset
import torch_geometric.transforms as T
from torch_geometric.utils import degree
from .LargeDataset import Large_GEDDataset
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import GEDDataset
from torch_geometric.data.collate import collate
from torch_geometric.transforms import OneHotDegree


class DatasetLocal(dataset):

    data = None

    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)
        self.core_array = []

    def load_core_data(self, config):
        print("load_core_data")
        # 先load teseting_graphs取max_node_size
        self.testing_graphs = self.load_test_data()
        self.trainval_graphs = self.load_train_data()

        self.get_max_node_size()
        updated_graphs = self.get_k_core(self.trainval_graphs)
        self.trainval_graphs.data, _, = self.collate(updated_graphs)
        self.trainval_graphs._data_list = updated_graphs
        if config['use_val']:
            val_ratio = config['val_ratio']
            num_trainval_gs = len(self.trainval_graphs)
            self.val_graphs = self.trainval_graphs[-int(num_trainval_gs * val_ratio):]
            self.training_graphs = self.trainval_graphs[0: -int(num_trainval_gs * val_ratio)]

        # 后x augmentation
        updated_graphs = self.get_k_core(self.testing_graphs)
        self.testing_graphs.data, _, = self.collate(updated_graphs)
        self.testing_graphs._data_list = updated_graphs

        self.trainval_nged_matrix = self.trainval_graphs.norm_ged
        self.trainval_ged_matrix = self.trainval_graphs.ged
        self.real_trainval_data_size = self.trainval_nged_matrix.size(0)  # 700
        self.num_graphs = len(self.trainval_graphs) + len(self.testing_graphs)
        self.num_train_graphs = len(self.training_graphs)
        self.num_val_graphs = len(self.val_graphs)
        self.num_test_graphs = len(self.testing_graphs)

        # 对没有label的数据集按照degree生成onehot向量，并拼接x
        dummy_trainval_graphs = self.load_train_data()
        if dummy_trainval_graphs[0].x is None:
            max_degree = 0
            for g in (self.trainval_graphs + self.testing_graphs):
                if g.edge_index.size(1) > 0:
                    max_degree = max(max_degree, int(degree(g.edge_index[0]).max().item()))
            
            one_hot_graphs = self.one_hot_init(max_degree, self.trainval_graphs)
            self.trainval_graphs.data, _, = self.collate(one_hot_graphs)
            self.trainval_graphs._data_list = one_hot_graphs
            if config['use_val']:
                val_ratio = config['val_ratio']
                num_trainval_gs = len(self.trainval_graphs)
                self.val_graphs = self.trainval_graphs[-int(num_trainval_gs * val_ratio):]  # 140
                self.training_graphs = self.trainval_graphs[0: -int(num_trainval_gs * val_ratio)]  # 420

            one_hot_graphs = self.one_hot_init(max_degree, self.testing_graphs)
            self.testing_graphs.data, _, = self.collate(one_hot_graphs)
            self.testing_graphs._data_list = one_hot_graphs

        self.number_of_labels = self.trainval_graphs.num_features
        # self.number_of_labels = self.trainval_graphs._data.num_node_features
        self.input_dim = self.number_of_labels  

    def load(self, config):
        self.trainval_graphs = self.load_train_data()
        if config['use_val']:
            val_ratio = config['val_ratio']
            num_trainval_gs = len(self.trainval_graphs)
            self.val_graphs = self.trainval_graphs[-int(num_trainval_gs * val_ratio):]         # 140
            self.training_graphs = self.trainval_graphs[0: -int(num_trainval_gs * val_ratio)]  # 420
        self.testing_graphs = self.load_test_data()

        self.trainval_nged_matrix = self.trainval_graphs.norm_ged
        self.trainval_ged_matrix = self.trainval_graphs.ged
        self.real_trainval_data_size = self.trainval_nged_matrix.size(0)                       # 700
        self.num_graphs = len(self.trainval_graphs) + len(self.testing_graphs)
        self.num_train_graphs = len(self.training_graphs)
        self.num_val_graphs = len(self.val_graphs)
        self.num_test_graphs = len(self.testing_graphs)

        if self.trainval_graphs[0].x is None:
            max_degree = 0
            for g in (self.trainval_graphs+ self.testing_graphs):
                if g.edge_index.size(1) > 0:
                    max_degree = max(
                        max_degree, int(degree(g.edge_index[0]).max().item())
                    )
            one_hot_degree = OneHotDegree(max_degree, cat=False)
            self.trainval_graphs.transform = one_hot_degree
            self.val_graphs.transform = one_hot_degree
            self.training_graphs.transform = one_hot_degree
            self.testing_graphs.transform = one_hot_degree

        self.number_of_labels = self.trainval_graphs.num_features
        self.input_dim = self.number_of_labels

    def load_train_data(self):
        if self.dataset_name == 'NCI109':
            return Large_GEDDataset('datasets/{}'.format(self.dataset_name),self.dataset_name,train=True)
        else:
            return GEDDataset('datasets/{}'.format(self.dataset_name),self.dataset_name,train=True)
        # self.custom_dataset = GEDDataset_Custom(ged_main_dir=self.dataset_source_folder_path, config=config)

    def load_test_data(self):
        if self.dataset_name == 'NCI109':
            return Large_GEDDataset('datasets/{}'.format(self.dataset_name),self.dataset_name,train=False)
        else:
            return GEDDataset('datasets/{}'.format(self.dataset_name),self.dataset_name,train=False)

    def create_batches(self, config):
        source_loader = DataLoader(
            self.training_graphs.shuffle(),
            batch_size=config['batch_size'], num_workers=config.get('num_works', 0)
        )

        target_loader = DataLoader(
            self.training_graphs.shuffle(),
            batch_size=config['batch_size'], num_workers=config.get('num_works', 0)
        )
        return list(zip(source_loader, target_loader))

    def transform_batch(self, batch, config):
        """
        Getting ged for graph pair and grouping with data into dictionary.
        :param data: Graph pair.
        :return new_data: Dictionary with data.
        """
        new_data = dict()

        new_data["g1"] = batch[0]  # DataBatch(edge_index=[2, 2254], i=[128], x=[1146, 29], num_nodes=1146, batch=[1146], ptr=[129])
        new_data["g2"] = batch[1]

        normalized_ged = self.trainval_nged_matrix[
            batch[0]["i"].reshape(-1).tolist(), batch[1]["i"].reshape(-1).tolist()
        ].tolist()
        # 对normalized_ged转换，e(-x)产生0-1之间的数, ged越大相似度越小，ged越小相似度越大
        new_data["target"] = (
            torch.from_numpy(np.exp([(-el * config.get('scale', 1)) for el in normalized_ged])).view(-1).float()
        )
        new_data['norm_ged'] = (
            torch.from_numpy(np.exp([(el) for el in normalized_ged])).view(-1).float()
        )
        ged = self.trainval_ged_matrix[
            batch[0]["i"].reshape(-1).tolist(), batch[1]["i"].reshape(-1).tolist()
        ].tolist()

        new_data["target_ged"] = (
            torch.from_numpy(np.array([(el) for el in ged])).view(-1).float()  # nged
        )
        
        return new_data

    def load_val_train_pairs(self):
        val_len = len(self.val_graphs)
        train_len = len(self.training_graphs)

        val_pairs_triples = []
        for m in range(val_len):
            g1 = self.val_graphs[m]
            for n in range(train_len):
                g2 = self.training_graphs[n]
                nged = self.trainval_nged_matrix[g1["i"], g2["i"]]
                ged = self.trainval_ged_matrix[g1["i"], g2["i"]]
                val_pairs_triples.append([g1, g2, nged, ged])
        return val_pairs_triples

    def generate_all_val_gs(self, config):

        # print(self.val_graphs[0])Data(edge_index=[2, 20], i=[1], x=[10, 29], num_nodes=10)
        source_gs = []
        target_gs = []

        for i in range(len(self.validation_triples)):
            g1, g2, nged, ged = self.validation_triples[i]
            source_gs.append(g1)
            target_gs.append(g2)

        source_val_loader = DataLoader(source_gs, batch_size=config['val_batch_size'])

        target_val_loader = DataLoader(target_gs, batch_size=config['val_batch_size'])

        return list(zip(source_val_loader, target_val_loader))
    
    def feat_augment(self):
        
        updated_graphs = self.get_k_core(self.trainval_graphs)
        self.trainval_graphs.data, _, _ = collate(
            updated_graphs[0].__class__,data_list=updated_graphs,increment=False,add_batch=False,
        )
        self.trainval_graphs._data_list = updated_graphs

        updated_graphs = self.get_k_core(self.training_graphs)
        self.training_graphs.data, _, _ = collate(
            updated_graphs[0].__class__,data_list=updated_graphs,increment=False,add_batch=False,
        )
        self.training_graphs._data_list = updated_graphs

        updated_graphs = self.get_k_core(self.val_graphs)
        self.val_graphs.data, _, _ = collate(updated_graphs[0].__class__,data_list=updated_graphs,increment=False,add_batch=False,)
        self.val_graphs._data_list = updated_graphs
        
        updated_graphs = self.get_k_core(self.testing_graphs)
        self.testing_graphs.data, _, _ = collate(
            updated_graphs[0].__class__,data_list=updated_graphs,increment=False,add_batch=False,
        )
        self.testing_graphs._data_list = updated_graphs
        self.input_dim = len(self.training_graphs.data.x[0])

    def get_k_core(self, graphs):
        '''
            argument: a list of need augment graphs
            return  : a list of augmented graphs, feature = origin_f + k-core_f 
        '''
        G = nx.Graph()  
        updated_graphs = []

        self.core_array = np.zeros((self.max_node_size, 1), dtype=int)  

        for i in range(len(graphs)):    
            graph = graphs[i]
            edge_index = graph.edge_index.t().contiguous().tolist()
            G.add_edges_from(tuple(edge_index))

            # 计算 k-core 并返回结果矩阵  
            kcore_matrix = self.cal_k_core(G, self.max_node_size)  
            kcore_matrix = torch.from_numpy(kcore_matrix).to(torch.float32)

            newGraph = Data(edge_index=graph.edge_index, i=graph.i)
            if graph.x is None:
                newGraph.x = kcore_matrix
            else:
                newGraph.x = torch.cat((kcore_matrix, graph.x), dim=1)
            newGraph.num_nodes = len(kcore_matrix)
            # 将计算k-core feature后的图对象添加到新列表中
            updated_graphs.append(newGraph)  
            G.clear()
        # 计算k-core后的graph data对象 Aids feature dim 29 => 39
        return updated_graphs

    def cal_k_core(self, G, max_node_size):  
        # 计算最大度数  
        # max_degree = max(dict(nx.degree(G)).values())  
    
        # 初始化结果矩阵  
        result = np.zeros((len(G), max_node_size), dtype=int)  
    
        # 计算 k-core  
        for k in range(0, len(G.nodes()) + 1): 
            if len(G.nodes()) < 1:
                break 
            # 移除度数小于 k 的节点  
            while True:
                remove_nodes = list(filter(lambda node: G.degree(node) < k, G.nodes()))
                if len(remove_nodes):
                    # 将不满足当前 k-core 的节点删除，记录k-core值
                    G.remove_nodes_from(remove_nodes)
                    for node in remove_nodes:
                        result[node, k-1] = 1
                        self.core_array[k-1] = self.core_array[k-1] + 1 
                else:
                    break 
        #返回一个node*max_node_size矩阵，1表示在col+1-core中
        return result  
    
    def get_max_node_size(self):
        #计算dataset中最大graph节点个数
        self.max_node_size = -1
        total_graph = self.trainval_graphs + self.testing_graphs
        for graph in total_graph:
            self.max_node_size = max(self.max_node_size, graph.num_nodes)
        return self.max_node_size

    def one_hot_init(self, max_degree, initial_graphs):
        one_hot_degree = OneHotDegree(max_degree, cat=True)
        one_hot_graps  = []
        for i in range(len(initial_graphs)):    
            graph = one_hot_degree(initial_graphs[i])
            one_hot_graps.append(graph)
        return one_hot_graps