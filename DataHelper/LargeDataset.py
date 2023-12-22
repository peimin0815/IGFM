import glob
import os
import os.path as osp
import pickle
from typing import Callable, List, Optional

import torch

from torch_geometric.data import (
    Data,
    InMemoryDataset,
    download_url,
    extract_zip,
)
from torch.nn.functional import one_hot
from torch_geometric.utils import to_undirected
import matplotlib.pyplot as plt

class Large_GEDDataset(InMemoryDataset):


    url = 'https://drive.google.com/uc?export=download&id={}'

    datasets = {
        'NCI109': {
            'id': '1As8XOLhY2-KbYBbCCvnvP_Z5ALmKK3hv',
            'extract': extract_zip,
            'pickle': '1v1j6ViRorLDG_xQRLS9LCkIL79qCbM1n',
        },
    }

    # List of atoms contained in the NCI109 dataset:
    types = [str(num) for num in range(1,39)]

    def __init__(self, root: str, name: str, train: bool = True,
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 pre_filter: Optional[Callable] = None):
        self.name = name
        assert self.name in self.datasets.keys()
        super().__init__(root, transform, pre_transform, pre_filter)
        path = self.processed_paths[0] if train else self.processed_paths[1]
        self.data, self.slices = torch.load(path)
        path = osp.join(self.processed_dir, f'{self.name}_ged.pt')
        self.ged = torch.load(path)
        path = osp.join(self.processed_dir, f'{self.name}_norm_ged.pt')
        self.norm_ged = torch.load(path)

    @property
    def raw_file_names(self) -> List[str]:
        # Returns, e.g., ['LINUX/train', 'LINUX/test']
        return [osp.join(self.name, s) for s in ['train', 'test']]

    @property
    def processed_file_names(self) -> List[str]:
        # Returns, e.g., ['LINUX_training.pt', 'LINUX_test.pt']
        return [f'{self.name}_{s}.pt' for s in ['training', 'test']]

    def download(self):
        # Downloads the .tar/.zip file of the graphs and extracts them:
        name = self.datasets[self.name]['id']
        path = download_url(self.url.format(name), self.raw_dir)
        self.datasets[self.name]['extract'](path, self.raw_dir)
        os.unlink(path)

        # Downloads the pickle file containing pre-computed GEDs:
        name = self.datasets[self.name]['pickle']
        path = download_url(self.url.format(name), self.raw_dir)
        os.rename(path, osp.join(self.raw_dir, self.name, 'ged.pickle'))

    def process(self):
        import networkx as nx

        ids, Ns = [], []
        # Iterating over paths for raw and processed data (train + test):
        for r_path, p_path in zip(self.raw_paths, self.processed_paths):
            # Find the paths of all raw graphs:
            names = glob.glob(osp.join(r_path, '*.gexf'))
            # Get sorted graph IDs given filename: 123.gexf -> 123
            ids.append(sorted([int(i.split(os.sep)[-1][:-5]) for i in names]))
            
            data_list = []
            count = 0
            # Convert graphs in .gexf format to a NetworkX Graph:
            for i, idx in enumerate(ids[-1]):
                i = i if len(ids) == 1 else i + len(ids[0])
                # Reading the raw `*.gexf` graph:
                G = nx.read_gexf(osp.join(r_path, f'{idx}.gexf'))
                # Mapping of nodes in `G` to a contiguous number:
                mapping = {name: j for j, name in enumerate(G.nodes())}
                G = nx.relabel_nodes(G, mapping)
                Ns.append(G.number_of_nodes())

                # if idx == 139 or idx == 127 or idx == 133:
                #     nx.draw(G, with_labels=True, node_color='lightblue', font_weight='bold')
                #     plt.savefig("graph_id_{}_i_{}.png".format(idx, i))
                #     plt.close()
                
                edge_index = torch.tensor(list(G.edges)).t().contiguous()
                if edge_index.numel() == 0:
                    edge_index = torch.empty((2, 0), dtype=torch.long)
                edge_index = to_undirected(edge_index, num_nodes=Ns[-1])

                data = Data(edge_index=edge_index, i=i)
                data.num_nodes = Ns[-1]

                # Create a one-hot encoded feature matrix denoting the atom
                # type (for the `NCI109` dataset):
                if self.name == 'NCI109':
                    x = torch.zeros(data.num_nodes, dtype=torch.long)
                    for node, info in G.nodes(data=True):
                        x[int(node)] = self.types.index(info['type'])
                    data.x = one_hot(x, num_classes=len(self.types))

                if self.pre_filter is not None and not self.pre_filter(data):
                    continue

                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                data_list.append(data)

            torch.save(self.collate(data_list), p_path)

        assoc = {idx: i for i, idx in enumerate(ids[0])}
        assoc.update({idx: i + len(ids[0]) for i, idx in enumerate(ids[1])})

        # Extracting ground-truth GEDs from the GED pickle file
        path = osp.join(self.raw_dir, self.name, 'ged.pickle')
        # Initialize GEDs as float('inf'):
        mat = torch.full((len(assoc), len(assoc)), float('inf'))

        with open(path, 'rb') as f:
            obj = pickle.load(f)
            xs, ys, gs = [], [], []
            
            # with open(out_index, 'w') as file:
            for (x, y), g in obj.items():
                # if assoc[x] < len(assoc) and assoc[y] < len(assoc):
                xs += [assoc[x]]
                ys += [assoc[y]]
                gs += [g]
                    # else:
                    #     file.write("x: {} y:{} g:{}\n".format(assoc[x], assoc[y], g))
            # The pickle file does not contain GEDs for test graph pairs, i.e.
            # GEDs for (test_graph, test_graph) pairs are still float('inf'):
            x, y = torch.tensor(xs), torch.tensor(ys)
            ged = torch.tensor(gs, dtype=torch.float)
            # mat[y, x] = ged
            # mat[x, y] = ged
            mat[x, y], mat[y, x] = ged, ged
            

        path = osp.join(self.processed_dir, f'{self.name}_ged.pt')
        torch.save(mat, path)

        # Calculate the normalized GEDs:
        N = torch.tensor(Ns, dtype=torch.float)
        norm_mat = mat / (0.5 * (N.view(-1, 1) + N.view(1, -1)))

        path = osp.join(self.processed_dir, f'{self.name}_norm_ged.pt')
        torch.save(norm_mat, path)

    def __repr__(self) -> str:
        return f'{self.name}({len(self)})'
