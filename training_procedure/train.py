import torch
import torch_geometric.transforms as T
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from utils.utils import fixed_mapping_loss

def train(self, graph_batch, model, loss_func, optimizer, target, config):
    model.train(True)
    
    optimizer.zero_grad()
    if config["model_name"] == "GEDGNN":
        gt_mapping = graph_batch
        prediction, mapping = model(graph_batch)
        loss = fixed_mapping_loss(mapping, gt_mapping) + config["weight"] * loss_func(target, prediction)
    elif config["model_name"] == "TaGSim":
        loss = torch.tensor([0]).float().cuda()
        graph_pair_nums = len(graph_batch['g1']) 
        for index in range(graph_pair_nums):
            target_p       = target[index]
            new_data       = dict()
            new_data["g1"] = graph_batch['g1'][index]   
            new_data["g2"] = graph_batch['g2'][index]
            prediction     = model(new_data)
            loss           = loss + loss_func(prediction, target_p)
    elif config['model_name'] == "Eric":
        use_ssl = config.get('use_ssl', False)
        prediction, loss_cl = model(graph_batch)
        loss = loss_func(prediction, target) if not use_ssl else loss_func(prediction, target) + loss_cl
    else:
        prediction = model(graph_batch)
        loss = loss_func(prediction, target)
    return model, loss