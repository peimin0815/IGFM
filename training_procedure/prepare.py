import torch
import torch.nn as nn
from importlib import import_module
import os.path as osp
from DataHelper.DatasetLocal import DatasetLocal
from model.CGD import CGD
from model.GSModel import GSModel

def prepare_train(self, model):
    config = self.config
    optimizer = getattr(torch.optim, config['optimizer'])(params=model.parameters(),lr=config['lr'],weight_decay=config.get('weight_decay', 0))
    if config.get('lr_scheduler', False):
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, config['step_size'], gamma=config['gamma'])
    loss_func = nn.MSELoss(reduction='sum')
    return optimizer, loss_func

def prepare_model(self, dataset: DatasetLocal):
    """
        return model
    """
    config = self.config
    model_name = config['model_name']
    if model_name in ['IGFM']:
        model = CGD(config, dataset.input_dim).cuda()
    else:
        assert("No matching model was found...")
    return model

def init(self, dataset):
    # model = GSC(config, dataset.input_dim).cuda()
    model = self.prepare_model(dataset)
    optimizer, loss_func = self.prepare_train(model)

    return model, optimizer, loss_func

def init_gs(self, dataset):
    config = self.config    
    gs_model = GSModel(config, dataset.input_dim).cuda()
    gs_optimizer, gs_loss_func = self.prepare_train(gs_model)

    return gs_model, gs_optimizer, gs_loss_func
