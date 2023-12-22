import torch.nn as nn

def train_gs(self, graph_batch, model, loss_func, optimizer, target):
    model.train(True)
    optimizer.zero_grad()

    prediction = model(graph_batch)
    loss = loss_func(prediction, target)
    return loss

  