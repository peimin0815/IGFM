import torch
from tqdm import tqdm
from utils.utils import *
import torch.nn.functional as F
from torch_geometric.data import Batch
from DataHelper.DatasetLocal import DatasetLocal

def evaluate(config, testing_graphs, training_graphs, model, dataset: DatasetLocal, validation = False):
    model.eval()
    num_test_pairs = len(testing_graphs) * len(training_graphs)     ## AIDS: 140 * 420 = 58,800
    num_pair_per_node = len(training_graphs)                        ## AIDS: 420

    scores = np.empty((len(testing_graphs), num_pair_per_node))     ## (140, 420)
    ground_truth = np.empty((len(testing_graphs), num_pair_per_node))
    ground_truth_ged = np.empty((len(testing_graphs), num_pair_per_node))
    prediction_mat = np.empty((len(testing_graphs), num_pair_per_node))

    t = tqdm(total=num_test_pairs)

    for i, g in enumerate(testing_graphs):
        source_batch = Batch.from_data_list([g] * num_pair_per_node)
        if validation:
            training_graphs = training_graphs.shuffle()[: num_pair_per_node]
        target_batch = Batch.from_data_list(training_graphs)
        data = dataset.transform_batch((source_batch, target_batch), config)
        target = data["target"]

        ground_truth[i] = target
        target_ged = data["target_ged"]
        ground_truth_ged[i] = target_ged
        prediction = model(data)
        prediction_mat[i] = prediction.cpu().detach().numpy()
        scores[i] = (F.mse_loss(prediction.cpu().detach(), target, reduction="none").numpy())
        t.update(num_pair_per_node)

    model_mse_error = np.mean(scores).item()
    return model_mse_error

def pretrain_gs(T, config, dataset, gs_model, gs_loss_func, gs_optim):
    pbar                             = tqdm(range(config['gs_epochs']), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
    best_mse_metric                  = 100000.0

    for epoch in pbar:
        batches                      = dataset.create_batches(config)
        main_index                   = 0
        per_loss_sum                 = 0
        for batch_pair in batches:
            data                 = dataset.transform_batch(batch_pair, config)
            target               = data["target"].cuda()
            loss                 = T.train_gs(data, gs_model, gs_loss_func, gs_optim, target)
            loss.backward()
            gs_optim.step()
            main_index               = main_index + batch_pair[0].num_graphs  
            per_loss_sum             = per_loss_sum + loss.item() 

        if config['use_val']:
            if epoch >= config['gs_epochs'] * 0.9 and epoch % 10 == 0:
                val_mse = evaluate(config, dataset.val_graphs, dataset.training_graphs, gs_model, dataset, validation=True)
                print("val_mse: {}".format(val_mse))
                if best_mse_metric                >= val_mse:
                    best_mse_metric               = val_mse
                    PATH_PRETRAIN = os.path.join(os.getcwd(),'model_saved/pretrained_model')
                    PATH_MODEL = os.path.join(PATH_PRETRAIN, config['dataset_name'] + '_' + str(config['gs_epochs']) + '_pretrained_model.pth')
                    torch.save(gs_model.state_dict(), PATH_MODEL)

        per_loss = per_loss_sum / main_index
        if epoch != config['epochs']-1:
            postfix_str = "<Epoch %d> [Train Loss] %.5f"% (epoch, per_loss) 
        pbar.set_postfix_str(postfix_str)
    return gs_model    