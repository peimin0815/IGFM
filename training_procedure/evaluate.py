import torch
from sklearn.metrics import f1_score
import torch.nn as nn
from utils.utils import Evaluation
import numpy as np
from torch_geometric.data import Batch
from DataHelper.DatasetLocal import DatasetLocal
from utils.utils import *
import torch.nn.functional as F
from scipy.stats import spearmanr, kendalltau
from tqdm import tqdm, trange


@torch.no_grad()
def get_eval_result(self, labels, pred_l, loss):
    if self.config['multilabel']:
        micro, macro = Evaluation(pred_l, labels)
    else:
        micro = f1_score(labels.cpu(), pred_l.cpu(), average="micro")
        macro = 0

    return {
        "micro": round(micro * 100, 2),  # to percentage
        "macro": round(macro * 100, 2)
    }


@torch.no_grad()
def evaluate(self, testing_graphs, training_graphs, model, dataset: DatasetLocal, validation=False, gs_model=None):
    model.eval()
    num_test_pairs    = len(testing_graphs) * len(training_graphs)     ## AIDS: 140 * 560 = 58,800
    num_pair_per_node = len(training_graphs)                        ## AIDS: 560

    scores            = np.empty((len(testing_graphs), num_pair_per_node))     ## (140, 560)
    ground_truth      = np.empty((len(testing_graphs), num_pair_per_node))
    ground_truth_ged  = np.empty((len(testing_graphs), num_pair_per_node))
    prediction_mat    = np.empty((len(testing_graphs), num_pair_per_node))

    rho_list        = []
    tau_list        = []
    prec_at_10_list = []
    prec_at_20_list = []

    t        = tqdm(total=num_test_pairs)
    iterator = iter(testing_graphs)
    for i, g in enumerate(iterator):
        source_batch = Batch.from_data_list([g] * num_pair_per_node)
        if validation:
            if not self.config.get('use_all_val', True):
                training_graphs = training_graphs.shuffle()[: num_pair_per_node]

        target_batch = Batch.from_data_list(training_graphs)
        data         = dataset.transform_batch((source_batch, target_batch), self.config)
        target       = data["target"]
        # target = data["norm_ged"]

        ground_truth[i]     = target
        target_ged          = data["target_ged"]
        ground_truth_ged[i] = target_ged

        if self.config["model_name"] == "TaGSim":
            pred_list       = []
            graph_pair_nums = len(data['g1']) 
            for index in range(graph_pair_nums):
                new_data       = dict()
                new_data["g1"] = data['g1'][index]   
                new_data["g2"] = data['g2'][index]
                pred_per       = model(new_data)
                pred_list.append(pred_per)
            prediction = torch.stack(pred_list, dim=0)
        elif self.config["model_name"] == "Eric":
            prediction, _ = model(data)
        elif self.config["model_name"] == "GS_GSC":
            prediction          = model(data)
            # if self.config['use_gs']:
            #     n_prediction    = gs_model(data)
            #     prediction      = prediction * (1 - self.config['gs_weight'])  + n_prediction * self.config['gs_weight']
        else:
            prediction          = model(data)

        prediction_mat[i]   = prediction.cpu().detach().numpy()
        scores[i]           = (F.mse_loss(prediction.cpu().detach(), target, reduction="none").numpy())

        rho_list.append(
            calculate_ranking_correlation(
                spearmanr, prediction_mat[i], ground_truth[i]
            )
        )
        tau_list.append(
            calculate_ranking_correlation(
                kendalltau, prediction_mat[i], ground_truth[i]
            )
        )
        prec_at_10_list.append(
            calculate_prec_at_k(
                10, prediction_mat[i], ground_truth[i], ground_truth_ged[i]
            )
        )
        prec_at_20_list.append(
            calculate_prec_at_k(
                20, prediction_mat[i], ground_truth[i], ground_truth_ged[i]
            )
        )
        t.update(num_pair_per_node)

    rho             = np.mean(rho_list).item()
    tau             = np.mean(tau_list).item()
    prec_at_10      = np.mean(prec_at_10_list).item()
    prec_at_20      = np.mean(prec_at_20_list).item()
    model_mse_error = np.mean(scores).item()

    return model_mse_error, rho, tau, prec_at_10, prec_at_20