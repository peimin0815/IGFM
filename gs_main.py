
import sys
import torch
import random
import logging
import os.path as osp
from tqdm import tqdm
from utils.utils import *
from utils.logger import Logger
from argparse import ArgumentParser
from training_procedure import Trainer
from utils.random_seeder import set_random_seed
from DataHelper.DatasetLocal import DatasetLocal

import numpy as np
from torch_geometric.data import Batch
from DataHelper.DatasetLocal import DatasetLocal
from utils.utils import *
import torch.nn.functional as F


from torch.utils.tensorboard import SummaryWriter

def main(args, config, logger, dataset: DatasetLocal):
	T                                = Trainer(config=config, args= args)

	gs_model, gs_optim, gs_loss_func = T.init_gs(dataset) # init model of gumbel-sinkhorn
	pbar                             = tqdm(range(config['epochs']), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
	writer                       = SummaryWriter(RESULT_PATH)

	loss_list                        = []
	for epoch in pbar:
		batches                  = dataset.create_batches(config)   # 128对 graph-pair
		main_index                   = 0
		per_loss_sum                 = 0 

		for batch_pair in batches:
			data                     = dataset.transform_batch(batch_pair, config)
			target                   = data["target"].cuda()
			loss2                    = T.train_gs(data, gs_model, gs_loss_func, gs_optim, target)
			loss2.backward()
			gs_optim.step()

			per_loss_sum             = per_loss_sum + loss2.item()
			main_index               = main_index + batch_pair[0].num_graphs    

		per_loss = per_loss_sum / main_index
		writer.add_scalar("per_loss", per_loss, epoch)

		loss_list.append(per_loss)
		if epoch != config['epochs']-1:
			postfix_str = "<Epoch %d> [Train Loss] %.5f"% (epoch ,      per_loss) 
		
		pbar.set_postfix_str(postfix_str)

	logger.info("start testing: {}".format(postfix_str))

	test_mse = evaluate(config, dataset.testing_graphs, dataset.trainval_graphs, gs_model, dataset)
	writer.close()


@torch.no_grad()
def evaluate(config, testing_graphs, training_graphs, model, dataset: DatasetLocal):
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
	print("model_mse_error: {}".format(model_mse_error))
	return model_mse_error

def print_evaluation(model_error,rho,tau,prec_at_10,prec_at_20):
    """
    Printing the error rates.
    """
    logger.info("\nmse(10^-3): "   + str(round(model_error * 1000, 5))         + ".")
    logger.info("Spearman's rho: " + str(round(rho, 5))                        + ".")
    logger.info("Kendall's tau: "  + str(round(tau, 5))                        + ".")
    logger.info("p@10: "           + str(round(prec_at_10, 5))                 + ".")
    logger.info("p@20: "           + str(round(prec_at_20, 5))                 + ".")

if __name__ == "__main__":
	parser = ArgumentParser()
	parser.add_argument('--dataset',           type = str,            default = 'AIDS700nef') 
	parser.add_argument('--hyper_dir',         type = str,            default = 'config/')
	parser.add_argument('--epochs',            type = int,            default = 1000)
	parser.add_argument('--start',             type = int,            default = 990, help="iter_val_start")
	parser.add_argument('--step',              type = int,            default = 10 , help="iter_val_every") 
	parser.add_argument('--gpu_id',            type = int,            default = 0)
	parser.add_argument('--gs_weight',         type = float,          default = 0.1)
	args = parser.parse_args()
	torch.cuda.set_device(args.gpu_id)


	# path to save log and model file
	RESULT_PATH  = os.path.join(os.path.join(os.getcwd(),'model_saved'), args.dataset, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
	if not os.path.exists(RESULT_PATH):
		os.makedirs(RESULT_PATH)

	# logger setup
	logger = logging.getLogger("GS Model")
	fh     = logging.FileHandler(filename=os.path.join(RESULT_PATH, "out.log"))
	fh.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s", datefmt="%m/%d %H:%M:%S"))
	logger.setLevel(logging.DEBUG)
	logger.addHandler(fh)

	config_path                  = osp.join(args.hyper_dir, 'GS.yml')
	config                       = get_config(config_path)  #获取训练参数
	config                       = config["GS_GSC"] 
	config['dataset_name']       = args.dataset
	config['epochs']             = args.epochs
	config['iter_val_start']     = args.start
	config['iter_val_every']     = args.step
	config['gs_weight']          = args.gs_weight

	custom                       = config.get('custom', False)
	dev_ress                     = []
	tes_ress                     = []
	tra_ress                     = []
	if config.get('seed',-1)     > 0:
		set_random_seed(config['seed'])
		logger.info("Seed set. %d" % (config['seed']))
	seeds                        = [random.randint(0,233333333) for _ in range(config['multirun'])]
	dataset                      = load_data(args, custom)
	if not custom:
		dataset.load(config)  # config dataset
	else:
		dataset.load_custom_data(config, args)
	print_config(config, logger)

	logger.info("total graphs = {}"                                         .format(dataset.num_graphs))
	logger.info("train_gs.len = {} and val_gs.len = {} and test_gs.len = {}".format(dataset.num_train_graphs, dataset.num_val_graphs, dataset.num_test_graphs))
	for run_id in range(config['multirun']):   # one mask
		logger.info("\t\t%d th Run" % run_id)

		main(args, config, logger, dataset)

