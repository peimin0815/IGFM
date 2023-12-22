import torch
import logging
import os.path as osp
from tqdm import tqdm
from utils.utils import *
from pretrain_gs import *
from argparse import ArgumentParser
from training_procedure import Trainer
from utils.random_seeder import set_random_seed
from DataHelper.DatasetLocal import DatasetLocal

from torch.utils.tensorboard import SummaryWriter

def main(args, config, logger, dataset: DatasetLocal):

    T                                = Trainer(config=config, args= args)
    model, optimizer, loss_func      = T.init(dataset)          # init gsc model
    if config.get('use_gs', False):
        gs_model, gs_optim, gs_loss_func = T.init_gs(dataset)   # init gumbel-sinkhorn model
        if config.get('run_first', False):
            print("train gs model")
            gs_model = pretrain_gs(T, config, dataset, gs_model, gs_loss_func, gs_optim)
        else:
            print("load saved model")
            PATH_PRETRAIN = os.path.join(os.getcwd(),'model_saved/pretrained_model')
            PATH_MODEL    = os.path.join(PATH_PRETRAIN, config['dataset_name'] + '_' + str(config['gs_epochs']) + '_pretrained_model.pth')
            gs_model.load_state_dict(torch.load(PATH_MODEL))

    pbar                             = tqdm(range(config['epochs']), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
    best_mse_metric                  = 100000.0
    best_metric_epoch 	             = -1
    report_mse_test 	             = 0
    report_rho_test                  = 0
    report_prec_at_10_test           = 0
    best_val_mse                     =  100000.
    best_val_tau                     = -100000.
    best_val_rho                     = -100000.
    best_val_p10                     = -100000.
    best_val_p20                     = -100000.
    best_val_epoch                   = -1
    loss_list                        = []
    monitor                          = config['monitor']
    best_val_paths                   = [None        , None        , None        , None        , None        ]
    best_val_metric                  = [best_val_mse, best_val_rho, best_val_tau, best_val_p10, best_val_p20]
    b_epoch                          = 0
    
    writer                           = SummaryWriter(RESULT_PATH)
    
    for epoch in pbar:
        batches                      = dataset.create_batches(config)   # 128对 graph-pair
        
        main_index                   = 0
        emb_loss_sum                 = 0
        per_loss_sum                 = 0 

        for batch_pair in batches:
            data                     = dataset.transform_batch(batch_pair, config)
            
            target                   = data["target"].cuda()
            model, loss1             = T.train(data, model, loss_func, optimizer, target, config)   
            main_index               = main_index + batch_pair[0].num_graphs               

            if config.get('use_gs', False):
                loss2                = T.train_gs(data, gs_model, gs_loss_func, gs_optim, target)
                loss_sum             = loss1 * (1 - config['gs_weight'])  + loss2 * config['gs_weight']
                loss_sum.backward()
                optimizer.step()
                gs_optim.step()

                per_loss_sum         = per_loss_sum + loss2.item()
            else:
                loss1.backward()
                optimizer.step()

            emb_loss_sum             = emb_loss_sum + loss1.item()

        emb_loss = emb_loss_sum / main_index    
        if config.get('use_gs', False):
            per_loss = per_loss_sum / main_index
            loss     = emb_loss * (1 - config['gs_weight'])  + per_loss * config['gs_weight']

            logger.info("per_loss:{}".format(per_loss))
            writer.add_scalar("per_loss", per_loss, epoch)
        else:
            loss     = emb_loss

        logger.info("emb_loss:{}".format(emb_loss))
        logger.info("sum_loss:{}".format(loss))
        logger.info("="*20)

        loss_list.append(loss)
        writer.add_scalar("emb_loss", emb_loss, epoch)
        writer.add_scalar("sum_loss", loss, epoch)
        if config['use_val']:
            if epoch >= config['iter_val_start'] and epoch % config['iter_val_every'] ==0:
                model.eval()
                if config.get('use_gs', False):
                    val_mse, val_rho, val_tau, val_prec_at_10, val_prec_at_20 = T.evaluation(dataset.val_graphs, dataset.training_graphs, model, dataset, validation=True, gs_model=gs_model)
                else:
                    val_mse, val_rho, val_tau, val_prec_at_10, val_prec_at_20 = T.evaluation(dataset.val_graphs, dataset.training_graphs, model, dataset, validation=True)
                logger.info("Validation Epoch = {}, MSE = {}(e-3), rho = {}, tau={}, prec_10 = {}, prec_20 = {}".format(epoch, val_mse*1000, val_rho, val_tau, val_prec_at_10, val_prec_at_20))
                if not config.get('save_best_all', False):  # run this 保存模型
                    logger.info("start save_best_all")
                    if best_mse_metric                >= val_mse:
                        best_mse_metric               = val_mse
                        best_val_epoch                = epoch
                        best_val_mse                  = val_mse
                        best_val_tau                  = val_tau
                        best_val_rho                  = val_rho
                        best_val_p10                  = val_prec_at_10
                        best_val_p20                  = val_prec_at_20
                        if config['save_best']:
                            best_val_model_path       = save_best_val_model(config, args.dataset, model, RESULT_PATH)

                else:
                    logger.info("start save_best_val_model_all")
                    current_metric                    = [val_mse        , val_rho     , val_tau     , val_prec_at_10,  val_prec_at_20, epoch]
                    best_val_metric, best_val_paths, b_epoch = save_best_val_model_all(config, args.dataset, model, RESULT_PATH, current_metric, best_val_metric, best_val_paths, b_epoch)
                    best_mse_metric                   = best_val_metric[0]
                    best_val_mse                      = best_val_metric[0]
                    best_val_rho                      = best_val_metric[1]
                    best_val_tau                      = best_val_metric[2]
                    best_val_p10                      = best_val_metric[3]
                    best_val_p20                      = best_val_metric[4]
                    best_val_epoch                    = b_epoch


        if epoch != config['epochs']-1:
            postfix_str = "<Epoch %d> [Train Loss] %.5f"% (epoch ,      loss) 
        elif epoch == config['epochs'] and config.get('show_last', False): 
            mse, rho, tau, prec_at_10, prec_at_20 = T.evaluation(dataset.testing_graphs, dataset.training_graphs, model, loss_func, dataset)
            best_mse_metric                       = mse
            best_metric_epoch                     = epoch
            report_mse_test                       = mse
            report_rho_test                       = rho
            report_tau_test                       = tau
            report_prec_at_10_test                = prec_at_10
            report_prec_at_20_test                = prec_at_20
            
            postfix_str = "<Epoch %d> [Train Loss] %.4f [Cur Tes %s] %.4f <Best Epoch %d> [Best Tes mse] %.4f [rho] %.4f [tau] %.4f [prec_at_10] %.4f [prec_at_20] %.4f " % ( 
                            epoch ,      loss,         monitor,      eval(monitor),  
                            best_metric_epoch ,report_mse_test, report_rho_test,report_tau_test,report_prec_at_10_test,report_prec_at_20_test)
        else:
            postfix_str = "<Epoch %d> [Train Loss] %.5f"% ( 
                            epoch ,      loss)
            
        pbar.set_postfix_str(postfix_str)

    logger.info("start testing using best val model: {}".format(postfix_str))

    if not config.get('save_best_all', False):
        logger.info("load model")
        model.load_state_dict(torch.load(best_val_model_path))
        if config.get('use_gs', False):
            test_mse, test_rho, test_tau, test_prec_at_10, test_prec_at_20 = T.evaluation(dataset.testing_graphs, dataset.trainval_graphs, model, dataset, gs_model=gs_model)
        else:
            test_mse, test_rho, test_tau, test_prec_at_10, test_prec_at_20 = T.evaluation(dataset.testing_graphs, dataset.trainval_graphs, model, dataset)
    else:
        met_test                                                       = load_model_all(dataset, model, loss_func, best_val_paths, T)
        test_mse, test_rho, test_tau, test_prec_at_10, test_prec_at_20 = met_test

    best_val_result = {
        'best_val_epoch': best_val_epoch,
        'best_val_mse'  : best_val_mse,
        'best_val_tau'  : best_val_tau,
        'best_val_rho'  : best_val_rho,
        'best_val_p10'  : best_val_p10,
        'best_val_p20'  : best_val_p20
    }
    writer.close() 

    return model, best_val_epoch , test_mse, test_rho, test_tau, test_prec_at_10, test_prec_at_20, loss, best_val_result

def print_evaluation(model_error,rho,tau,prec_at_10,prec_at_20):
    """
    Printing the error rates.
    """
    logger.info("\nmse(10^-3): "   + str(round(model_error * 1000, 5))         + ".")
    logger.info("Spearman's rho: " + str(round(rho, 5))                        + ".")
    logger.info("Kendall's tau: "  + str(round(tau, 5))                        + ".")
    logger.info("p@10: "           + str(round(prec_at_10, 5))                 + ".")
    logger.info("p@20: "           + str(round(prec_at_20, 5))                 + ".")
    print("\nmse(10^-3): "   + str(round(model_error * 1000, 5))         + ".")
    print("Spearman's rho: " + str(round(rho, 5))                        + ".")
    print("Kendall's tau: "  + str(round(tau, 5))                        + ".")
    print("p@10: "           + str(round(prec_at_10, 5))                 + ".")
    print("p@20: "           + str(round(prec_at_20, 5))                 + ".")
    
if __name__ == "__main__":
    ap = ArgumentParser()
    ap.add_argument('--model_name',        type = str,            default = 'IGFM')
    ap.add_argument('--dataset',           type = str,            default = 'AIDS700nef')
    ap.add_argument('--config_dir',        type = str,            default = 'config/')
    ap.add_argument('--gpu_id',            type = int,            default = 0)
    args = ap.parse_args()
    torch.cuda.set_device(args.gpu_id)

    # path to save log and model file
    RESULT_PATH  = os.path.join(os.getcwd(),'model_saved/', args.model_name, args.dataset, datetime.datetime.now().strftime('%Y-%m-%d'), datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    if not os.path.exists(RESULT_PATH):
        os.makedirs(RESULT_PATH)

    # logger setup
    logger = logging.getLogger("GS Model")
    fh     = logging.FileHandler(filename=os.path.join(RESULT_PATH, "out.log"))               
    fh     .setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s", datefmt="%m/%d %H:%M:%S"))
    logger .setLevel(logging.DEBUG)
    logger .addHandler(fh)
    config_path                  = osp.join(args.config_dir, args.dataset + '.yml')
    config                       = get_config(config_path)
    config                       = config.get(args.model_name, 'GS_GSC')
    dev_ress                     = []
    tes_ress                     = []
    tra_ress                     = []
    if config.get('seed',-1)     > 0:
        set_random_seed(config['seed'])
        logger.info("Seed set. %d" % (config['seed']))
    dataset                      = load_data(args)

    if config.get('x_augment', False):
        dataset.load_core_data(config) 
    else:
        dataset.load(config)
    config['max_set_size'] = dataset.get_max_node_size()

    logger.info("total graphs = {}"                                         .format(dataset.num_graphs))
    logger.info("train_gs.len = {} and val_gs.len = {} and test_gs.len = {}".format(dataset.num_train_graphs, dataset.num_val_graphs, dataset.num_test_graphs))
    for run_id in range(config['multirun']):   # one mask
        logger.info("\t\t%d th Run" % run_id)

        model, best_metric_epoch, report_mse_test, report_rho_test,report_tau_test,report_prec_at_10_test,report_prec_at_20_test, loss, best_val_results = main(args, config, logger, dataset)

        print_evaluation(report_mse_test,report_rho_test,report_tau_test,report_prec_at_10_test,report_prec_at_20_test)

        test_results = {
            'mse'       : report_mse_test,
            'rho'       : report_rho_test,
            'tau'       : report_tau_test,
            'prec_at_10': report_prec_at_10_test,
            'prec_at_20': report_prec_at_20_test
        }
        # model_saved/result.txt
        with open(osp.join(RESULT_PATH, 'result.txt'), 'w') as f:
            f.write('model: %s\n' % config.get('model_name'))
            f.write('\n')
            for k, v       in   best_val_results.items():
                f.write('%s: %s\n'         % (k, v))
            f.write('\n')
            for key, value in   test_results.items():
                f.write('%s: %s\n'         % (key, value))