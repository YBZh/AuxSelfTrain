import torch
import ipdb
import argparse
import os
import numpy as np
from torch.backends import cudnn
import random
import sys
import pprint
import json
import time
from utils.extract_acc import extract_acc_from_file
import datetime

def opts():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train script.')
    ### On data loader
    parser.add_argument('--source', type=str, default='Art', help='source domain')
    parser.add_argument('--target', type=str, default='Clipart', help='target domain')
    parser.add_argument('--test', type=str, default='', help='target domain')
    parser.add_argument('--dataset', type=str, default='DomainNet', help='OfficeHome | office | multi')
    parser.add_argument('--datapath', type=str, default='/data1/domain_adaptation/DomainNet', help='datapath')
    parser.add_argument('--num_labeled', type=int, default=3, help='number of labeled examples in the target')
    parser.add_argument('--transform_type', type=str, default='mme', help='ours | mme | simple')
    parser.add_argument('--batchsize', type=int, default=64, help='number of labeled examples in the target')
    parser.add_argument('--num_class', type=int, default=126, help='number of classes')
    parser.add_argument('--num_workers', type=int, default=4, help='number of classes')
    parser.add_argument('--ratio_t', type=float, default=0.5, help='threshold for pseudo labeling')
    parser.add_argument('--weak_aug', action='store_true', default=False,
                        help='process test data with weak aug')
    parser.add_argument('--mixup', type=str, default='none', help='vanilla | ours | none')
    parser.add_argument('--div_loss', type=str, default='l1', help='l1 | l2')
    parser.add_argument('--no_uniform_prior', action='store_true', default=False,
                        help='whether adopt the uniform prior knowledge')
    parser.add_argument('--ablation', type=str, default='none', help='no_struc | no_self_perceive | no_multi_step')
    parser.add_argument('--source_weight', type=str, default='none', help='none | overall | instance')

    ####
    parser.add_argument('--s_start', type=float, default=0.0, help='source rotation start from')
    parser.add_argument('--s_end', type=float, default=5.0, help='source rotation start from')
    parser.add_argument('--t_start', type=float, default=10.0, help='source rotation start from')
    parser.add_argument('--t_end', type=float, default=60.0, help='source rotation start from')
    parser.add_argument('--target_selection', type=str, default='ours', help='ours | gt | rand')

    parser.add_argument('--num_M', type=int, default=1000, help='number of intermediate domains')

    parser.add_argument('--category_mean', action='store_true', default=False,
                        help='for visda, if true, the score is the mean over all categories instead of all samples')
    parser.add_argument('--no_uniform', action='store_true', default=False,
                        help='remove the uniform prior')


    # for SSL, e.g., Cifar10 & Cifar100
    parser.add_argument('--n_iters_per_epoch', type=int, default=1024, help='unl is mu times larger than the labeled data')
    parser.add_argument('--fc_with_grad', action='store_true', default=False,
                        help='whether the gradient from auxiliary classifiers')
    parser.add_argument('--test_weakaug', action='store_true', default=False,
                        help='process test data with weak aug')  #### the test results are bad than the sample selection, since test and val data are different.
    parser.add_argument('--entropy_filter', action='store_true', default=False,
                        help='process test data with weak aug')


    # for fixmatch
    parser.add_argument('--mu', type=int, default=7, help='unl is mu times larger than the labeled data')
    parser.add_argument('--filter', type=int, default=32, help='32 for Cifar10 and 128 for Cifar 100')
    parser.add_argument('--thr', type=float, default=0.95, help='threshold for pseudo labeling')

    ## for classifiers consistency
    parser.add_argument('--filter_type', type=str, default='both', help='fc | cluster | ssl | both')
    parser.add_argument('--pseudo_type', type=str, default='hard', help='hard | soft')
    parser.add_argument('--weight', type=float, default=1.0, help='threshold for pseudo labeling')
    parser.add_argument('--feat_type_pseudo', type=str, default='eval', help='eval | train')
    parser.add_argument('--pre_trained_G', action='store_true', default=False, help='whether pre-train the feature extractor')
    parser.add_argument('--fixbn', action='store_true', default=False,
                        help='whether pre-train the feature extractor')
    parser.add_argument('--pseudo_label_generator', type=str, default='fc', help='fc | kmeans | lp')
    parser.add_argument('--init_center', type=str, default='tl', help='tl | s | st')
    parser.add_argument('--category_rank', action='store_true', default=False, help='assign pseudo label accroding category')
    parser.add_argument('--noprogressive', action='store_true', default=False,
                        help='assign pseudo label accroding category')
    parser.add_argument('--pace_strategy', type=str, default='implicit', help='implicit | manual')



    parser.add_argument('--lp_labeled', type=str, default='st', help='t | s | st')
    parser.add_argument('--lp_dis', type=str, default='cos', help='cos | l2 | nndescent')
    parser.add_argument('--lp_solver', type=str, default='closedform', help='closedform | CG')
    parser.add_argument('--lp_graphk', type=int, default=20, help='the number of nearest graph')
    parser.add_argument('--lp_alpha', type=float, default=0.75, help='')


    parser.add_argument('--prototype_classifier', action='store_true', default=False,
                        help='whether adopt loss from prototype classifier')

    # for source only strong
    parser.add_argument('--source_only_type', type=str, default='SwTw', help='type of data aug')

    # for fixmatch_dcf
    parser.add_argument('--d_layer', type=str, default='one', help='number of layers in discriminator')
    parser.add_argument('--d_thr', type=float, default=0.2, help='threshold for domain score')

    ### FixMatch_proto
    parser.add_argument('--dis', type=str, default='l2', help='')
    parser.add_argument('--thresh_type', type=str, default='prot', help='cla | prot | both')

    ### FixMatch_MMD
    parser.add_argument('--LearnM', action='store_true', default=False, help='whether M is learnable in each epoch')
    parser.add_argument('--WithoutM', action='store_true', default=False, help='both with and without mapping')
    parser.add_argument('--mmd', type=str, default='global', help='')
    parser.add_argument('--mmd_lambda', type=float, default=1.0, help='lambda')
    parser.add_argument('--L2norm', action='store_true', default=False, help='whether M is learnable in each epoch')
    parser.add_argument('--mmd_type', type=str, default='st2st', help='')

    ### On model construction
    parser.add_argument('--net', type=str, default='resnet34', help='which network to use')
    parser.add_argument('--multi_fc', action='store_true', default=False, help='whether adopt multiple fc layer as the classifier')
    parser.add_argument('--MME_classifier', action='store_true', default=False,
                        help='whether adopt the classifier proposed in MME')
    parser.add_argument('--T', default=0.05, type=float)
    parser.add_argument('--bottle_neck', action='store_true', default=False,
                        help='whether add a bottle neck at the end of the feature extractor')


    ### On optimization
    parser.add_argument('--optimizer', type=str, default='SGD', help='which network to use')
    parser.add_argument('--max_iters', type=int, default=10000, help='maximum number of iterations to train (default: 30000)')
    parser.add_argument('--pre_epoch', type=int, default=5,
                        help='number of epoch fine-tuning feature extractor and classifier before training')
    parser.add_argument('--base_lr', type=float, default=0.01, help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--wd', type=float, default=0.0001, help='weight decay')
    parser.add_argument('--lr_schedule', type=str, default='inv', help='lr change schedule')
    parser.add_argument('--inv_alpha', type=float, default=10, help='weight decay')
    parser.add_argument('--inv_beta', type=float, default=0.75, help='weight decay')
    parser.add_argument('--alpha', type=float, default=0.75, help='weight decay')
    parser.add_argument('--lambda_u', type=float, default=100, help='weight of the unlabeled data')
    parser.add_argument('--ema_decay', type=float, default=0.999, help='weight of the ema')
    parser.add_argument('--interleave', action='store_true', default=False, help='make each batch the same attribute')
    parser.add_argument('--pretrain_fc', action='store_true', default=False, help='pre-train the fc layer used in the paper.')
    parser.add_argument('--initial_only', action='store_true', default=False, help='watch the initial results only')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--fix_submodel', action='store_true', default=False, help='watch the initial results only')


    ### others

    parser.add_argument('--shallow_only', action='store_true', default=False, help='')
    parser.add_argument('--saving', action='store_true', default=False, help='')
    parser.add_argument('--method', help='set the method to use', default='MixMatch', type=str)
    parser.add_argument('--resume', help='set the resume file path', default='', type=str)
    parser.add_argument('--exp_name', help='the log name', default='log', type=str)
    parser.add_argument('--save_dir', help='the log file', default='mixmatch', type=str)
    parser.add_argument('--log_interval', type=int, default=100, help='how many batches to wait before logging training status')

    # parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    #
    parser.add_argument('--patience', type=int, default=5,
                        help='early stopping to wait for improvment '
                             'before terminating. (default: 5 (5 epoch))')
    parser.add_argument('--early', action='store_true', default=False,
                        help='early stopping on validation or not')


    args = parser.parse_args()

    if args.test == '':
        args.test = args.target

    if args.dataset == 'OfficeHome':
        args.num_class = 65
    elif args.dataset == 'Office31':
        args.num_class = 31
    elif args.dataset == 'DomainNet':
        args.num_class = 126
    elif args.dataset == 'image_CLEF':
        args.num_class = 12
    elif args.dataset == 'visDA':
        args.num_class = 12
    elif args.dataset == 'cifar10':
        args.num_class = 10
    elif args.dataset == 'mnist':
        args.num_class = 10
    elif args.dataset == 'cifar100':
        args.num_class = 100

    if args.method == 'cc_da':
        args.lp_labeled = 's'
        args.init_center = 's'

    data = datetime.date.today()
    args.exp_name = str(data.month) + str(data.day) + '_' + args.exp_name
    args.exp_name = args.exp_name + '_' + args.dataset + '_' + args.method + '_' + args.net + '_' + args.source + '2' + args.target + '_' \
                    + str(args.num_labeled) + 'labeled' + '_Filter' + args.filter_type + '_thr' + str(args.thr)

    if args.no_uniform:
        args.exp_name = args.exp_name + '_nouniform'


    if args.source_weight != 'none':
        args.exp_name = args.exp_name + '_sw' + args.source_weight

    if args.mixup != 'none':
        args.exp_name = args.exp_name + '_mixup' + args.mixup

    if args.method == 'FixMatch_proto':
        args.exp_name = args.exp_name + '_' + args.dis + '_' + args.thresh_type

    if args.method == 'ccfc_div':
        args.exp_name = args.exp_name + '_div' + args.div_loss

    if args.method == 'mnist':
        args.exp_name = args.exp_name + '_s_' + str(args.s_start) + 'to' + str(args.s_end) + '_t_' + str(args.t_start) + 'to' + str(args.t_end)

    if args.target_selection != 'ours':
        args.exp_name = args.exp_name + '_se_' + args.target_selection

    if args.method == 'ssda_ablation' or args.method == 'da_ablation':
        args.exp_name = args.exp_name +  args.ablation

    if args.ablation == 'no_struc' and args.filter_type != 'fc':
        raise NotImplementedError

    if args.method == 'cc':
        args.exp_name = args.exp_name + '_' + str(args.weight) + '_' + args.feat_type_pseudo + '_' + args.pseudo_type
        if args.pre_trained_G:
            args.exp_name = args.exp_name + '_preGF'

    return args

if __name__ == '__main__':

    cudnn.benchmark = True
    args = opts()
    if args.seed > 0:
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)
    print('Called with args:')
    print(args)

    args.save_dir = os.path.join(args.save_dir, args.exp_name)
    print('Output will be saved to %s.' % args.save_dir)
    if not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir)

    log = open(os.path.join(args.save_dir, 'log.txt'), 'a')
    log.write("\n")
    log.write('\n------------ training start ------------------\n')
    log.write(time.asctime(time.localtime(time.time())))
    log.write('\n-------------------------------------------\n')
    state = {k: v for k, v in args._get_kwargs()}
    log.write(json.dumps(state) + '\n')
    log.close()



    if args.method == 'SourceOnly':
        from model.construct_model import construct_model
        from data.prepare_data_fixmatch import generate_dataloader as Dataloader
        from solver.solver_sourceonly import Solver as Solver
        G, F = construct_model(args)
        dataloaders = Dataloader(args)

        G = torch.nn.DataParallel(G)
        F = torch.nn.DataParallel(F)
        if torch.cuda.is_available():
            G.cuda()
            F.cuda()

    # elif args.method == 'FixMatch':
    #     from model.construct_model import construct_model
    #     from data.prepare_data_fixmatch import generate_dataloader as Dataloader
    #     from solver.solver_fixmatch import Solver as Solver
    #     G, F = construct_model(args)
    #     dataloaders = Dataloader(args)
    #
    #     G = torch.nn.DataParallel(G)
    #     F = torch.nn.DataParallel(F)
    #     if torch.cuda.is_available():
    #         G.cuda()
    #         F.cuda()
    #
    # elif args.method == 'FixMatch_ssl':
    #     from model.construct_model import construct_model
    #     from data.prepare_data_cifar import generate_dataloader as Dataloader
    #     from solver.solver_fixmatch_ssl import Solver as Solver
    #     G, F = construct_model(args)
    #     dataloaders = Dataloader(args)
    #
    #     G = torch.nn.DataParallel(G)
    #     F = torch.nn.DataParallel(F)
    #     if torch.cuda.is_available():
    #         G.cuda()
    #         F.cuda()
    ## for Semi-supervised domain adaptaiton
    elif args.method == 'cc':
        from model.construct_model import construct_model
        from data.prepare_data_fixmatch import generate_dataloader as Dataloader
        from solver.solver_cc import Solver as Solver
        G, F = construct_model(args)
        dataloaders = Dataloader(args)

        G = torch.nn.DataParallel(G)
        F = torch.nn.DataParallel(F)
        if torch.cuda.is_available():
            G.cuda()
            F.cuda()
    ## for domain adaptation
    elif args.method == 'cc_da':
        from model.construct_model import construct_model
        from data.prepare_data_da import generate_dataloader as Dataloader
        from solver.solver_cc_da import Solver as Solver
        G, F = construct_model(args)
        dataloaders = Dataloader(args)

        G = torch.nn.DataParallel(G)
        F = torch.nn.DataParallel(F)
        if torch.cuda.is_available():
            G.cuda()
            F.cuda()

    # elif args.method == 'ssda_ablation':
    #     from model.construct_model import construct_model
    #     from data.prepare_data_fixmatch import generate_dataloader as Dataloader
    #     from solver.solver_ssda_ablation import Solver as Solver
    #     G, F = construct_model(args)
    #     dataloaders = Dataloader(args)
    #
    #     G = torch.nn.DataParallel(G)
    #     F = torch.nn.DataParallel(F)
    #     if torch.cuda.is_available():
    #         G.cuda()
    #         F.cuda()
    #
    # elif args.method == 'da_ablation':
    #     from model.construct_model import construct_model
    #     from data.prepare_data_da import generate_dataloader as Dataloader
    #     from solver.solver_da_ablation import Solver as Solver
    #     G, F = construct_model(args)
    #     dataloaders = Dataloader(args)
    #
    #     G = torch.nn.DataParallel(G)
    #     F = torch.nn.DataParallel(F)
    #     if torch.cuda.is_available():
    #         G.cuda()
    #         F.cuda()
    #
    # elif args.method == 'ccfc':
    #     from model.construct_model import construct_model
    #     from data.prepare_data_fixmatch import generate_dataloader as Dataloader
    #     from solver.solver_cc_fc import Solver as Solver
    #     G, F = construct_model(args)
    #     dataloaders = Dataloader(args)
    #
    #     G = torch.nn.DataParallel(G)
    #     F = torch.nn.DataParallel(F)
    #     if torch.cuda.is_available():
    #         G.cuda()
    #         F.cuda()
    #
    # elif args.method == 'ccfc_div':
    #     from model.construct_model import construct_model
    #     from data.prepare_data_fixmatch import generate_dataloader as Dataloader
    #     from solver.solver_cc_fc_div import Solver as Solver
    #     G, F = construct_model(args)
    #     dataloaders = Dataloader(args)
    #
    #     G = torch.nn.DataParallel(G)
    #     F = torch.nn.DataParallel(F)
    #     if torch.cuda.is_available():
    #         G.cuda()
    #         F.cuda()



    # elif args.method == 'pi_da':
    #     from model.construct_model import construct_model
    #     from data.prepare_data_da import generate_dataloader as Dataloader
    #     from solver.solver_pi_da import Solver as Solver
    #     G, F = construct_model(args)
    #     dataloaders = Dataloader(args)
    #
    #     G = torch.nn.DataParallel(G)
    #     F = torch.nn.DataParallel(F)
    #     if torch.cuda.is_available():
    #         G.cuda()
    #         F.cuda()
    #
    # elif args.method == 'da_discrimination':
    #     from model.construct_model import construct_model
    #     from data.prepare_data_da import generate_dataloader as Dataloader
    #     from solver.solver_da_discrimination import Solver as Solver
    #     G, F = construct_model(args)
    #     dataloaders = Dataloader(args)
    #
    #     G = torch.nn.DataParallel(G)
    #     F = torch.nn.DataParallel(F)
    #     if torch.cuda.is_available():
    #         G.cuda()
    #         F.cuda()
    #
    # elif args.method == 'dann':
    #     from model.construct_model import construct_model
    #     from data.prepare_data_da import generate_dataloader as Dataloader
    #     from solver.solver_dann_da import Solver as Solver
    #     G, F = construct_model(args)
    #     dataloaders = Dataloader(args)
    #
    #     G = torch.nn.DataParallel(G)
    #     F = torch.nn.DataParallel(F)
    #     if torch.cuda.is_available():
    #         G.cuda()
    #         F.cuda()
    #
    # elif args.method == 'cc_da_variousM':
    #     from model.construct_model import construct_model
    #     from data.prepare_data_da import generate_dataloader as Dataloader
    #     from solver.solver_cc_da_variousM import Solver as Solver
    #     G, F = construct_model(args)
    #     dataloaders = Dataloader(args)
    #
    #     G = torch.nn.DataParallel(G)
    #     F = torch.nn.DataParallel(F)
    #     if torch.cuda.is_available():
    #         G.cuda()
    #         F.cuda()
    #
    # elif args.method == 'mnist':
    #     from model.construct_model import construct_model
    #     from data.prepare_data_mnist import generate_dataloader as Dataloader
    #     from solver.solver_mnist import Solver as Solver
    #     G, F = construct_model(args)
    #     dataloaders = Dataloader(args)
    #
    #     G = torch.nn.DataParallel(G)
    #     F = torch.nn.DataParallel(F)
    #     if torch.cuda.is_available():
    #         G.cuda()
    #         F.cuda()
    #
    # elif args.method == 'mnist_plot':
    #     from model.construct_model import construct_model
    #     from data.prepare_data_mnist import generate_dataloader as Dataloader
    #     from solver.solver_mnist_acc_score_a_dis import Solver as Solver
    #     G, F = construct_model(args)
    #     dataloaders = Dataloader(args)
    #
    #     G = torch.nn.DataParallel(G)
    #     F = torch.nn.DataParallel(F)
    #     if torch.cuda.is_available():
    #         G.cuda()
    #         F.cuda()
    #
    # elif args.method == 'visda_test':
    #     from model.construct_model import construct_model
    #     from data.prepare_visda_test import generate_dataloader as Dataloader
    #     from solver.solver_da_eval_visda import Solver as Solver
    #     G, F = construct_model(args)
    #     dataloaders = Dataloader(args)
    #
    #     G = torch.nn.DataParallel(G)
    #     F = torch.nn.DataParallel(F)
    #     if torch.cuda.is_available():
    #         G.cuda()
    #         F.cuda()
    #
    # elif args.method == 'da_source_only':
    #     from model.construct_model import construct_model
    #     from data.prepare_data_da import generate_dataloader as Dataloader
    #     from solver.solver_da_source_only import Solver as Solver
    #     G, F = construct_model(args)
    #     dataloaders = Dataloader(args)
    #
    #     G = torch.nn.DataParallel(G)
    #     F = torch.nn.DataParallel(F)
    #     if torch.cuda.is_available():
    #         G.cuda()
    #         F.cuda()
    #
    # elif args.method == 'ccfc_da':
    #     from model.construct_model import construct_model
    #     from data.prepare_data_da import generate_dataloader as Dataloader
    #     from solver.solver_cc_fc_da import Solver as Solver
    #     G, F = construct_model(args)
    #     dataloaders = Dataloader(args)
    #
    #     G = torch.nn.DataParallel(G)
    #     F = torch.nn.DataParallel(F)
    #     if torch.cuda.is_available():
    #         G.cuda()
    #         F.cuda()
    #
    # elif args.method == 'cc_ssl':
    #     from model.construct_model import construct_model
    #     from data.prepare_data_cifar import generate_dataloader as Dataloader
    #     from solver.solver_cc_ssl import Solver as Solver
    #     G, F = construct_model(args)
    #     dataloaders = Dataloader(args)
    #
    #     G = torch.nn.DataParallel(G)
    #     F = torch.nn.DataParallel(F)
    #     if torch.cuda.is_available():
    #         G.cuda()
    #         F.cuda()
    #
    # elif args.method == 'sl':
    #     from model.construct_model import construct_model
    #     from data.prepare_data_cifar import generate_dataloader_sl as Dataloader
    #     from solver.solver_ssl_upbound import Solver as Solver
    #     G, F = construct_model(args)
    #     dataloaders = Dataloader(args)
    #
    #     G = torch.nn.DataParallel(G)
    #     F = torch.nn.DataParallel(F)
    #     if torch.cuda.is_available():
    #         G.cuda()
    #         F.cuda()
    #
    # elif args.method == 'ccfc_ssl':
    #     from model.construct_model import construct_model
    #     from data.prepare_data_cifar import generate_dataloader as Dataloader
    #     from solver.solver_cc_fc_ssl import Solver as Solver
    #     G, F = construct_model(args)
    #     dataloaders = Dataloader(args)
    #
    #     G = torch.nn.DataParallel(G)
    #     F = torch.nn.DataParallel(F)
    #     if torch.cuda.is_available():
    #         G.cuda()
    #         F.cuda()





    # initialize solver
    train_solver = Solver(G, F, dataloaders, args)
    # train
    if args.shallow_only:
        train_solver.shallow()
    else:
        train_solver.solve()

    print('Finished!')
    selected_acc, normal_acc, ema_acc = extract_acc_from_file(os.path.join(args.save_dir, 'log.txt'))

    log = open(os.path.join(args.save_dir, 'log.txt'), 'a')
    log.write("\n")
    log.write("  Selected_acc: %3f, normal_acc: %3f, ema_acc: %3f" % \
              (selected_acc, normal_acc, ema_acc))
    log.write('\n------------ training end ------------------\n')
    log.write(time.asctime(time.localtime(time.time())))
    log.write('\n-------------------------------------------\n')
    log.close()