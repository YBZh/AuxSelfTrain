import torch
import os
import math
import torch.nn as nn
import time
import numpy as np
from .base_solver import BaseSolver
import torch.nn.functional as F
from utils.utils import AverageMeter, to_cuda, accuracy, weight_ema, to_onehot, EMA_fixmatch, LabelGuessor, fix_bn, \
    release_bn, get_labels_from_classifier_prediction, get_labels_from_Sphericalkmeans, get_labels_from_kmeans, get_labels_from_lp
import ipdb
from torch.distributions import Categorical
from sklearn import svm
import ot
import ot.plot
import random

def accuracy_for_each_class(output, target, total_vector, correct_vector):
    """Computes the precision for each class"""
    batch_size = target.size(0)
    _, pred = output.topk(1, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1)).float().cpu().squeeze()
    for i in range(batch_size):
        total_vector[target[i]] += 1
        correct_vector[torch.LongTensor([target[i]])] += correct[i]

    return total_vector, correct_vector

def proxy_a_distance(source_X, target_X, verbose=False):
    """
    Compute the Proxy-A-Distance of a source/target representation
    """
    nb_source = np.shape(source_X)[0]
    nb_target = np.shape(target_X)[0]

    if verbose:
        print('PAD on', (nb_source, nb_target), 'examples')

    C_list = np.logspace(-5, -1, 5)

    half_source, half_target = int(nb_source/2), int(nb_target/2)
    train_X = np.vstack((source_X[0:half_source, :], target_X[0:half_target, :]))
    train_Y = np.hstack((np.zeros(half_source, dtype=int), np.ones(half_target, dtype=int)))

    test_X = np.vstack((source_X[half_source:, :], target_X[half_target:, :]))
    test_Y = np.hstack((np.zeros(nb_source - half_source, dtype=int), np.ones(nb_target - half_target, dtype=int)))

    best_risk = 1.0
    for C in C_list:
        clf = svm.SVC(C=C, kernel='linear', verbose=False)
        clf.fit(train_X, train_Y)

        train_risk = np.mean(clf.predict(train_X) != train_Y)
        test_risk = np.mean(clf.predict(test_X) != test_Y)

        if verbose:
            print('[ PAD C = %f ] train risk: %f  test risk: %f' % (C, train_risk, test_risk))

        if test_risk > .5:
            test_risk = 1. - test_risk

        best_risk = min(best_risk, test_risk)

    return 2 * (1. - 2 * best_risk)


def wasserstein_infinity_calculation(s_cate_i, t_cate_i, power=10):
    ### set power as 10 to approximate the wasserstein_infinity
    nb_source = np.shape(s_cate_i)[0]
    nb_target = np.shape(t_cate_i)[0]
    if nb_source == 0 or nb_target == 0:
        return 0
    else:
        num_min = min(nb_target, nb_source)
        s_cate_i = s_cate_i[: num_min]
        t_cate_i = t_cate_i[: num_min]
        s_cate_i = s_cate_i.reshape(num_min, 1, -1)
        t_cate_i = t_cate_i.reshape(num_min, -1)
        M = ((s_cate_i - t_cate_i) ** 2).sum(2) ** 0.5 #### the pair wise distance
        M = M ** power
        a = np.ones(num_min)
        b = np.ones(num_min)
        ot_cost = ot.emd2(a, b, M, numItermax=2000000)
        ot_cost = ot_cost ** (1/power)
        return ot_cost


class Solver(BaseSolver):
    def __init__(self, G, F, dataloaders, args, **kwargs):
        super(Solver, self).__init__(G, F, dataloaders, args, **kwargs)

        self.ema = EMA_fixmatch(G, F, args.ema_decay)
        self.lb_guessor = LabelGuessor(thresh=args.thr)
        self.CELoss = nn.CrossEntropyLoss(reduction='none')
        from data.prepare_data_da import generate_dataloader_pseudo_label as Dataloader
        dataloaders_pseudo = Dataloader(args)
        self.init_data_pseudo(dataloaders_pseudo)
        self.selected_index = None
        self.selected_index_source = None
        if args.resume != '':
            resume_dict = torch.load(args.resume)
            self.G.load_state_dict(resume_dict['G_state_dict'])
            self.F.load_state_dict(resume_dict['F_state_dict'])
            self.best_prec1 = resume_dict['best_prec1']
            self.iter = resume_dict['iter']

    def init_data_pseudo(self, dataloaders):
        self.pseudo_data = {key: dict() for key in dataloaders}
        for key in self.pseudo_data.keys():
            if key not in dataloaders:
                continue
            cur_dataloader = dataloaders[key]
            self.pseudo_data[key]['loader'] = cur_dataloader

    def pre_train_classifier(self):
        # if not self.args.pre_trained_G:
        #     print('fix the running mean and var for BN')
        #     self.G.apply(fix_bn)
        #     self.F.apply(fix_bn)
        if self.args.fixbn:
            print('fix the running mean and var for BN')
            self.G.apply(fix_bn)
            self.F.apply(fix_bn)
        initial_lr = self.args.base_lr
        for i in range(self.args.pre_epoch):
            new_lr = initial_lr #/ (10 ** i)
            print('new lr for classifier training is: %3f' % (new_lr))
            self.G.train()
            self.F.train()
            self.train_data['source']['iterator'] = iter(self.train_data['source']['loader'])
            self.train_data['target']['iterator'] = iter(self.train_data['target']['loader'])
            self.update_lr(given_lr=new_lr)
            self.iters_per_epoch = len(self.train_data['source']['loader'])
            print('iters in each epoch is: %d' % (self.iters_per_epoch))
            iters_counter_within_epoch = 0
            data_time = AverageMeter()
            batch_time = AverageMeter()
            losses_all = AverageMeter()
            losses_s = AverageMeter()
            losses_t = AverageMeter()
            stop = False
            end = time.time()
            while not stop:
                source_data, _, source_gt, _, _ = self.get_samples('source')
                # target_data_u, _, _, _, _ = self.get_samples('target')
                ########################
                source_data = to_cuda(source_data)
                source_gt = to_cuda(source_gt)

                data_time.update(time.time() - end)
                logit = self.F(self.G(source_data))
                # feature_not_use = self.G(target_data)   ### update the bn with target data.
                loss = self.CELoss(logit, source_gt).mean()

                self.optimizer_G.zero_grad()
                self.optimizer_F.zero_grad()
                loss.backward()
                if self.args.pre_trained_G:
                    self.optimizer_G.step()
                self.optimizer_F.step()
                self.ema.update_params()

                losses_all.update(loss.item(), source_data.size(0))

                batch_time.update(time.time() - end)
                end = time.time()
                self.iters += 1
                iters_counter_within_epoch += 1
                if self.iters % 10 == 0:
                    print(
                                "  Pre-Train:epoch: %d:[%d/%d], Tdata: %3f, Tbatch: %3f, LossL: %3f, LossU: %3f, LossAll:%3f" % \
                                (i, self.iters, self.args.max_iters, data_time.avg, batch_time.avg, losses_s.avg,
                                 losses_t.avg, losses_all.avg))

                if iters_counter_within_epoch >= self.iters_per_epoch:
                    log = open(os.path.join(self.args.save_dir, 'log.txt'), 'a')
                    log.write("\n")
                    log.write(
                        "  Pre-Train:epoch: %d:[%d/%d], Tdata: %3f, Tbatch: %3f, LossL: %3f, LossU: %3f, LossAll:%3f" % \
                            (i, self.iters, self.args.max_iters, data_time.avg, batch_time.avg, losses_s.avg,
                             losses_t.avg, losses_all.avg))
                    log.close()
                    stop = True
                    self.ema.update_buffer()

            acc, acc_val = self.test()
            log = open(os.path.join(self.args.save_dir, 'log.txt'), 'a')
            log.write("          Best acc by far:%3f" % (acc))
            log.close()
        # if not self.args.pre_trained_G:
        #     self.G.apply(release_bn)
        #     self.F.apply(release_bn)

    def Get_pseudo_labels_with_classifiers_consistency(self):
        # self.G.apply(fix_bn)
        # self.F.apply(fix_bn)
        if self.args.feat_type_pseudo == 'train':
            self.G.train()
            self.F.train()
        elif self.args.feat_type_pseudo == 'eval':
            self.G.eval()
            self.F.eval()
        else:
            raise NotImplementedError
        ################## prepare all features and other ###############################################
        target_u_feature_list = []
        target_u_prediction_list = []
        target_u_index_list = []
        target_u_label_list = []
        target_u_path_list = []
        print('prepare feature of target unlabeled data')
        for i, (input, _, target_for_visual, index, path) in enumerate(self.pseudo_data['target']['loader']):
            if i % 100 == 0:
                print(i)
            input = to_cuda(input)
            target_for_visual = to_cuda(target_for_visual)
            if self.args.feat_type_pseudo == 'train':
                org_state_G = {
                    k: v.clone().detach()
                    for k, v in self.G.state_dict().items()
                }
                org_state_F = {
                    k: v.clone().detach()
                    for k, v in self.F.state_dict().items()
                }
            with torch.no_grad():
                target_u_feature_iter = self.G(input)
                target_u_prediction_itre = self.F(target_u_feature_iter)
            if self.args.feat_type_pseudo == 'train':
                self.G.load_state_dict(org_state_G)
                self.F.load_state_dict(org_state_F)
            target_u_feature_list.append(target_u_feature_iter)
            target_u_prediction_list.append(target_u_prediction_itre)
            target_u_index_list.append(index)
            # ipdb.set_trace()
            # target_u_path_list+=path
            target_u_label_list.append(target_for_visual)

        target_u_feature_matrix = torch.cat(target_u_feature_list, dim=0)
        target_u_prediction_matrix = torch.cat(target_u_prediction_list, dim=0)
        target_u_index = torch.cat(target_u_index_list, dim=0)
        target_u_gt_label_for_visual = torch.cat(target_u_label_list)

        source_feature_list = []
        source_label_list = []
        source_index_list = []
        source_cate_feature_list = []
        for i in range(self.args.num_class):
            source_cate_feature_list.append([])
        print('prepare features of source data')
        for i, (input, _, target, index, path) in enumerate(self.pseudo_data['source']['loader']):
            input = to_cuda(input)
            if self.args.feat_type_pseudo == 'train':
                org_state_G = {
                    k: v.clone().detach()
                    for k, v in self.G.state_dict().items()
                }
                org_state_F = {
                    k: v.clone().detach()
                    for k, v in self.F.state_dict().items()
                }
            with torch.no_grad():
                source_feature_iter = self.G(input)
            if self.args.feat_type_pseudo == 'train':
                self.G.load_state_dict(org_state_G)
                self.F.load_state_dict(org_state_F)
            source_feature_list.append(source_feature_iter)
            source_label_list.append(target)
            source_index_list.append(index)
            for j in range(input.size(0)):
                img_label = target[j]
                source_cate_feature_list[img_label].append(source_feature_iter[j].view(1, source_feature_iter.size(1)))
        source_feature_matrix = torch.cat(source_feature_list, dim=0)
        hard_label_s = torch.cat(source_label_list, dim=0)
        source_index = torch.cat(source_index_list, dim=0)
        soft_label_s = to_onehot(hard_label_s, self.args.num_class)
        ############################################################################################

        ################# get the prediction with the discriminative clustering#######
        soft_label_fc, soft_label_uniform_fc, hard_label_fc, hard_label_uniform_fc, acc_fc, acc_uniform_fc = get_labels_from_classifier_prediction(target_u_prediction_matrix, self.args.T, target_u_gt_label_for_visual)
        if self.args.no_uniform:
            soft_label_uniform_fc = soft_label_fc
            hard_label_uniform_fc = hard_label_fc
            acc_uniform_fc = acc_fc
        log = open(os.path.join(self.args.save_dir, 'log.txt'), 'a')
        log.write("\n")
        log.write('FC Prediction without and with uniform prior are: %3f and %3f' % (acc_fc, acc_uniform_fc))
        log.close()
        scores_fc, hard_label_fc = torch.max(soft_label_uniform_fc, dim=1)

        if self.args.filter_type == 'cluster' or self.args.filter_type == 'cluster_lp' or self.args.filter_type == 'all':
            if self.args.pseudo_label_generator == 'spheticalkmeans':
                for i in range(self.args.num_class):
                    source_cate_feature_list[i] = torch.cat(source_cate_feature_list[i], dim=0)
                    source_cate_feature_list[i] = source_cate_feature_list[i].mean(0)
                    source_cate_feature_list[i] = F.normalize(source_cate_feature_list[i], dim=0, p=2)
                    source_cate_feature_list[i] = source_cate_feature_list[i].cpu().numpy()
                source_cate_feature_list = np.array(source_cate_feature_list)

                target_u_feature_matrix_norm = F.normalize(target_u_feature_matrix, dim=1, p=2)
                soft_label_kmean, soft_label_uniform_kmean, hard_label_kmean, hard_label_uniform_kmean, acc_kmean, acc_uniform_kmean = \
                get_labels_from_Sphericalkmeans(initial_centers_array=source_cate_feature_list, target_u_feature=target_u_feature_matrix_norm.cpu(),
                                                num_class=self.args.num_class, gt_label=target_u_gt_label_for_visual.cpu(), T=0.05, max_iter=100, target_l_feature=None)
                if self.args.no_uniform:
                    soft_label_uniform_kmean = soft_label_kmean
                    hard_label_uniform_kmean = hard_label_kmean
                    acc_uniform_kmean = acc_kmean
                log = open(os.path.join(self.args.save_dir, 'log.txt'), 'a')
                log.write("\n")
                log.write('sphetical Kmeans Prediction without and with uniform prior are: %3f and %3f' % (acc_kmean, acc_uniform_kmean))
                log.close()
                # scores, hard_label_fc = torch.max(soft_label_uniform_kmean, dim=1)
                # idx = target_u_index[scores > self.args.thr]
                # self.all_hard_pseudo_label = hard_label_uniform_kmean
                # self.all_soft_pseudo_label = soft_label_uniform_kmean

            elif self.args.pseudo_label_generator == 'kmeans':
                for i in range(self.args.num_class):
                    ## only one option here, adopt the source data to initial centers
                    source_cate_feature_list[i] = torch.cat(source_cate_feature_list[i], dim=0)
                    source_cate_feature_list[i] = source_cate_feature_list[i].mean(0)
                    source_cate_feature_list[i] = source_cate_feature_list[i].cpu().numpy()
                target_l_cate_feature_list = np.array(source_cate_feature_list)

                # target_u_feature_matrix_norm = F.normalize(target_u_feature_matrix, dim=1, p=2)
                # target_l_feature_matrix_norm = F.normalize(target_l_feature_matrix, dim=1, p=2)
                soft_label_kmean, soft_label_uniform_kmean, hard_label_kmean, hard_label_uniform_kmean, acc_kmean, acc_uniform_kmean = \
                get_labels_from_kmeans(initial_centers_array=target_l_cate_feature_list, target_u_feature=target_u_feature_matrix.cpu(),
                                                num_class=self.args.num_class, gt_label=target_u_gt_label_for_visual.cpu(), T=0.05, max_iter=100, target_l_feature=None)
                if self.args.no_uniform:
                    soft_label_uniform_kmean = soft_label_kmean
                    hard_label_uniform_kmean = hard_label_kmean
                    acc_uniform_kmean = acc_kmean
                log = open(os.path.join(self.args.save_dir, 'log.txt'), 'a')
                log.write("\n")
                log.write('Kmeans Prediction without and with uniform prior are: %3f and %3f' % (acc_kmean, acc_uniform_kmean))
                log.close()
                # scores, hard_label_fc = torch.max(soft_label_uniform_kmean, dim=1)
                # idx = target_u_index[scores > self.args.thr]
                # self.all_hard_pseudo_label = hard_label_uniform_kmean
                # self.all_soft_pseudo_label = soft_label_uniform_kmean
        if self.args.filter_type == 'lp' or self.args.filter_type == 'fc_lp' or self.args.filter_type == 'cluster_lp' or self.args.filter_type == 'all':
            if self.args.lp_labeled == 's':
                labeled_feature_matrix = source_feature_matrix
                labeled_onehot = soft_label_s
            else:
                raise NotImplementedError

            soft_label_lp, soft_label_uniform_lp, hard_label_lp, hard_label_uniform_lp, acc_lp, acc_uniform_lp = \
            get_labels_from_lp(labeled_features=labeled_feature_matrix.cpu(), labeled_onehot_gt=labeled_onehot.cpu(),
                               unlabeled_features = target_u_feature_matrix.cpu(), gt_label=target_u_gt_label_for_visual.cpu(),
                               num_class=self.args.num_class, dis=self.args.lp_dis, solver=self.args.lp_solver, graphk = self.args.lp_graphk, alpha=self.args.lp_alpha)
            if self.args.no_uniform:
                soft_label_uniform_lp = soft_label_lp
                hard_label_uniform_lp = hard_label_lp
                acc_uniform_lp = acc_lp
            log = open(os.path.join(self.args.save_dir, 'log.txt'), 'a')
            log.write("\n")
            log.write('LP Prediction without and with uniform prior are: %3f and %3f' % (acc_lp, acc_uniform_lp))
            log.close()
            # scores, hard_label_fc = torch.max(soft_label_uniform_lp, dim=1)

        ###################################### norm all scores,
        #idx = target_u_index[scores > self.args.thr]
        soft_label_uniform_fc = soft_label_uniform_fc.cpu()

        if not self.args.noprogressive:
            percent = self.iters / (self.args.max_iters * 0.9)  ### the last 0.1 * epoch train with all labeled data.
            if percent > 1.0:
                percent = 1.0
            num_unl = hard_label_fc.size(0)
            selected_num = int(percent * num_unl)
            if self.args.filter_type == 'fc':
                scores_for_prediction = soft_label_uniform_fc
                scores, hard_label_prediction = torch.max(scores_for_prediction, dim=1)
            elif self.args.filter_type == 'cluster':
                scores_for_prediction = soft_label_uniform_kmean
                scores, hard_label_prediction = torch.max(scores_for_prediction, dim=1)
            elif self.args.filter_type == 'lp':
                scores_for_prediction = soft_label_uniform_lp
                scores, hard_label_prediction = torch.max(scores_for_prediction, dim=1)
            elif self.args.filter_type == 'all':
                scores_for_prediction = (soft_label_uniform_fc + soft_label_uniform_kmean + soft_label_uniform_lp) / 3
                scores, hard_label_prediction = torch.max(scores_for_prediction, dim=1)

            if self.args.entropy_filter:  #### adopt the entropy of sample
                scores = Categorical(probs=scores_for_prediction).entropy()    ### N dimension vector.

            if self.args.category_rank:  ### select samples of each category with TOP percent scores
                raise NotImplementedError  ### the code to be checked before usage.
                index_category = torch.BoolTensor(num_unl).fill_(False)
                for i in range(self.args.num_class):
                    category_per_index = hard_label_prediction == i
                    scores_of_per_category = scores[category_per_index]
                    num_in_category = scores_of_per_category.size(0)
                    selected_num_in_category = int(percent * num_in_category)
                    if selected_num_in_category == 0:
                        print('no samples have been selected for category %d' % (i))
                        break
                    value, _ = torch.topk(scores_of_per_category, selected_num_in_category)
                    thr = value[-1] - 1e-8
                    idx_in_category = (scores > thr) & category_per_index
                    index_category = index_category | idx_in_category
                idx = target_u_index[index_category]

            else:  ### select samples with TOP percent scores
                if selected_num == 0:
                    index_all = torch.BoolTensor(num_unl).fill_(False)
                    idx = target_u_index[index_all]
                    acc_pseudo_selected = 0
                else:
                    if self.args.entropy_filter:  #### adopt the entropy of sample
                        value, _ = torch.topk(scores, selected_num, largest=False)
                        threshold_manully = (1 - self.args.thr) * math.log(self.args.num_class)
                        thr = min(value[-1], threshold_manully)
                        idx = target_u_index[(scores <= thr)]
                        acc_pseudo_selected = accuracy(scores_for_prediction[(scores <= thr)],
                                                       target_u_gt_label_for_visual[(scores <= thr)])
                    else:
                        value, _ = torch.topk(scores, selected_num)
                        threshold_manully = self.args.thr
                        thr = max(value[-1], threshold_manully)  #### filter samples with low prediction.
                        print('the threshold is: %3f' % (thr))
                        idx = target_u_index[(scores >= thr)]
                        if len(idx) == 0:
                            acc_pseudo_selected = 0
                        else:
                            acc_pseudo_selected = accuracy(scores_for_prediction[(scores >= thr)].cuda(),
                                                           target_u_gt_label_for_visual[(scores >= thr)])

            ## construct the list with the global index of [0,1, .....]
            reverse_index = torch.LongTensor(len(target_u_index)).fill_(0)
            for normal_index in range(len(target_u_index)):
                reverse_index[target_u_index[normal_index]] = normal_index

            self.all_hard_pseudo_label = hard_label_prediction[reverse_index].cuda()
            self.all_soft_pseudo_label = scores_for_prediction[reverse_index].cuda()
        else:
            raise NotImplementedError
            num_unl = hard_label_fc.size(0)
            if self.args.filter_type == 'fc':
                idx = target_u_index[(scores_fc.cpu() > self.args.thr)]
            elif self.args.filter_type == 'cluster':
                idx = target_u_index[
                    (scores_fc.cpu() > self.args.thr) & (hard_label_uniform_fc.cpu() == hard_label_uniform_kmean)]
            elif self.args.filter_type == 'ssl':
                idx = target_u_index[
                    (scores_fc.cpu() > self.args.thr) & (hard_label_uniform_fc.cpu() == hard_label_uniform_lp)]
            elif self.args.filter_type == 'both':
                idx = target_u_index[
                    (scores_fc.cpu() > self.args.thr) & (hard_label_uniform_fc.cpu() == hard_label_uniform_lp) & (
                                hard_label_uniform_fc.cpu() == hard_label_uniform_kmean)]
            # elif self.args.filter_type == 'either':
            #     idx = target_u_index[
            #         ((scores_fc.cpu() > self.args.thr) & (hard_label_uniform_fc.cpu() == hard_label_uniform_lp)) | (
            #                     (scores_fc.cpu() > self.args.thr) & (
            #                         hard_label_uniform_fc.cpu() == hard_label_uniform_kmean))]
            if len(idx) == 0:
                acc_pseudo_selected = 0
            else:
                acc_pseudo_selected = accuracy(soft_label_uniform_fc[(scores_fc >= self.args.thr)].cuda(),
                                               target_u_gt_label_for_visual[(scores_fc >= self.args.thr)])
            ## construct the list with the global index of [0,1, .....]
            reverse_index = torch.LongTensor(len(target_u_index)).fill_(0)
            for normal_index in range(len(target_u_index)):
                reverse_index[target_u_index[normal_index]] = normal_index

            self.all_hard_pseudo_label = hard_label_uniform_fc[reverse_index].cuda()
            self.all_soft_pseudo_label = soft_label_uniform_fc[reverse_index].cuda()

        acc_pseudo_label = accuracy(self.all_soft_pseudo_label, target_u_gt_label_for_visual[reverse_index])
        target_u_gt_label_for_visual = target_u_gt_label_for_visual[reverse_index]
        print('Select number: [%d/%d], acc: %3f, acc_seltect: %3f ' % (idx.size(0), num_unl, acc_pseudo_label, acc_pseudo_selected))
        log = open(os.path.join(self.args.save_dir, 'log.txt'), 'a')
        log.write("\n")
        log.write('Select number: [%d/%d], acc: %3f, acc_seltect: %3f ' % (idx.size(0), num_unl, acc_pseudo_label, acc_pseudo_selected))
        log.close()

        #### calculate the target category center with the self.all_hard_pseudo_label
        target_u_feature_matrix = target_u_feature_matrix[reverse_index]  #### re-rank the feature to match the self.all_hard_pseudo_label
        target_feature_category_list = []
        for i in range(self.args.num_class):
            target_feature_category_list.append([])
        for i in range(target_u_feature_matrix.size(0)):
            pseudo_label = self.all_hard_pseudo_label[i]
            target_feature_category_list[pseudo_label].append(target_u_feature_matrix[i].view(1, target_u_feature_matrix.size(1)))

        for i in range(self.args.num_class):
            if len(target_feature_category_list[i]) >0:
                target_feature_category_list[i] = torch.cat(target_feature_category_list[i], dim=0)
                target_feature_category_list[i] = target_feature_category_list[i].mean(0).view(1, target_u_feature_matrix.size(1))   #### get the mean of each target category
            else:
                ## if no target samples is selected for category i
                target_feature_category_list[i] = self.previous_target_feature_category_matrix[i].view(1, target_u_feature_matrix.size(1))
        target_feature_category_matrix = torch.cat(target_feature_category_list, dim=0)

        self.previous_target_feature_category_matrix = target_feature_category_matrix

        ## predicting the source category with the target category mean, and remove the source data with smaller scores [they make the prototypical classifier confusing.]
        ## 1. re-order the source data to the default index (0,1,2,3....)
        ## construct the list with the global index of [0,1, .....]
        reverse_index_source = torch.LongTensor(len(source_index)).fill_(0)
        for normal_index in range(len(source_index)):
            reverse_index_source[source_index[normal_index]] = normal_index
        source_feature_matrix = source_feature_matrix[reverse_index_source]
        hard_label_s = hard_label_s[reverse_index_source]

        target_feature_category_matrix_unsq = torch.unsqueeze(target_feature_category_matrix, 0)
        source_feature_matrix_unsq = torch.unsqueeze(source_feature_matrix, 1)
        L2_dis = ((source_feature_matrix_unsq.cpu() - target_feature_category_matrix_unsq.cpu()) ** 2).mean(2)
        soft_label_target_prototypical = torch.softmax(1 + 1.0 / (L2_dis + 1e-8), dim=1).cuda()
        ### get the score on the source GT categories, and use the score to filter source samples with lower probability.
        instance_index = torch.arange(soft_label_target_prototypical.size(0))
        score_source = soft_label_target_prototypical[instance_index, hard_label_s]  #### score for each source sample.

        ## calculate the source soft weight following SRDC
        target_center_mul_source_feature = torch.matmul(source_feature_matrix, target_feature_category_matrix.transpose(0,1))
        target_center_mul_source_feature = target_center_mul_source_feature[instance_index, hard_label_s]
        target_feature_category_matrix_expand_for_source = target_feature_category_matrix[hard_label_s]
        target_center_norm_mul_source_feature_norm = torch.norm(source_feature_matrix, dim=1) * torch.norm(target_feature_category_matrix_expand_for_source, dim=1)
        source_soft_weight = (target_center_mul_source_feature / target_center_norm_mul_source_feature_norm + 1) / 2.0
        self.source_soft_weight = source_soft_weight


        source_percent = 1 - percent
        num_source = soft_label_target_prototypical.size(0)
        selected_num_source = int(source_percent * num_source)
        if selected_num_source == 0:
            index_source_selected = torch.BoolTensor(num_source).fill_(False)
            idx_source = source_index[index_source_selected]
        else:
            value, _ = torch.topk(score_source, selected_num_source)
            threshold_manully = self.args.thr
            thr = max(value[-1], threshold_manully)  #### filter samples with low prediction.
            print('the threshold is: %3f' % (thr))
            idx_source = source_index[(score_source >= thr)]

            print("source selection: %3f, [%d/%d]" % (source_percent, idx_source.size(0), num_source))

        ## construct a protopical network classifier, where the protopicals are calculated by calculated by the mean of the labeled data and pseudo-labeled unlabeled data.

        for i in range(self.args.num_class):
            source_feature_category = source_feature_matrix[hard_label_s == i]
            target_u_feature_category = target_u_feature_matrix[self.all_hard_pseudo_label == i]
            all_feature_category = torch.cat((source_feature_category, target_u_feature_category), dim=0)
            self.proto.data[i] = all_feature_category.mean(0).data.clone()


        self.previous_selected_index = self.selected_index
        self.previsoud_selected_index_source = self.selected_index_source
        self.hard_label_s = hard_label_s.cuda()
        self.hard_label_t = target_u_gt_label_for_visual.cuda()

        if self.args.target_selection == 'ours':
            self.selected_index = idx.numpy().tolist()  ### the threshold pseudo label
            self.selected_index_source = idx_source.numpy().tolist()  ### the threshold pseudo label
        # elif self.args.target_selection == 'gt':
        #     self.selected_index = list(range(selected_num))
        elif self.args.target_selection == 'rand':
            all_list = list(range(num_unl))
            self.selected_index = random.sample(all_list, selected_num)
            all_list = list(range(num_source))
            self.selected_index_source = random.sample(all_list, selected_num_source)  ### the threshold pseudo label
        else:
            raise NotImplementedError

        ##### the sample distribution in different intermediate domains.
        # if self.epoch != 0:
        #     self.calculate_wasserstein_infinity_dis_of_consecutive_domains(source_feature_matrix, target_u_feature_matrix)
        #     self.calculate_a_dis_of_consecutive_domains(source_feature_matrix, target_u_feature_matrix)

        # self.selected_index = idx.numpy().tolist()  ### the threshold pseudo label
        # self.selected_index_source = idx_source.numpy().tolist()  ### the threshold pseudo label

    def calculate_a_dis_of_consecutive_domains(self, source_feature_matrix, target_u_feature_matrix):
        num_for_dis = 2500
        num_source = source_feature_matrix.size(0)
        num_target = target_u_feature_matrix.size(0)
        indices_source = torch.randperm(num_source)
        indices_target = torch.randperm(num_target)
        source_feature_matrix_sampled = source_feature_matrix[indices_source][:2500]
        target_u_feature_matrix_sampled = target_u_feature_matrix[indices_target][:2500]
        a_dis_st = proxy_a_distance(source_feature_matrix_sampled.cpu().numpy(), target_u_feature_matrix_sampled.cpu().numpy(),
                                 verbose=True)

        previous_domain = torch.cat((source_feature_matrix[self.previsoud_selected_index_source], target_u_feature_matrix[self.previous_selected_index]), dim=0)
        current_domain = torch.cat(
            (source_feature_matrix[self.selected_index_source], target_u_feature_matrix[self.selected_index]),
            dim=0)
        num_previous = previous_domain.size(0)
        num_current = current_domain.size(0)
        indices_previous = torch.randperm(num_previous)
        indices_current = torch.randperm(num_current)
        previous_feature_matrix_sampled = previous_domain[indices_previous][:2500]
        current_feature_matrix_sampled = current_domain[indices_current][:2500]
        a_dis_pc = proxy_a_distance(previous_feature_matrix_sampled.cpu().numpy(), current_feature_matrix_sampled.cpu().numpy(),
                                 verbose=True)
        log = open(os.path.join(self.args.save_dir, 'log.txt'), 'a')
        log.write("\n")
        log.write('A-distance ST: %3f, consecutive: %3f' % (a_dis_st, a_dis_pc))
        log.close()

    def calculate_wasserstein_infinity_dis_of_consecutive_domains(self, source_feature_matrix, target_u_feature_matrix):
        # self.hard_label_s = hard_label_s
        # self.hard_label_t = target_u_gt_label_for_visual
        # self.previsoud_selected_index_source
        # self.previous_selected_index
        wasser_dis_st = []
        wasser_dis_pc = []
        wasser_dis_same_domain_s = []
        wasser_dis_same_domain_t = []
        for i in range(self.args.num_class):
            s_cate_i = source_feature_matrix[self.hard_label_s == i]
            t_cate_i = target_u_feature_matrix[self.hard_label_t == i]
            num_source = s_cate_i.size(0)
            num_target = t_cate_i.size(0)
            indices_previous = torch.randperm(num_source)
            indices_current = torch.randperm(num_target)
            s_cate_i = s_cate_i[indices_previous]
            t_cate_i = t_cate_i[indices_current]
            t_cate_i_half = int(num_target /2.0)
            s_cate_i_half = int(num_source / 2.0)
            wasserstein_infinity = wasserstein_infinity_calculation(s_cate_i.cpu().numpy(), t_cate_i.cpu().numpy())

            wasserstein_infinity_same_t = wasserstein_infinity_calculation(t_cate_i[:t_cate_i_half].cpu().numpy(), t_cate_i[t_cate_i_half:].cpu().numpy())
            wasser_dis_same_domain_t.append(wasserstein_infinity_same_t)

            wasserstein_infinity_same_s = wasserstein_infinity_calculation(s_cate_i[:s_cate_i_half].cpu().numpy(), s_cate_i[s_cate_i_half:].cpu().numpy())
            wasser_dis_same_domain_s.append(wasserstein_infinity_same_s)

            wasser_dis_st.append(wasserstein_infinity)
        wasser_dis_st = max(wasser_dis_st)
        wasser_dis_same_domain_t = max(wasser_dis_same_domain_t)
        wasser_dis_same_domain_s = max(wasser_dis_same_domain_s)

        ########
        previous_domain = torch.cat((source_feature_matrix[self.previsoud_selected_index_source], target_u_feature_matrix[self.previous_selected_index]), dim=0)
        current_domain = torch.cat(
            (source_feature_matrix[self.selected_index_source], target_u_feature_matrix[self.selected_index]), dim=0)
        label_for_previous = torch.cat((self.hard_label_s[self.previsoud_selected_index_source], self.hard_label_t[self.previous_selected_index]), dim=0)
        label_for_current = torch.cat((self.hard_label_s[self.selected_index_source], self.hard_label_t[self.selected_index]), dim=0)
        for i in range(self.args.num_class):
            s_cate_i = previous_domain[label_for_previous == i]
            t_cate_i = current_domain[label_for_current == i]
            num_source = s_cate_i.size(0)
            num_target = t_cate_i.size(0)
            indices_previous = torch.randperm(num_source)
            indices_current = torch.randperm(num_target)
            s_cate_i = s_cate_i[indices_previous]
            t_cate_i = t_cate_i[indices_current]
            wasserstein_infinity = wasserstein_infinity_calculation(s_cate_i.cpu().numpy(), t_cate_i.cpu().numpy())
            wasser_dis_pc.append(wasserstein_infinity)
        wasser_dis_pc = max(wasser_dis_pc)
        log = open(os.path.join(self.args.save_dir, 'log.txt'), 'a')
        log.write("\n")
        log.write('wasser-distance ST: %3f, consecutive: %3f, intra_s: %3f, intra_t: %3f' % (wasser_dis_st, wasser_dis_pc, wasser_dis_same_domain_s, wasser_dis_same_domain_t))
        log.close()


    def get_pseudo_labels(self, index_batch):
        index_list = index_batch.numpy().tolist()
        valid_u = []
        for index, element in enumerate(index_list):
            if element in self.selected_index:
                valid_u.append(True)
            else:
                valid_u.append(False)
        selected_index_in_all_data = index_batch[valid_u]
        label_u_hard = self.all_hard_pseudo_label[selected_index_in_all_data.numpy().tolist()]
        label_u_soft = self.all_soft_pseudo_label[selected_index_in_all_data.numpy().tolist()]


        return label_u_hard.cuda(), label_u_soft.cuda(), valid_u

    def get_source_index(self, index_source):
        index_source = index_source.numpy().tolist()
        valid_u = []
        for index, element in enumerate(index_source):
            if element in self.selected_index_source:
                valid_u.append(True)
            else:
                valid_u.append(False)
        return valid_u

    def get_source_soft_weight(self, index_source):
        return self.source_soft_weight[index_source]

    def solve(self):
        stop = False
        counter=0
        best_prec1_val = 0
        self.pre_train_classifier()
        self.iters = 0  ### initial the training iteration
        if self.args.initial_only:
            self.Get_pseudo_labels_with_classifiers_consistency()
            return 0
        while not stop:
            #### adopted in the overall weight setting.
            self.weight_source = self.iters / (self.args.max_iters * 0.9)
            if self.weight_source >= 1:
                self.weight_source = 0
            else:
                self.weight_source = 1 - self.weight_source    #### the reverse percent of the target data
            stop = self.complete_training()
            self.Get_pseudo_labels_with_classifiers_consistency()
            self.update_network()
            acc, acc_val = self.test()

            if acc > self.best_prec1:
                self.best_prec1 = acc
                log = open(os.path.join(self.args.save_dir, 'log.txt'), 'a')
                log.write("          Best acc by far:%3f" % \
                          (acc))
                log.close()
            #     self.save_ckpt(True)
            # else:
            #     self.save_ckpt(False)


            if acc_val > best_prec1_val:
                best_prec1_val = acc_val
                counter = 0
            else:
                counter += 1
            if self.args.early:
                if counter > self.args.patience:
                    break

            self.epoch += 1

    def update_network(self, **kwargs):

        stop = False
        self.G.train()
        self.F.train()
        self.train_data['source']['iterator'] = iter(self.train_data['source']['loader'])
        self.train_data['target']['iterator'] = iter(self.train_data['target']['loader'])
        if self.args.dataset == 'visDA':
            self.iters_per_epoch = len(self.train_data['source']['loader']) / 2
        else:
            self.iters_per_epoch = len(self.train_data['source']['loader'])
        print('iters in each epoch is: %d' % (self.iters_per_epoch))
        iters_counter_within_epoch = 0
        data_time = AverageMeter()
        batch_time = AverageMeter()
        losses_all = AverageMeter()
        losses_s = AverageMeter()
        losses_t = AverageMeter()
        end = time.time()
        while not stop:
            # lam = 2 / (1 + math.exp(-1 * 10 * self.iters / (self.opt.TRAIN.MAX_EPOCH * self.iters_per_epoch))) - 1
            # print('value of lam is: %3f' % (lam))
            self.update_lr()

            source_data, _, source_gt, index_source, _ = self.get_samples('source')
            target_data_u, target_data_u_strong, _, index, target_path = self.get_samples('target')

            ########################
            source_data = to_cuda(source_data)
            source_gt = to_cuda(source_gt)
            target_data_u = to_cuda(target_data_u)
            target_data_u_strong = to_cuda(target_data_u_strong)
            data_time.update(time.time() - end)
            #label_u, valid_u = self.lb_guessor(self.G, self.F, target_data_u)
            label_u_hard, label_u_soft, valid_u = self.get_pseudo_labels(index)
            target_data_u_strong = target_data_u_strong[valid_u]
            n_u = target_data_u_strong.size(0)

            if n_u != 0:
                num_labeled = source_data.size(0)
                data = torch.cat([source_data, target_data_u_strong], dim=0)
                label = source_gt
                feature = self.G(data)
                # feature_not_use = self.G(target_data_u)  ### update the BN with target data
                logit = self.F(feature)
                logit_labeled, logit_unl = logit[:num_labeled], logit[num_labeled:]
                feature_labeled, feature_unl = feature[:num_labeled, :], feature[num_labeled:, :]

                if self.args.source_weight == 'none':
                    loss_labeled = self.CELoss(logit_labeled, label).mean()
                elif self.args.source_weight == 'overall':
                    loss_labeled = self.CELoss(logit_labeled, label).mean() * self.weight_source
                elif self.args.source_weight == 'instance':
                    ## which index in index_source is also in self.selected_index_source
                    selected_index = self.get_source_index(index_source)
                    print('source selection: %d/%d' % (label[selected_index].size(0), index_source.size(0)))
                    if len(label[selected_index]) == 0:
                        loss_labeled = 0
                    else:
                        loss_labeled = self.CELoss(logit_labeled[selected_index], label[selected_index]).sum() / source_data.size(0)
                elif self.args.source_weight == 'soft':
                    soft_weight = self.get_source_soft_weight(index_source)
                    loss_labeled = (self.CELoss(logit_labeled, label) * soft_weight).mean()
                else:
                    raise NotImplementedError


                if self.args.pseudo_type == 'hard':
                    loss_unl = self.CELoss(logit_unl, label_u_hard).sum() / target_data_u.size(0)
                elif self.args.pseudo_type == 'w_hard':
                    entropy = Categorical(probs=label_u_soft).entropy()
                    weight = 1 - entropy / math.log(self.args.num_class)
                    loss_unl = (self.CELoss(logit_unl, label_u_hard) * weight).sum() / target_data_u.size(0)
                elif self.args.pseudo_type == 'soft':
                    loss_unl = - (label_u_soft * F.log_softmax(logit_unl, dim=1)).sum(1).sum() / target_data_u.size(0)
                elif self.args.pseudo_type == 'w_soft':
                    entropy = Categorical(probs=label_u_soft).entropy()
                    weight = 1 - entropy / math.log(self.args.num_class)
                    loss_unl = - ((label_u_soft * F.log_softmax(logit_unl, dim=1)).sum(1) * weight).sum() / target_data_u.size(0)
                else:
                    raise NotImplementedError
                loss = loss_labeled + loss_unl * self.args.weight
                if self.args.prototype_classifier:
                    loss_prototype_labeled = self.return_labeled_prototype_loss(feature_labeled, label, self.proto)
                    loss_prototype_unl = self.return_unlabeled_prototype_loss(feature_unl, label_u_hard, label_u_soft, self.proto, target_data_u.size(0))
                    loss_prototype = loss_prototype_labeled + loss_prototype_unl
                    loss += loss_prototype

            else:
                data = source_data
                label = source_gt
                feature = self.G(data)
                # feature_not_use = self.G(target_data_u)  ### update the BN with target data
                logit_labeled = self.F(feature)
                if self.args.source_weight == 'none':
                    loss = self.CELoss(logit_labeled, label).mean()
                elif self.args.source_weight == 'overall':
                    loss = self.CELoss(logit_labeled, label).mean() * self.weight_source
                elif self.args.source_weight == 'instance':
                    ## which index in index_source is also in self.selected_index_source
                    selected_index = self.get_source_index(index_source)
                    if len(label[selected_index]) == 0:
                        loss = 0
                    else:
                        loss = self.CELoss(logit_labeled[selected_index], label[selected_index]).sum() / source_data.size(0)
                elif self.args.source_weight == 'soft':
                    soft_weight = self.get_source_soft_weight(index_source)
                    loss = (self.CELoss(logit_labeled, label) * soft_weight).mean()
                else:
                    raise NotImplementedError
                # loss = self.CELoss(logit, label).mean()
                if self.args.prototype_classifier:
                    loss_prototype = self.return_labeled_prototype_loss(feature, label, self.proto)
                    loss += loss_prototype

            self.optimizer_G.zero_grad()
            self.optimizer_F.zero_grad()
            loss.backward()
            self.optimizer_G.step()
            self.optimizer_F.step()
            # self.ema.update_params()

            losses_all.update(loss.item(), data.size(0))
            # if n_u != 0:
            #     losses_s.update(loss_labeled.item(), logit_labeled.size(0))
            #     losses_t.update(loss_unl.item(), logit_unl.size(0))

            batch_time.update(time.time() - end)
            end = time.time()
            self.iters += 1
            iters_counter_within_epoch += 1
            if self.iters % 10 == 0:
                print("  Train:epoch: %d:[%d/%d], Tdata: %3f, Tbatch: %3f, LossL: %3f, LossU: %3f, LossAll:%3f, Select[%d/%d]" % \
                      (self.epoch, self.iters, self.args.max_iters, data_time.avg, batch_time.avg, losses_s.avg, losses_t.avg, losses_all.avg, n_u, target_data_u.size(0)))

            if iters_counter_within_epoch >= self.iters_per_epoch:
                log = open(os.path.join(self.args.save_dir, 'log.txt'), 'a')
                log.write("\n")
                log.write("  Train:epoch: %d:[%d/%d], Tdata: %3f, Tbatch: %3f, LossL: %3f, LossU: %3f, LossAll:%3f, Select[%d/%d]" % \
                  (self.epoch, self.iters, self.args.max_iters, data_time.avg, batch_time.avg, losses_s.avg, losses_t.avg, losses_all.avg, n_u, target_data_u.size(0)))
                log.close()
                stop = True
                self.ema.update_buffer()

    def return_labeled_prototype_loss(self, feature_labeled, label, proto):
        prob_pred = (1 + (feature_labeled.unsqueeze(1) - proto.unsqueeze(0)).pow(2).sum(2)).pow(- 1)
        loss = self.CELoss(prob_pred, label).mean()
        return loss

    def return_unlabeled_prototype_loss(self, feature_unl, label_u_hard, label_u_soft, proto, num_all):
        prob_pred = (1 + (feature_unl.unsqueeze(1) - proto.unsqueeze(0)).pow(2).sum(2)).pow(- 1)
        if self.args.pseudo_type == 'hard':
            loss_unl = self.CELoss(prob_pred, label_u_hard).sum() / num_all
        elif self.args.pseudo_type == 'w_hard':
            entropy = Categorical(probs=label_u_soft).entropy()
            weight = 1 - entropy / math.log(self.args.num_class)
            loss_unl = (self.CELoss(prob_pred, label_u_hard) * weight).sum() / num_all
        elif self.args.pseudo_type == 'soft':
            loss_unl = - (label_u_soft * F.log_softmax(prob_pred, dim=1)).sum(1).sum() / num_all
        elif self.args.pseudo_type == 'w_soft':
            entropy = Categorical(probs=label_u_soft).entropy()
            weight = 1 - entropy / math.log(self.args.num_class)
            loss_unl = - ((label_u_soft * F.log_softmax(prob_pred, dim=1)).sum(1) * weight).sum() / num_all
        else:
            raise NotImplementedError
        return loss_unl


    def test(self):
        ##
        self.G.eval()
        self.F.eval()
        if self.args.category_mean:
            counter_all_ft = torch.FloatTensor(self.args.num_class).fill_(0)
            counter_acc_ft = torch.FloatTensor(self.args.num_class).fill_(0)
        prec1 = AverageMeter()
        for i, (input, target) in enumerate(self.test_data['loader']):
            input, target = to_cuda(input), to_cuda(target)
            with torch.no_grad():
                feature_test = self.G(input)
                output_test = self.F(feature_test)
            prec1_iter = accuracy(output_test, target)
            prec1.update(prec1_iter, input.size(0))
            if self.args.category_mean:
                counter_all_ft, counter_acc_ft = accuracy_for_each_class(output_test, target,
                                                                         counter_all_ft, counter_acc_ft)

        if self.args.category_mean:
            acc_for_each_class_ft = counter_acc_ft / counter_all_ft
            log = open(os.path.join(self.args.save_dir, 'log.txt'), 'a')
            log.write("\nClass-wise Acc of Ft:")  ## based on the task classifier.
            for i in range(self.args.num_class):
                if i == 0:
                    log.write("%dst: %3f" % (i + 1, acc_for_each_class_ft[i]))
                elif i == 1:
                    log.write(",  %dnd: %3f" % (i + 1, acc_for_each_class_ft[i]))
                elif i == 2:
                    log.write(", %drd: %3f" % (i + 1, acc_for_each_class_ft[i]))
                else:
                    log.write(", %dth: %3f" % (i + 1, acc_for_each_class_ft[i]))
            log.close()
        # self.ema.apply_shadow()
        # self.ema.G.eval()
        # self.ema.F.eval()
        # self.ema.G.cuda()
        # self.ema.F.cuda()


        prec1_ema = AverageMeter()

        # for i, (input, target) in enumerate(self.test_data['loader']):
        #     input, target = to_cuda(input), to_cuda(target)
        #     with torch.no_grad():
        #         feature_test = self.ema.G(input)
        #         output_test = self.ema.F(feature_test)
        #     prec1_iter = accuracy(output_test, target)
        #     prec1_ema.update(prec1_iter, input.size(0))
        #
        # print("                       Test:epoch: %d, iter: %d, Acc: %3f, ema_ACC: %3f" % \
        #     (self.epoch, self.iters, prec1.avg, prec1_ema.avg))
        log = open(os.path.join(self.args.save_dir, 'log.txt'), 'a')
        log.write("\n")
        log.write("                                                                                 Test:epoch: %d, iter: %d, Acc: %3f, ema_Acc: %3f" % \
            (self.epoch, self.iters, prec1.avg, prec1_ema.avg))
        log.close()
        # self.ema.restore()
        if self.args.category_mean:
            acc_for_each_class_ft = counter_acc_ft / counter_all_ft
            log = open(os.path.join(self.args.save_dir, 'log.txt'), 'a')
            log.write("\nClass-wise Acc of Ft:")  ## based on the task classifier.
            for i in range(self.args.num_class):
                if i == 0:
                    log.write("%dst: %3f" % (i + 1, acc_for_each_class_ft[i]))
                elif i == 1:
                    log.write(",  %dnd: %3f" % (i + 1, acc_for_each_class_ft[i]))
                elif i == 2:
                    log.write(", %drd: %3f" % (i + 1, acc_for_each_class_ft[i]))
                else:
                    log.write(", %dth: %3f" % (i + 1, acc_for_each_class_ft[i]))
            log.close()
            return max(acc_for_each_class_ft.mean(), prec1_ema.avg), max(acc_for_each_class_ft.mean(), prec1_ema.avg)
        else:
            return max(prec1.avg, prec1_ema.avg), max(prec1.avg, prec1_ema.avg)

    def build_optimizer(self):
        if self.args.optimizer == 'SGD':  ## some params may not contribute the loss_all, thus they are not updated in the training process.
            if self.args.fix_submodel:
                self.optimizer_G = torch.optim.SGD([
                    {'params': self.G.module.conv1.parameters(), 'name': 'fixed'},
                    {'params': self.G.module.bn1.parameters(), 'name': 'fixed'},
                    {'params': self.G.module.layer1.parameters(), 'name': 'fixed'},
                    {'params': self.G.module.layer2.parameters(), 'name': 'pre-trained'},
                    {'params': self.G.module.layer3.parameters(), 'name': 'pre-trained'},
                    {'params': self.G.module.layer4.parameters(), 'name': 'pre-trained'},
                ],
                    lr=self.args.base_lr * 0.1,
                    momentum=self.args.momentum,
                    weight_decay=self.args.wd,   ## note the wd is implemented in the ema_optimizer
                    nesterov=True)
                self.G.module.bn1.apply(fix_bn)
                self.G.module.layer1.apply(fix_bn)
            else:
                self.optimizer_G = torch.optim.SGD([
                    {'params': self.G.parameters(), 'name': 'pre-trained'},
                ],
                    lr=self.args.base_lr * 0.1,
                    momentum=self.args.momentum,
                    weight_decay=self.args.wd,   ## note the wd is implemented in the ema_optimizer
                    nesterov=True)

            if self.args.prototype_classifier:
                raise NotImplementedError
                self.optimizer_F = torch.optim.SGD([
                    {'params': self.F.parameters(), 'name': 'new-added'},
                    {'params': self.proto, 'name': 'pre-trained'},
                ],
                    lr=self.args.base_lr,
                    momentum=self.args.momentum,
                    weight_decay=self.args.wd,  ## ## note the wd is implemented in the ema_optimizer
                    nesterov=True)
            else:
                self.optimizer_F = torch.optim.SGD([
                    {'params': self.F.parameters(), 'name': 'new-added'},
                ],
                    lr=self.args.base_lr,
                    momentum=self.args.momentum,
                    weight_decay=self.args.wd,  ## ## note the wd is implemented in the ema_optimizer
                    nesterov=True)
        else:
            raise NotImplementedError
        print('Optimizer built')

    def update_lr(self, given_lr=0.0):
        if given_lr == 0.0:
            if self.args.lr_schedule == 'inv':
                lr = self.args.base_lr / pow((1 + self.args.inv_alpha * self.iters / (self.args.max_iters)), self.args.inv_beta)
            elif self.args.lr_schedule == 'fix':
                lr = self.args.base_lr
            else:
                raise NotImplementedError
        else:
            lr = given_lr
        lr_pretrain = lr * 0.1
        print('the lr is: %3f' % (lr_pretrain))
        for param_group in self.optimizer_G.param_groups:
            if param_group['name'] == 'pre-trained':
                param_group['lr'] = lr_pretrain
            elif param_group['name'] == 'new-added':
                param_group['lr'] = lr
            elif param_group['name'] == 'fixed': ## Fix the lr as 0 can not fix the runing mean/var of the BN layer
                param_group['lr'] = 0

        for param_group in self.optimizer_F.param_groups:
            if param_group['name'] == 'pre-trained':
                param_group['lr'] = lr_pretrain
            elif param_group['name'] == 'new-added':
                param_group['lr'] = lr
            elif param_group['name'] == 'fixed': ## Fix the lr as 0 can not fix the runing mean/var of the BN layer
                param_group['lr'] = 0

    def save_ckpt(self, best):
        log = open(os.path.join(self.args.save_dir, 'log.txt'), 'a')
        log.write("      Best Acc so far: %3f" % (self.best_prec1))
        log.close()
        if self.args.saving:
            save_path = self.args.save_dir
            ckpt_resume = os.path.join(save_path, 'last.resume')
            torch.save({'iters': self.iters,
                        'best_prec1': self.best_prec1,
                        'G_state_dict': self.ema.G.state_dict(),
                        'F_state_dict': self.ema.F.state_dict()
                        }, ckpt_resume)
            if best:
                ckpt_resume = os.path.join(save_path, 'best.resume')
                torch.save({'iters': self.iters,
                            'best_prec1': self.best_prec1,
                            'G_state_dict': self.ema.G.state_dict(),
                            'F_state_dict': self.ema.F.state_dict()
                            }, ckpt_resume)