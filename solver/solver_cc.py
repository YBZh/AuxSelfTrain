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
import random

class Solver(BaseSolver):
    def __init__(self, G, F, dataloaders, args, **kwargs):
        super(Solver, self).__init__(G, F, dataloaders, args, **kwargs)

        self.ema = EMA_fixmatch(G, F, args.ema_decay)
        self.lb_guessor = LabelGuessor(thresh=args.thr)
        self.CELoss = nn.CrossEntropyLoss(reduction='none')
        from data.prepare_data_fixmatch import generate_dataloader_mmd as Dataloader
        dataloaders_mmd = Dataloader(args)
        self.init_data_mmd(dataloaders_mmd)


        if args.resume != '':
            resume_dict = torch.load(args.resume)
            self.G.load_state_dict(resume_dict['G_state_dict'])
            self.F.load_state_dict(resume_dict['F_state_dict'])
            self.best_prec1 = resume_dict['best_prec1']
            self.iter = resume_dict['iter']

    def init_data_mmd(self, dataloaders):
        self.mmd_data = {key: dict() for key in dataloaders}
        for key in self.mmd_data.keys():
            if key not in dataloaders:
                continue
            cur_dataloader = dataloaders[key]
            self.mmd_data[key]['loader'] = cur_dataloader

    def pre_train_classifier(self):
        # if not self.args.pre_trained_G:
        #     print('fix the running mean and var for BN')
        #     self.G.apply(fix_bn)
        #     self.F.apply(fix_bn)
        initial_lr = 0.01
        for i in range(self.args.pre_epoch):
            new_lr = initial_lr #/ (10 ** i)
            print('new lr for classifier training is: %3f' % (new_lr))
            self.G.train()
            self.F.train()
            self.train_data['source']['iterator'] = iter(self.train_data['source']['loader'])
            self.train_data['target_l']['iterator'] = iter(self.train_data['target_l']['loader'])
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
                source_data, source_data_strong, source_gt, _, _ = self.get_samples('source')
                target_data_l, target_data_l_strong, target_gt_l = self.get_samples('target_l')

                # if self.args.weak_aug:
                source_data_train = to_cuda(source_data)
                target_data_l_train = to_cuda(target_data_l)
                # else:
                #     source_data_train = to_cuda(source_data_strong)
                #     target_data_l_train = to_cuda(target_data_l_strong)
                source_gt = to_cuda(source_gt)
                target_gt_l = to_cuda(target_gt_l)
                data_time.update(time.time() - end)
                data = torch.cat([source_data_train, target_data_l_train], dim=0)
                label = torch.cat([source_gt, target_gt_l], dim=0)
                logit = self.F(self.G(data))
                loss = self.CELoss(logit, label).mean()

                self.optimizer_G.zero_grad()
                self.optimizer_F.zero_grad()
                loss.backward()
                if self.args.pre_trained_G:
                    self.optimizer_G.step()
                self.optimizer_F.step()
                self.ema.update_params()

                losses_all.update(loss.item(), data.size(0))

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
        # target_u_path_list = ()
        print('prepare feature of target unlabeled data')
        for i, (input, target_for_visual, index, path) in enumerate(self.mmd_data['target_u']['loader']):
            if (i+1) % 100 == 0:
                print("%d / %d" % (i, len(self.mmd_data['target_u']['loader'])))
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
            # target_u_path_list+=path
            target_u_label_list.append(target_for_visual)

        target_u_feature_matrix = torch.cat(target_u_feature_list, dim=0)
        target_u_prediction_matrix = torch.cat(target_u_prediction_list, dim=0)
        target_u_index = torch.cat(target_u_index_list, dim=0)
        target_u_gt_label_for_visual = torch.cat(target_u_label_list)



        target_l_feature_list = []
        target_l_label_list = []
        target_l_cate_feature_list = []
        for i in range(self.args.num_class):
            target_l_cate_feature_list.append([])
        print('prepare features of target labeled data')
        for i, (input, target) in enumerate(self.mmd_data['target_l']['loader']):

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
                target_l_feature_iter = self.G(input)
            if self.args.feat_type_pseudo == 'train':
                self.G.load_state_dict(org_state_G)
                self.F.load_state_dict(org_state_F)
            target_l_feature_list.append(target_l_feature_iter)
            target_l_label_list.append(target)
            for j in range(input.size(0)):
                img_label = target[j]
                target_l_cate_feature_list[img_label].append(target_l_feature_iter[j].view(1, target_l_feature_iter.size(1)))
        target_l_feature_matrix = torch.cat(target_l_feature_list, dim=0)
        hard_label_tl = torch.cat(target_l_label_list, dim=0)
        soft_label_tl = to_onehot(hard_label_tl, self.args.num_class)

        source_feature_list = []
        source_label_list = []
        source_cate_feature_list = []
        source_index_list = []
        for i in range(self.args.num_class):
            source_cate_feature_list.append([])
        print('prepare features of source data')

        debug_flag = False
        for i, (input, target, index, _) in enumerate(self.mmd_data['source']['loader']):
            if (i+1) % 100 == 0:
                print("%d / %d" % (i, len(self.mmd_data['source']['loader'])))
            # input = copy.deepcopy(input)
            # del input
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
        soft_label_normal_fc, soft_label_uniform_fc, hard_label_normal_fc, hard_label_uniform_fc, acc_fc, acc_uniform_fc = get_labels_from_classifier_prediction(target_u_prediction_matrix, self.args.T, target_u_gt_label_for_visual)
        log = open(os.path.join(self.args.save_dir, 'log.txt'), 'a')
        log.write("\n")
        log.write('FC Prediction without and with uniform prior are: %3f and %3f' % (acc_fc, acc_uniform_fc))
        log.close()
        if self.args.no_uniform_prior:
            soft_label_fc = soft_label_normal_fc
            scores_fc, hard_label_fc = torch.max(soft_label_normal_fc, dim=1)
        else:
            soft_label_fc = soft_label_uniform_fc
            scores_fc, hard_label_fc = torch.max(soft_label_uniform_fc, dim=1)


        if self.args.filter_type == 'cluster' or self.args.filter_type == 'cluster_lp' or self.args.filter_type == 'all':
            if self.args.pseudo_label_generator == 'spheticalkmeans':
                for i in range(self.args.num_class):
                    if self.args.init_center == 'tl':
                        target_l_cate_feature_list[i] = torch.cat(target_l_cate_feature_list[i], dim=0)
                        target_l_cate_feature_list[i] = target_l_cate_feature_list[i].mean(0)
                    elif self.args.init_center == 's':
                        target_l_cate_feature_list[i] = torch.cat(source_cate_feature_list[i], dim=0)
                        target_l_cate_feature_list[i] = target_l_cate_feature_list[i].mean(0)
                    elif self.args.init_center == 'st':  ### 最开始有52，看起来这个是最好的。
                        target_l_cate_feature_list[i] = torch.cat(target_l_cate_feature_list[i] , dim=0)
                        source_cate_feature_list[i] = torch.cat(source_cate_feature_list[i] , dim=0)
                        target_l_cate_feature_list[i] = (target_l_cate_feature_list[i].mean(0) + source_cate_feature_list[i].mean(0)) / 2
                    target_l_cate_feature_list[i] = F.normalize(target_l_cate_feature_list[i], dim=0, p=2)
                    target_l_cate_feature_list[i] = target_l_cate_feature_list[i].cpu().numpy()
                target_l_cate_feature_list = np.array(target_l_cate_feature_list)

                target_u_feature_matrix_norm = F.normalize(target_u_feature_matrix, dim=1, p=2)
                target_l_feature_matrix_norm = F.normalize(target_l_feature_matrix, dim=1, p=2)
                soft_label_normal_kmean, soft_label_uniform_kmean, hard_label_normal_kmean, hard_label_uniform_kmean, acc_kmean, acc_uniform_kmean = \
                get_labels_from_Sphericalkmeans(initial_centers_array=target_l_cate_feature_list, target_u_feature=target_u_feature_matrix_norm.cpu(),
                                                num_class=self.args.num_class, gt_label=target_u_gt_label_for_visual.cpu(), T=0.05, max_iter=100, target_l_feature=target_l_feature_matrix_norm.cpu())
                log = open(os.path.join(self.args.save_dir, 'log.txt'), 'a')
                log.write("\n")
                log.write('sphetical Kmeans Prediction without and with uniform prior are: %3f and %3f' % (acc_kmean, acc_uniform_kmean))
                log.close()

            elif self.args.pseudo_label_generator == 'kmeans':
                for i in range(self.args.num_class):
                    if self.args.init_center == 'tl':
                        target_l_cate_feature_list[i] = torch.cat(target_l_cate_feature_list[i], dim=0)
                        target_l_cate_feature_list[i] = target_l_cate_feature_list[i].mean(0)
                    elif self.args.init_center == 's':
                        target_l_cate_feature_list[i] = torch.cat(source_cate_feature_list[i], dim=0)
                        target_l_cate_feature_list[i] = target_l_cate_feature_list[i].mean(0)
                    elif self.args.init_center == 'st':  ### 最开始有52，看起来这个是最好的。
                        target_l_cate_feature_list[i] = torch.cat(target_l_cate_feature_list[i] , dim=0)
                        source_cate_feature_list[i] = torch.cat(source_cate_feature_list[i] , dim=0)
                        target_l_cate_feature_list[i] = (target_l_cate_feature_list[i].mean(0) + source_cate_feature_list[i].mean(0)) / 2
                    #target_l_cate_feature_list[i] = F.normalize(target_l_cate_feature_list[i], dim=0, p=2)
                    target_l_cate_feature_list[i] = target_l_cate_feature_list[i].cpu().numpy()
                target_l_cate_feature_list = np.array(target_l_cate_feature_list)

                # target_u_feature_matrix_norm = F.normalize(target_u_feature_matrix, dim=1, p=2)
                # target_l_feature_matrix_norm = F.normalize(target_l_feature_matrix, dim=1, p=2)
                soft_label_normal_kmean, soft_label_uniform_kmean, hard_label_normal_kmean, hard_label_uniform_kmean, acc_kmean, acc_uniform_kmean = \
                get_labels_from_kmeans(initial_centers_array=target_l_cate_feature_list, target_u_feature=target_u_feature_matrix.cpu(),
                                                num_class=self.args.num_class, gt_label=target_u_gt_label_for_visual.cpu(), T=0.05, max_iter=100, target_l_feature=target_l_feature_matrix.cpu())
                log = open(os.path.join(self.args.save_dir, 'log.txt'), 'a')
                log.write("\n")
                log.write('Kmeans Prediction without and with uniform prior are: %3f and %3f' % (acc_kmean, acc_uniform_kmean))
                log.close()
            else:
                raise NotImplementedError

            if self.args.no_uniform_prior:
                soft_label_kmean = soft_label_normal_kmean
                hard_label_kmean = hard_label_normal_kmean
            else:
                soft_label_kmean = soft_label_uniform_kmean
                hard_label_kmean = hard_label_uniform_kmean

        if self.args.filter_type == 'lp' or self.args.filter_type == 'fc_lp' or self.args.filter_type == 'cluster_lp' or  self.args.filter_type == 'all':
            if self.args.lp_labeled == 's':
                labeled_feature_matrix = source_feature_matrix
                labeled_onehot = soft_label_s
            elif self.args.lp_labeled == 't':
                labeled_feature_matrix = target_l_feature_matrix
                labeled_onehot = soft_label_tl
            elif self.args.lp_labeled == 'st':
                labeled_feature_matrix = torch.cat((source_feature_matrix, target_l_feature_matrix), dim=0)
                labeled_onehot = torch.cat((soft_label_s, soft_label_tl), dim=0)
            else:
                raise NotImplementedError

            soft_label_normal_lp, soft_label_uniform_lp, hard_label_normal_lp, hard_label_uniform_lp, acc_lp, acc_uniform_lp = \
            get_labels_from_lp(labeled_features=labeled_feature_matrix.cpu(), labeled_onehot_gt=labeled_onehot.cpu(),
                               unlabeled_features = target_u_feature_matrix.cpu(), gt_label=target_u_gt_label_for_visual.cpu(),
                               num_class=self.args.num_class, dis=self.args.lp_dis, solver=self.args.lp_solver, graphk = self.args.lp_graphk, alpha=self.args.lp_alpha)
            log = open(os.path.join(self.args.save_dir, 'log.txt'), 'a')
            log.write("\n")
            log.write('LP Prediction without and with uniform prior are: %3f and %3f' % (acc_lp, acc_uniform_lp))
            log.close()

            if self.args.no_uniform_prior:
                soft_label_lp = soft_label_normal_lp
                hard_label_lp = hard_label_normal_lp
            else:
                soft_label_lp = soft_label_uniform_lp
                hard_label_lp = hard_label_uniform_lp

        ###################################### norm all scores,
        #idx = target_u_index[scores > self.args.thr]
        soft_label_fc = soft_label_fc.cpu()


        if not self.args.noprogressive:   #### the self-paced strategy
            percent = self.iters / (self.args.max_iters * 0.9)        ### the last 0.1 * epoch train with all labeled data.
            if percent > 1.0:
                percent = 1.0
            num_unl = hard_label_fc.size(0)
            selected_num = int(percent * num_unl)
            if self.args.filter_type == 'fc':
                scores_for_prediction = soft_label_fc
                scores, hard_label_prediction = torch.max(scores_for_prediction, dim=1)
            elif self.args.filter_type == 'cluster':
                scores_for_prediction = soft_label_kmean
                scores, hard_label_prediction = torch.max(scores_for_prediction, dim=1)
            elif self.args.filter_type == 'lp':
                scores_for_prediction = soft_label_lp
                scores, hard_label_prediction = torch.max(scores_for_prediction, dim=1)
            elif self.args.filter_type == 'fc_lp':
                scores_for_prediction = (soft_label_lp + soft_label_fc) /2
                scores, hard_label_prediction = torch.max(scores_for_prediction, dim=1)
            elif self.args.filter_type == 'cluster_lp':
                scores_for_prediction = (soft_label_lp + soft_label_kmean) / 2
                scores, hard_label_prediction = torch.max(scores_for_prediction, dim=1)
            elif self.args.filter_type == 'all':
                scores_for_prediction = (soft_label_fc + soft_label_kmean + soft_label_lp) / 3
                scores, hard_label_prediction = torch.max(scores_for_prediction, dim=1)
            if self.args.entropy_filter:  #### the entropy term is not adopted.
                scores = Categorical(probs=scores_for_prediction).entropy()    ### N dimension vector.

            if self.args.pace_strategy == 'implicit':
                if self.args.category_rank:  ### select samples of each category with TOP percent scores。 WHY not work??
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
                        thr = value[-1] - 1e-12
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
                            thr = max(value[-1], threshold_manully)   #### filter samples with low prediction.
                            print('the threshold is: %3f' % (thr))
                            idx = target_u_index[(scores >= thr)]
                            if len(idx) == 0:
                                acc_pseudo_selected = 0
                            else:
                                acc_pseudo_selected = accuracy(scores_for_prediction[(scores >= thr)].cuda(), target_u_gt_label_for_visual[(scores >= thr)])
            elif self.args.pace_strategy == 'manual':
                raise NotImplementedError
                threshold = (self.args.max_iters - self.iters) / (self.args.max_iters)
                if threshold > 1.0:
                    threshold = 1.0
                if threshold < 0.0:
                    threshold = 0.0
                if self.args.entropy_filter:  #### adopt the entropy of sample
                    raise NotImplementedError
                else:
                    idx = target_u_index[(scores >= threshold)]
                    if len(idx) == 0:
                        acc_pseudo_selected = 0
                    else:
                        acc_pseudo_selected = accuracy(scores_for_prediction[(scores >= threshold)].cuda(),
                                                       target_u_gt_label_for_visual[(scores >= threshold)])
            else:
                raise NotImplementedError

            ## construct the list with the global index of [0,1, .....]
            reverse_index = torch.LongTensor(len(target_u_index)).fill_(0)
            for normal_index in range(len(target_u_index)):
                reverse_index[target_u_index[normal_index]] = normal_index

            self.all_hard_pseudo_label = hard_label_prediction[reverse_index].cuda()   ##hard_label_prediction.cuda()
            self.all_soft_pseudo_label = scores_for_prediction[reverse_index].cuda()   #scores_for_prediction.cuda()
        else:  #####  with fixed threshold.
            print('similar to fixmatch, but generate pseudo labels per epoch')
            num_unl = hard_label_fc.size(0)
            if self.args.filter_type == 'fc':
                idx = target_u_index[(scores_fc.cpu() > self.args.thr)]
            # elif self.args.filter_type == 'cluster':
            #     idx = target_u_index[
            #         (scores_fc.cpu() > self.args.thr) & (hard_label_fc.cpu() == hard_label_kmean)]
            # elif self.args.filter_type == 'lp':
            #     idx = target_u_index[
            #         (scores_fc.cpu() > self.args.thr) & (hard_label_fc.cpu() == hard_label_lp)]
            # elif self.args.filter_type == 'all':
            #     # idx = target_u_index[
            #     #     (scores_fc.cpu() > self.args.thr) & (hard_label_fc.cpu() == hard_label_lp) & (
            #     #                 hard_label_fc.cpu() == hard_label_kmean)]
            #     scores_for_prediction = (soft_label_fc + soft_label_kmean + soft_label_lp) / 3
            #     scores_fc, hard_label_prediction = torch.max(scores_for_prediction, dim=1)
            #     idx = target_u_index[(scores_fc.cpu() > self.args.thr)]

            if len(idx) == 0:
                acc_pseudo_selected = 0
            else:
                acc_pseudo_selected = accuracy(soft_label_fc[(scores_fc >= self.args.thr)].cuda(),
                                               target_u_gt_label_for_visual[(scores_fc >= self.args.thr)])
            ## construct the list with the global index of [0,1, .....]
            reverse_index = torch.LongTensor(len(target_u_index)).fill_(0)
            for normal_index in range(len(target_u_index)):
                reverse_index[target_u_index[normal_index]] = normal_index
            self.all_hard_pseudo_label = hard_label_fc[reverse_index].cuda()  #  torch.scatter(hard_label_fc, 0, target_u_index, hard_label_fc).cuda()  ##hard_label_fc[target_u_index].cuda()  ###
            self.all_soft_pseudo_label = soft_label_fc[reverse_index].cuda()  #torch.scatter(soft_label_fc, 0, target_u_index, soft_label_fc).cuda()  ##soft_label_fc[target_u_index].cuda()

        acc_pseudo_label = accuracy(self.all_soft_pseudo_label, target_u_gt_label_for_visual[reverse_index.cuda()])

        print('Select number: [%d/%d], acc: %3f, acc_seltect: %3f ' % (idx.size(0), num_unl, acc_pseudo_label, acc_pseudo_selected))
        log = open(os.path.join(self.args.save_dir, 'log.txt'), 'a')
        log.write("\n")
        log.write('Select number: [%d/%d], acc: %3f, acc_seltect: %3f ' % (idx.size(0), num_unl, acc_pseudo_label, acc_pseudo_selected))
        log.close()
        # percent = self.iters / self.args.max_iters
        # if percent > 1.0:
        #     percent = 1.0
        # num_unl = hard_label_fc.size(0)
        # selected_num = int(percent * num_unl)
        # if self.args.filter_type == 'fc':
        #     scores_for_prediction = soft_label_fc
        #     scores, hard_label_prediction = torch.max(scores_for_prediction, dim=1)
        # elif self.args.filter_type == 'cluster':
        #     scores_for_prediction = soft_label_kmean
        #     scores, hard_label_prediction = torch.max(scores_for_prediction, dim=1)
        # elif self.args.filter_type == 'ssl':
        #     scores_for_prediction = soft_label_lp
        #     scores, hard_label_prediction = torch.max(scores_for_prediction, dim=1)
        # elif self.args.filter_type == 'all':
        #     scores_for_prediction = soft_label_fc + soft_label_kmean + soft_label_lp
        #     scores, hard_label_prediction = torch.max(scores_for_prediction, dim=1)
        #
        # if self.args.category_rank:  ### select samples of each category with TOP percent scores
        #     index_category = torch.BoolTensor(num_unl).fill_(False)
        #     for i in range(self.args.num_class):
        #         category_per_index = hard_label_prediction == i
        #         scores_of_per_category = scores[category_per_index]
        #         num_in_category = scores_of_per_category.size(0)
        #         selected_num_in_category = int(percent * num_in_category)
        #         if selected_num_in_category == 0:
        #             print('no samples have been selected for category %d' % (i))
        #             break
        #         value, _ = torch.topk(scores_of_per_category, selected_num_in_category)
        #         thr = value[-1] - 1e-8
        #         idx_in_category = (scores > thr) & category_per_index
        #         index_category = index_category | idx_in_category
        #     idx = target_u_index[index_category]
        #
        # else:  ### select samples with TOP percent scores
        #     if selected_num == 0:
        #         index_all = torch.BoolTensor(num_unl).fill_(False)
        #         idx = target_u_index[index_all]
        #     else:
        #         value, _ = torch.topk(scores, selected_num)
        #         thr = value[-1] - 1e-8
        #         idx = target_u_index[(scores.cpu() > thr)]
        # self.all_hard_pseudo_label = hard_label_prediction.cuda()
        # self.all_soft_pseudo_label = scores_for_prediction.cuda()
        #
        # acc_pseudo_label = accuracy(self.all_soft_pseudo_label, target_u_gt_label_for_visual)
        # print('Select number: [%d/%d], acc: %3f ' % (idx.size(0), num_unl, acc_pseudo_label))
        # log = open(os.path.join(self.args.save_dir, 'log.txt'), 'a')
        # log.write("\n")
        # log.write('Select number: [%d/%d], acc: %3f ' % (idx.size(0), num_unl, acc_pseudo_label))
        # log.close()


        ## construct a protopical network classifier, where the protopicals are calculated by calculated by the mean of the labeled data and pseudo-labeled unlabeled data.

        #### calculate the target category center with the self.all_hard_pseudo_label
        target_u_feature_matrix = target_u_feature_matrix[reverse_index]  #### re-rank the feature to match the self.all_hard_pseudo_label
        target_feature_category_list = []
        for i in range(self.args.num_class):
            target_feature_category_list.append([])
        for i in range(target_u_feature_matrix.size(0)):   ### source unlabeled data
            pseudo_label = self.all_hard_pseudo_label[i]
            target_feature_category_list[pseudo_label].append(target_u_feature_matrix[i].view(1, target_u_feature_matrix.size(1)))
        for i in range(target_l_feature_matrix.size(0)):   ### target unlabeled data
            pseudo_label = hard_label_tl[i]
            target_feature_category_list[pseudo_label].append(
                target_l_feature_matrix[i].view(1, target_l_feature_matrix.size(1)))
        for i in range(self.args.num_class):
            target_feature_category_list[i] = torch.cat(target_feature_category_list[i], dim=0)
            target_feature_category_list[i] = target_feature_category_list[i].mean(0).view(1, target_u_feature_matrix.size(1))   #### get the mean of each target category
        target_feature_category_matrix = torch.cat(target_feature_category_list, dim=0)

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
        L2_dis = ((source_feature_matrix_unsq - target_feature_category_matrix_unsq) ** 2).mean(2)
        soft_label_target_prototypical = torch.softmax(1 + 1.0 / (L2_dis + 1e-8), dim=1)
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

        for i in range(self.args.num_class):
            source_feature_category = source_feature_matrix[hard_label_s == i]
            target_l_feature_category = target_l_feature_matrix[hard_label_tl == i]
            target_u_feature_category = target_u_feature_matrix[self.all_hard_pseudo_label == i]
            all_feature_category = torch.cat((source_feature_category, target_l_feature_category, target_u_feature_category), dim=0)
            self.proto.data[i] = all_feature_category.mean(0).data.clone()

        self.selected_index = idx.numpy().tolist()  ### the threshold pseudo label
        self.selected_index_source = idx_source.numpy().tolist()  ### the threshold pseudo label
        # self.G.apply(release_bn)
        # self.F.apply(release_bn)

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
        ### 将value 小的值进行截断 (TO DO), whether is work is needed to be verified.
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

        while not stop:
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

            if acc_val > best_prec1_val:
                best_prec1_val = acc_val
                counter = 0
            else:
                counter += 1
            if self.args.early:
                if counter > self.args.patience:
                    log = open(os.path.join(self.args.save_dir, 'log.txt'), 'a')
                    log.write(" !!!!! early stop !!!!! ")
                    log.close()
                    break

            self.epoch += 1

        self.save_ckpt()

    def update_network(self, **kwargs):

        stop = False
        self.G.train()
        self.F.train()

        self.train_data['source']['iterator'] = iter(self.train_data['source']['loader'])
        self.train_data['target_l']['iterator'] = iter(self.train_data['target_l']['loader'])
        self.train_data['target_u']['iterator'] = iter(self.train_data['target_u']['loader'])
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
            target_data_l, target_data_l_strong, target_gt_l = self.get_samples('target_l')
            target_data_u, target_data_u_strong, target_gt_u_not_use, index, target_path = self.get_samples('target_u')

            source_data_train = to_cuda(source_data)
            target_data_l_train = to_cuda(target_data_l)
            ########################
            if self.args.weak_aug:
                target_data_u_train = to_cuda(target_data_u)
            else:
            # source_data_train = to_cuda(source_data_strong)
            # target_data_l_train = to_cuda(target_data_l_strong)
                target_data_u_train = to_cuda(target_data_u_strong)

            source_gt = to_cuda(source_gt)
            target_gt_l = to_cuda(target_gt_l)
            target_data_u_train = to_cuda(target_data_u_train)

            data_time.update(time.time() - end)
            #label_u, valid_u = self.lb_guessor(self.G, self.F, target_data_u)
            label_u_hard, label_u_soft, valid_u = self.get_pseudo_labels(index)
            target_data_u_train = target_data_u_train[valid_u]
            n_u = target_data_u_train.size(0)

            if n_u != 0:
                # ipdb.set_trace()
                num_labeled = source_data_train.size(0) + target_data_l_train.size(0)
                if self.args.mixup == 'vanilla':
                    raise NotImplementedError
                    ## mix [source data & target_data_l] + [target_data_u] to construct the new data, duplicate the few sets
                    data_labeled = torch.cat([source_data_train, target_data_l_train], dim=0)
                    gt_labeled = torch.cat([source_gt, target_gt_l], dim=0)
                    if num_labeled < n_u:
                        quotient = int(n_u / num_labeled)
                        remainder = n_u % num_labeled
                        duplicate_index = list(range(num_labeled)) * quotient + random.sample(list(range(num_labeled)), remainder)

                        data_labeled = data_labeled[duplicate_index]
                        gt_labeled = gt_labeled[duplicate_index]
                        gt_labeled = to_onehot(gt_labeled, self.args.num_class)
                        label_u_hard = to_onehot(label_u_hard, self.args.num_class)

                    elif num_labeled > n_u:
                        quotient = int(num_labeled / n_u)
                        remainder = num_labeled % n_u
                        duplicate_index = list(range(n_u)) * quotient + random.sample(list(range(n_u)), remainder)

                        target_data_u_train = target_data_u_train[duplicate_index]
                        label_u_hard = label_u_hard[duplicate_index]
                        label_u_hard = to_onehot(label_u_hard, self.args.num_class)
                        gt_labeled = to_onehot(gt_labeled, self.args.num_class)
                        label_u_soft = label_u_soft[duplicate_index]
                    #### generate the mixed data
                    beta = torch.from_numpy(np.random.beta(1, 1, [max(num_labeled, n_u), 1])).float().cuda()
                    data = beta.resize(max(num_labeled, n_u), 1, 1, 1) * data_labeled + (1 - beta.resize(max(num_labeled, n_u), 1, 1, 1)) * target_data_u_train
                    mixed_gt = beta * gt_labeled + (1 - beta) * label_u_hard
                    mixed_gt_soft = beta * gt_labeled + (1-beta) * label_u_soft

                    feature = self.G(data)
                    logit = self.F(feature)
                    if self.args.pseudo_type == 'hard':
                        loss = - (mixed_gt * F.log_softmax(logit, dim=1)).sum(1).sum() / data.size(0)
                    elif self.args.pseudo_type == 'soft':
                        loss = - (mixed_gt_soft * F.log_softmax(logit, dim=1)).sum(1).sum() / data.size(0)
                    else:
                        raise NotImplementedError

                    if self.args.prototype_classifier:
                        loss_prototype = self.return_mixdata_prototype_loss(feature, mixed_gt, mixed_gt_soft, self.proto)
                        loss += loss_prototype


                elif self.args.mixup == 'ours':
                    ## mix [source data & target_data_l] + [target_data_u] to construct the new data, if less unlabeled data are selected, small weight applied
                    raise NotImplementedError
                elif self.args.mixup == 'none':
                    num_labeled = source_data_train.size(0) + target_data_l_train.size(0)
                    data = torch.cat([source_data_train, target_data_l_train, target_data_u_train], dim=0)
                    label = torch.cat([source_gt, target_gt_l], dim=0)
                    feature = self.G(data)
                    logit = self.F(feature)
                    logit_labeled, logit_unl = logit[:num_labeled], logit[num_labeled:]
                    feature_labeled, feature_unl = feature[:num_labeled, :], feature[num_labeled:, :]

                    if self.args.source_weight == 'none':
                        loss_labeled = self.CELoss(logit_labeled, label).mean()
                    elif self.args.source_weight == 'overall':
                        loss_labeled = self.CELoss(logit_labeled[:source_data_train.size(0)], source_gt).mean() * self.weight_source \
                                     + self.CELoss(logit_labeled[source_data_train.size(0):], target_gt_l).mean()
                        loss_labeled = loss_labeled * 0.5
                    elif self.args.source_weight == 'instance':
                        ## which index in index_source is also in self.selected_index_source
                        selected_index = self.get_source_index(index_source)
                        print('source selection: %d/%d' % (source_gt[selected_index].size(0), index_source.size(0)))
                        if len(source_gt[selected_index]) == 0:
                            loss_labeled = self.CELoss(logit_labeled[source_data_train.size(0):], target_gt_l).mean() * 0.5
                        else:
                            loss_labeled = self.CELoss(logit_labeled[:source_data_train.size(0)][selected_index], source_gt[selected_index]).sum() / source_data.size(0) +\
                                           self.CELoss(logit_labeled[source_data_train.size(0):], target_gt_l).mean()
                            loss_labeled = loss_labeled * 0.5
                    elif self.args.source_weight == 'soft':
                        soft_weight = self.get_source_soft_weight(index_source)
                        loss_labeled = (self.CELoss(logit_labeled[:source_data_train.size(0)], source_gt) * soft_weight).mean() \
                                     + self.CELoss(logit_labeled[source_data_train.size(0):], target_gt_l).mean()
                        loss_labeled = loss_labeled * 0.5
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
                    # if self.args.prototype_classifier:
                    #     loss_prototype_labeled = self.return_labeled_prototype_loss(feature_labeled, label, self.proto)
                    #     loss_prototype_unl = self.return_unlabeled_prototype_loss(feature_unl, label_u_hard, label_u_soft, self.proto, target_data_u.size(0))
                    #     loss_prototype = loss_prototype_labeled + loss_prototype_unl
                    #     loss += loss_prototype

                else:
                    data = torch.cat([source_data_train, target_data_l_train], dim=0)
                    label = torch.cat([source_gt, target_gt_l], dim=0)
                    feature = self.G(data)
                    logit_labeled = self.F(feature)
                    if self.args.source_weight == 'none':
                        loss = self.CELoss(logit_labeled, label).mean()
                    elif self.args.source_weight == 'overall':
                        loss = self.CELoss(logit_labeled[:source_data_train.size(0)], source_gt).mean() * self.weight_source \
                                     + self.CELoss(logit_labeled[source_data_train.size(0):], target_gt_l).mean()
                        loss = loss * 0.5
                    elif self.args.source_weight == 'instance':
                        ## which index in index_source is also in self.selected_index_source
                        selected_index = self.get_source_index(index_source)
                        if len(source_gt[selected_index]) == 0:
                            loss = self.CELoss(logit_labeled[source_data_train.size(0):], target_gt_l).mean() * 0.5
                        else:
                            loss = self.CELoss(logit_labeled[:source_data_train.size(0)][selected_index],
                                                       source_gt[selected_index]).sum() / source_data.size(0) + \
                                           self.CELoss(logit_labeled[source_data_train.size(0):], target_gt_l).mean()
                            loss = loss * 0.5
                    elif self.args.source_weight == 'soft':
                        soft_weight = self.get_source_soft_weight(index_source)
                        loss = (self.CELoss(logit_labeled[:source_data_train.size(0)], source_gt) * soft_weight).mean() \
                                     + self.CELoss(logit_labeled[source_data_train.size(0):], target_gt_l).mean()
                        loss = loss * 0.5
                    else:
                        raise NotImplementedError
                    if self.args.prototype_classifier:
                        loss_prototype = self.return_labeled_prototype_loss(feature, label, self.proto)
                        loss += loss_prototype

                self.optimizer_G.zero_grad()
                self.optimizer_F.zero_grad()
                loss.backward()
                self.optimizer_G.step()
                self.optimizer_F.step()
                self.ema.update_params()

                losses_all.update(loss.item(), data.size(0))
                # if n_u != 0 and self.args.mixup == 'none':
                #     losses_s.update(loss_labeled.item(), logit_labeled.size(0))
                #     losses_t.update(loss_unl.item(), logit_unl.size(0))

            batch_time.update(time.time() - end)
            end = time.time()
            self.iters += 1
            iters_counter_within_epoch += 1
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

    def return_mixdata_prototype_loss(self, feature_mixed, label_u_hard, label_u_soft, proto):
        prob_pred = (1 + (feature_mixed.unsqueeze(1) - proto.unsqueeze(0)).pow(2).sum(2)).pow(- 1)
        if self.args.pseudo_type == 'hard':
            loss_mix = - (label_u_hard * F.log_softmax(prob_pred, dim=1)).sum(1).mean()
        elif self.args.pseudo_type == 'soft':
            loss_mix = - (label_u_soft * F.log_softmax(prob_pred, dim=1)).sum(1).mean()
        else:
            raise NotImplementedError
        return loss_mix


    def test(self):
        self.ema.G.eval()
        self.ema.F.eval()
        prec1 = AverageMeter()
        for i, (input, target) in enumerate(self.test_data['loader']):
            input, target = to_cuda(input), to_cuda(target)
            with torch.no_grad():
                feature_test = self.ema.G(input)
                output_test = self.ema.F(feature_test)
            prec1_iter = accuracy(output_test, target)
            prec1.update(prec1_iter, input.size(0))


        self.ema.apply_shadow()
        self.ema.G.eval()
        self.ema.F.eval()
        self.ema.G.cuda()
        self.ema.F.cuda()


        prec1_ema = AverageMeter()

        for i, (input, target) in enumerate(self.test_data['loader']):
            input, target = to_cuda(input), to_cuda(target)
            with torch.no_grad():
                feature_test = self.ema.G(input)
                output_test = self.ema.F(feature_test)
            prec1_iter = accuracy(output_test, target)
            prec1_ema.update(prec1_iter, input.size(0))

        print("                       Test:epoch: %d, iter: %d, Acc: %3f, ema_ACC: %3f" % \
            (self.epoch, self.iters, prec1.avg, prec1_ema.avg))
        log = open(os.path.join(self.args.save_dir, 'log.txt'), 'a')
        log.write("\n")
        log.write("                                                                                 Test:epoch: %d, iter: %d, Acc: %3f, ema_Acc: %3f" % \
            (self.epoch, self.iters, prec1.avg, prec1_ema.avg))
        log.close()
        self.ema.restore()
        return max(prec1.avg, prec1_ema.avg), max(prec1.avg, prec1_ema.avg)


    def build_optimizer(self):
        if self.args.optimizer == 'SGD':  ## some params may not contribute the loss_all, thus they are not updated in the training process.
            if self.args.bottle_neck:
                self.optimizer_G = torch.optim.SGD([
                    {'params': self.G.module.conv1.parameters(), 'name': 'pre-trained'},
                    {'params': self.G.module.bn1.parameters(), 'name': 'pre-trained'},
                    {'params': self.G.module.layer1.parameters(), 'name': 'pre-trained'},
                    {'params': self.G.module.layer2.parameters(), 'name': 'pre-trained'},
                    {'params': self.G.module.layer3.parameters(), 'name': 'pre-trained'},
                    {'params': self.G.module.layer4.parameters(), 'name': 'pre-trained'},
                    # {'params': model.module.fc.parameters(), 'name': 'pre-trained'}
                    {'params': self.G.module.bottle_fc.parameters(), 'name': 'new-added'}
                ],
                    lr=self.args.base_lr * 0.1,
                    momentum=self.args.momentum,
                    weight_decay=self.args.wd, ## note the wd is implemented in the ema_optimizer
                    nesterov=True)
            else:
                self.optimizer_G = torch.optim.SGD([
                    {'params': self.G.parameters(), 'name': 'pre-trained'},
                ],
                    lr=self.args.base_lr * 0.1,
                    momentum=self.args.momentum,
                    weight_decay=self.args.wd, ## note the wd is implemented in the ema_optimizer
                    nesterov=True)
            if self.args.prototype_classifier:
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

    # def interleave_offsets(self, batch, nu):
    #     groups = [batch // (nu + 1)] * (nu + 1)
    #     for x in range(batch - sum(groups)):
    #         groups[-x - 1] += 1
    #     offsets = [0]
    #     for g in groups:
    #         offsets.append(offsets[-1] + g)
    #     assert offsets[-1] == batch
    #     return offsets
    #
    # def interleave(self, xy, batch):
    #     nu = len(xy) - 1
    #     offsets = self.interleave_offsets(batch, nu)
    #     xy = [[v[offsets[p]:offsets[p + 1]] for p in range(nu + 1)] for v in xy]
    #     for i in range(1, nu + 1):
    #         xy[0][i], xy[i][i] = xy[i][i], xy[0][i]
    #     return [torch.cat(v, dim=0) for v in xy]

    def save_ckpt(self):
        save_path = self.args.save_dir
        ckpt_resume = os.path.join(save_path, 'last.resume')
        torch.save({'iters': self.iter,
                    'best_prec1': self.best_prec1,
                    'G_state_dict': self.G.state_dict(),
                    'F_state_dict': self.F.state_dict()
                    }, ckpt_resume)