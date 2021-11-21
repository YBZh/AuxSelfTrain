import torch
import torch.nn as nn
import ipdb

class BaseSolver:
    def __init__(self, G, F, dataloaders, args, **kwargs):
        self.args = args
        self.G = G
        self.F = F
        self.dataloaders = dataloaders
        self.CELoss = nn.CrossEntropyLoss()
        if torch.cuda.is_available():
            self.CELoss.cuda()
        self.num_classes = args.num_class
        self.epoch = 0
        self.iters = 0
        self.best_prec1 = 0
        self.iters_per_epoch = None
        if self.args.net == 'resnet34':
            self.inc = 512
        elif self.args.net == "alexnet":
            self.inc = 4096
        elif args.net == "wideresnet_fixmatch":
            self.inc = 4096
        elif args.net == "wideresnet":
            self.inc = 4096
        from model.construct_model import construct_prototype
        self.proto = construct_prototype(args)
        self.build_optimizer()
        self.init_data(self.dataloaders)


    def init_data(self, dataloaders):
        self.train_data = {key: dict() for key in dataloaders if key != 'test'}
        for key in self.train_data.keys():
            if key not in dataloaders:
                continue
            cur_dataloader = dataloaders[key]
            self.train_data[key]['loader'] = cur_dataloader
            self.train_data[key]['iterator'] = None

        if 'test' in dataloaders:
            self.test_data = dict()
            self.test_data['loader'] = dataloaders['test']

    def build_optimizer(self):
        print('Optimizer built')


    def complete_training(self):
        if self.iters > self.args.max_iters:
            return True

    def solve(self):
        print('Training Done!')

    def get_samples(self, data_name):
        assert(data_name in self.train_data)
        assert('loader' in self.train_data[data_name] and \
               'iterator' in self.train_data[data_name])

        data_loader = self.train_data[data_name]['loader']
        data_iterator = self.train_data[data_name]['iterator']
        assert data_loader is not None and data_iterator is not None, \
            'Check your dataloader of %s.' % data_name

        try:
            sample = next(data_iterator)
        except StopIteration:
            data_iterator = iter(data_loader)
            sample = next(data_iterator)
            self.train_data[data_name]['iterator'] = data_iterator
        return sample


    def update_network(self, **kwargs):
        pass