import pickle
import numpy as np
import os
import random
import torch
from torch.utils.data import Dataset
from data.numpy_precess import transform as T
from data.numpy_precess.randaugment import RandomAugment
import ipdb
# from torchvision import transforms
# import torchvision.datasets as datasets
# from data.folder_da import ImageFolder_Strong
# from data.randaugment import RandAugmentMC

def _select_image_process(dataset='cifar10'):
    if dataset == 'cifar10':
        mean, std = (0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)
    elif dataset == 'cifar100':
        mean, std = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
    else:
        raise NotImplementedError

    transforms_train_weak = T.Compose([
                T.Resize((32, 32)),
                T.PadandRandomCrop(border=4, cropsize=(32, 32)),
                T.RandomHorizontalFlip(p=0.5),
                T.Normalize(mean, std),
                T.ToTensor(),
            ])
    transforms_train_strong = T.Compose([
                T.Resize((32, 32)),
                T.PadandRandomCrop(border=4, cropsize=(32, 32)),
                T.RandomHorizontalFlip(p=0.5),
                RandomAugment(2, 10),
                T.Normalize(mean, std),
                T.ToTensor(),
            ])
    transforms_test = T.Compose([
                T.Resize((32, 32)),
                T.Normalize(mean, std),
                T.ToTensor(),
            ])


    return transforms_train_weak, transforms_train_strong, transforms_test

def _load_data_train(L=250, dataset='cifar10', dspth='./data'):
    if dataset == 'cifar10':
        datalist = [
            os.path.join(dspth, 'cifar-10-batches-py', 'data_batch_{}'.format(i + 1))
            for i in range(5)
        ]
        n_class = 10
        assert L in [40, 250, 4000]
    elif dataset == 'cifar100':
        datalist = [
            os.path.join(dspth, 'cifar-100-python', 'train')]
        n_class = 100
        assert L in [400, 2500, 10000]

    data, labels = [], []
    for data_batch in datalist:
        with open(data_batch, 'rb') as fr:
            entry = pickle.load(fr, encoding='latin1')
            lbs = entry['labels'] if 'labels' in entry.keys() else entry['fine_labels']
            data.append(entry['data'])
            labels.append(lbs)
    data = np.concatenate(data, axis=0)
    labels = np.concatenate(labels, axis=0)
    n_labels = L // n_class
    data_x, label_x, data_u, label_u = [], [], [], []
    for i in range(n_class):
        indices = np.where(labels == i)[0]
        np.random.shuffle(indices)
        inds_x, inds_u = indices[:n_labels], indices[n_labels:]
        data_x += [
            data[i].reshape(3, 32, 32).transpose(1, 2, 0)
            for i in inds_x
        ]
        label_x += [labels[i] for i in inds_x]
        data_u += [
            data[i].reshape(3, 32, 32).transpose(1, 2, 0)
            for i in inds_u
        ]
        label_u += [labels[i] for i in inds_u]
    return data_x, label_x, data_u, label_u


def _load_data_train_sl( dataset='cifar10', dspth='./data'):
    if dataset == 'cifar10':
        datalist = [
            os.path.join(dspth, 'cifar-10-batches-py', 'data_batch_{}'.format(i + 1))
            for i in range(5)
        ]
        n_class = 10

    elif dataset == 'cifar100':
        datalist = [
            os.path.join(dspth, 'cifar-100-python', 'train')]
        n_class = 100


    data, labels = [], []
    for data_batch in datalist:
        with open(data_batch, 'rb') as fr:
            entry = pickle.load(fr, encoding='latin1')
            lbs = entry['labels'] if 'labels' in entry.keys() else entry['fine_labels']
            data.append(entry['data'])
            labels.append(lbs)
    data = np.concatenate(data, axis=0)
    labels = np.concatenate(labels, axis=0)

    data_x, label_x = [], []
    for i in range(n_class):
        indices = np.where(labels == i)[0]
        np.random.shuffle(indices)

        data_x += [
            data[i].reshape(3, 32, 32).transpose(1, 2, 0)
            for i in indices
        ]
        label_x += [labels[i] for i in indices]

    return data_x, label_x

def _load_data_val(dataset, dspth='./data'):
    if dataset == 'cifar10':
        datalist = [
            os.path.join(dspth, 'cifar-10-batches-py', 'test_batch')
        ]
    elif dataset == 'cifar100':
        datalist = [
            os.path.join(dspth, 'cifar-100-python', 'test')
        ]

    data, labels = [], []
    for data_batch in datalist:
        with open(data_batch, 'rb') as fr:
            entry = pickle.load(fr, encoding='latin1')
            lbs = entry['labels'] if 'labels' in entry.keys() else entry['fine_labels']
            data.append(entry['data'])
            labels.append(lbs)
    data = np.concatenate(data, axis=0)
    labels = np.concatenate(labels, axis=0)
    data = [
        el.reshape(3, 32, 32).transpose(1, 2, 0)
        for el in data
    ]
    return data, labels

class Cifar(Dataset):
    def __init__(self, data, labels, transform_weak=None, transform_strong=None,  path=False):
        super(Cifar, self).__init__()
        self.data, self.labels = data, labels
        assert len(self.data) == len(self.labels)

        self.trans_weak = transform_weak
        self.trans_strong = transform_strong
        self.path = path


    def __getitem__(self, idx):
        im, lb = self.data[idx], self.labels[idx]
        sample_weak = self.trans_weak(im)
        if self.trans_strong is not None:
            sample_strong = self.trans_strong(im)
            if self.path:
                return sample_weak, sample_strong, lb, idx
            else:
                return sample_weak, sample_strong, lb
        else:
            if self.path:
                return sample_weak, lb, idx
            else:
                return sample_weak, lb


    def __len__(self):
        leng = len(self.data)
        return leng

def generate_dataloader(args):
    dataloaders = {}
    transforms_train_weak, transforms_train_strong, transforms_test = _select_image_process(args.dataset)
    data_x, label_x, data_u, label_u = _load_data_train(L=args.num_labeled, dataset=args.dataset, dspth=args.datapath)
    data_test, labels_test = _load_data_val(dataset=args.dataset, dspth=args.datapath)
    labeled_dataset = Cifar(data=data_x, labels=label_x, transform_weak=transforms_train_weak, transform_strong=transforms_train_strong,  path=False)
    unlabeled_dataset = Cifar(data=data_u, labels=label_u, transform_weak=transforms_train_weak, transform_strong=transforms_train_strong,  path=True)
    if args.test_weakaug:
        test_dataset = Cifar(data=data_test, labels=labels_test, transform_weak=transforms_train_weak, transform_strong=None, path=False)
    else:
        test_dataset = Cifar(data=data_test, labels=labels_test, transform_weak=transforms_test, transform_strong=None, path=False)

    randombatchsampler_l = RandomBatchSampler(min(args.batchsize, len(labeled_dataset)), len(labeled_dataset))
    randombatchsampler_u = RandomBatchSampler(int(args.batchsize * args.mu) , len(unlabeled_dataset))

    labeled_loader = torch.utils.data.DataLoader(labeled_dataset,
                                    num_workers=args.num_workers, batch_sampler=randombatchsampler_l, pin_memory=True)
    unlabeled_loader = torch.utils.data.DataLoader(unlabeled_dataset,
                                                 num_workers=args.num_workers, batch_sampler=randombatchsampler_u, pin_memory=True)
    target_loader_test = torch.utils.data.DataLoader(test_dataset,
                                    batch_size=args.batchsize, num_workers=args.num_workers, shuffle=False, drop_last=False)

    print('the number of instances of labeled, unlabeled, and test data are: %d, %d, and %d, respectively.' % (len(labeled_dataset), len(unlabeled_dataset), len(test_dataset)))
    dataloaders['source'] = labeled_loader    #### modified from DA, the actual name should be labeled ranther than source
    dataloaders['target'] = unlabeled_loader
    dataloaders['test'] = target_loader_test

    return dataloaders


def generate_dataloader_sl(args):
    dataloaders = {}
    transforms_train_weak, transforms_train_strong, transforms_test = _select_image_process(args.dataset)
    data_x, label_x = _load_data_train_sl(dataset=args.dataset, dspth=args.datapath)
    data_test, labels_test = _load_data_val(dataset=args.dataset, dspth=args.datapath)
    labeled_dataset = Cifar(data=data_x, labels=label_x, transform_weak=transforms_train_weak, transform_strong=None,  path=False)

    if args.test_weakaug:
        test_dataset = Cifar(data=data_test, labels=labels_test, transform_weak=transforms_train_weak, transform_strong=None, path=False)
    else:
        test_dataset = Cifar(data=data_test, labels=labels_test, transform_weak=transforms_test, transform_strong=None, path=False)

    randombatchsampler_l = RandomBatchSampler(args.batchsize, len(labeled_dataset))

    labeled_loader = torch.utils.data.DataLoader(labeled_dataset,
                                    num_workers=args.num_workers, batch_sampler=randombatchsampler_l, pin_memory=True)

    target_loader_test = torch.utils.data.DataLoader(test_dataset,
                                    batch_size=args.batchsize, num_workers=args.num_workers, shuffle=False, drop_last=False)

    print('the number of instances of labeled, unlabeled, and test data are: %d, and %d, respectively.' % (len(labeled_dataset), len(test_dataset)))
    dataloaders['source'] = labeled_loader    #### modified from DA, the actual name should be labeled ranther than source
    dataloaders['test'] = target_loader_test

    return dataloaders

def generate_dataloader_pseudo_label(args):
    dataloaders = {}
    transforms_train_weak, transforms_train_strong, transforms_test = _select_image_process(args.dataset)
    data_x, label_x, data_u, label_u = _load_data_train(L=args.num_labeled, dataset=args.dataset, dspth=args.datapath)

    labeled_dataset = Cifar(data=data_x, labels=label_x, transform_weak=transforms_train_weak, transform_strong=None,  path=False)
    unlabeled_dataset = Cifar(data=data_u, labels=label_u, transform_weak=transforms_train_weak, transform_strong=None,  path=True)

    source_loader = torch.utils.data.DataLoader(labeled_dataset,
                                    batch_size=int(args.batchsize * args.mu), num_workers=args.num_workers, shuffle=False, drop_last=False)
    target_l_loader = torch.utils.data.DataLoader(unlabeled_dataset,
                                    batch_size=int(args.batchsize * args.mu), num_workers=args.num_workers, shuffle=False, drop_last=False)

    dataloaders['source'] = source_loader
    dataloaders['target'] = target_l_loader

    return dataloaders


class Sampler(object):
    def __init__(self, data_source):
        pass

    def __iter__(self):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

class UniformBatchSampler(Sampler):
    def __init__(self, per_category, category_index_list):

        self.per_category = per_category
        self.category_index_list = category_index_list

    def __iter__(self):
        for bat in range(60000):
            batch = []
            for i in range(len(self.category_index_list)):
                batch = batch + random.sample(self.category_index_list[i], self.per_category)
            random.shuffle(batch)
            yield batch

class RandomBatchSampler(Sampler):
    def __init__(self, batch_size, len_imgs):

        self.batch_size = batch_size
        self.imgs_list = list(range(len_imgs))

    def __iter__(self):
        for bat in range(60000):
            batch = []
            batch = batch + random.sample(self.imgs_list, self.batch_size)
            random.shuffle(batch)
            yield batch

