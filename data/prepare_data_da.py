import os
import random
import torch
from torchvision import transforms
import torchvision.datasets as datasets
from data.folder_da import ImageFolder_Strong
from data.randaugment import RandAugmentMC

def _select_image_process(DATA_TRANSFORM_TYPE):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    if DATA_TRANSFORM_TYPE == 'ours':
        transforms_train_weak = transforms.Compose([
                transforms.Resize(256),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
        transforms_train_strong = transforms.Compose([
                transforms.Resize(256),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                RandAugmentMC(2, 10),
                transforms.ToTensor(),
                normalize,
            ])
        transforms_test = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])
    elif DATA_TRANSFORM_TYPE == 'mme':
        transforms_train_weak = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
        transforms_train_strong = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                RandAugmentMC(2, 10),
                transforms.ToTensor(),
                normalize,
            ])
        transforms_test = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])
    elif DATA_TRANSFORM_TYPE == 'mme_2weak':
        transforms_train_weak = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
        transforms_train_strong = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
        transforms_test = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])
    elif DATA_TRANSFORM_TYPE == 'simple':
        transforms_train_weak = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                normalize,
            ])
        transforms_train_strong = transforms.Compose([
                transforms.Resize((224, 224)),
                RandAugmentMC(2, 10),
                transforms.ToTensor(),
                normalize,
            ])
        transforms_test = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                normalize,
            ])
    else:
        raise NotImplementedError

    return transforms_train_weak, transforms_train_strong, transforms_test

def generate_dataloader(args):
    dataloaders = {}
    base_path =  os.path.join(args.datapath, args.dataset)
    file_source = os.path.join(base_path, args.source)
    file_target = os.path.join(base_path, args.target)
    file_test = os.path.join(base_path, args.test)
    transforms_train_weak, transforms_train_strong, transforms_test = _select_image_process(args.transform_type)

    source_dataset = ImageFolder_Strong(file_source, transforms_train_weak, transforms_train_strong, path=True)
    target_dataset = ImageFolder_Strong(file_target, transforms_train_weak, transforms_train_strong, path=True)
    test_dataset = datasets.ImageFolder(file_test, transforms_test)



    source_loader = torch.utils.data.DataLoader(source_dataset, batch_size=args.batchsize,
                                                num_workers=args.num_workers, shuffle=True, drop_last=True)
    ##########################################################################################################################
    # randombatchsampler = RandomBatchSampler(batch_size=int(args.batchsize * args.ratio_t), len_imgs=len(target_dataset))
    # target_loader = torch.utils.data.DataLoader(target_dataset,
    #                                 num_workers=args.num_workers, batch_sampler=randombatchsampler)
    # print('In each iteration: source labeleed data: %d, target labeled data: %d, target Unl data: %d' % (args.batchsize - int(args.batchsize * args.ratio_t), int(args.batchsize * args.ratio_t), int(args.batchsize * args.mu)))
    ########################################################################################################################
    # target_loader = torch.utils.data.DataLoader(target_dataset, batch_size=min(args.batchsize, len(target_dataset)),
    #                                 num_workers=3, shuffle=True, drop_last=True)
    print('number of target data is: %d' % (len(target_dataset)))
    batch_size_unl = min(len(target_dataset), int(args.batchsize * args.mu))
    randombatchsampler_unl = RandomBatchSampler(batch_size_unl, len_imgs=len(target_dataset))
    target_loader = torch.utils.data.DataLoader(target_dataset,
                                    num_workers=args.num_workers, batch_sampler=randombatchsampler_unl)
    target_loader_test = torch.utils.data.DataLoader(test_dataset,
                                    batch_size=args.batchsize, num_workers=args.num_workers, shuffle=False, drop_last=False)

    dataloaders['source'] = source_loader
    dataloaders['target'] = target_loader
    dataloaders['test'] = target_loader_test

    return dataloaders


def generate_dataloader_pseudo_label(args):
    dataloaders = {}
    base_path =  os.path.join(args.datapath, args.dataset)
    file_source = os.path.join(base_path, args.source)
    file_target = os.path.join(base_path, args.target)

    transforms_train_weak, transforms_train_strong, transforms_test = _select_image_process(args.transform_type)

    source_dataset = ImageFolder_Strong(file_source, transforms_train_weak, transforms_train_strong, path=True)
    target_dataset = ImageFolder_Strong(file_target, transforms_train_weak, transforms_train_strong, path=True)

    source_loader = torch.utils.data.DataLoader(source_dataset,
                                    batch_size=args.batchsize * args.mu, num_workers=args.num_workers, shuffle=False, drop_last=False)
    target_l_loader = torch.utils.data.DataLoader(target_dataset,
                                    batch_size=args.batchsize * args.mu, num_workers=args.num_workers, shuffle=False, drop_last=False)

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

