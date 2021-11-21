import os
import random
import torch
from torchvision import transforms
from data.data_list import Imagelists_VISDA, return_classlist, Imagelists_VISDA_unl

def _select_image_process(DATA_TRANSFORM_TYPE):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    if DATA_TRANSFORM_TYPE == 'ours':
        transforms_train = transforms.Compose([
                transforms.Resize(256),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
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
        transforms_train = transforms.Compose([
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
        transforms_train = transforms.Compose([
                transforms.Resize((224, 224)),
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

    return transforms_train, transforms_test

def generate_dataloader(args):
    dataloaders = {}
    base_path = './data/txt/%s' % args.dataset
    root = args.datapath
    image_set_file_s = os.path.join(base_path, 'labeled_source_images_' + args.source + '.txt')
    image_set_file_t = os.path.join(base_path, 'labeled_target_images_' + args.target + '_%d.txt' % (args.num_labeled))
    image_set_file_t_val = os.path.join(base_path, 'validation_target_images_' + args.target + '_3.txt') #### actually, we don't utilize the val dataset
    image_set_file_unl = os.path.join(base_path, 'unlabeled_target_images_' + args.target + '_%d.txt' % (args.num_labeled))

    transforms_train, transforms_test = _select_image_process(args.transform_type)
    ############ dataloader #############################
    source_dataset = Imagelists_VISDA(image_set_file_s, root=root, transform=transforms_train)
    target_dataset = Imagelists_VISDA(image_set_file_t, root=root, transform=transforms_train)
    target_dataset_val = Imagelists_VISDA(image_set_file_t_val, root=root, transform=transforms_train)
    target_dataset_unl = Imagelists_VISDA_unl(image_set_file_unl, root=root, transform=transforms_train, transform2=transforms_train)
    target_dataset_test = Imagelists_VISDA(image_set_file_unl, root=root,  transform=transforms_test)
    class_list = return_classlist(image_set_file_s)
    print("%d classes in this dataset" % len(class_list))
    source_loader = torch.utils.data.DataLoader(source_dataset, batch_size=args.batchsize,
                                                num_workers=3, shuffle=True, drop_last=True)
    #################### sample per_category samples from each category to construct the batch ##############
    # uniformbatchsampler = UniformBatchSampler(per_category=1, category_index_list=category_index_list)
    # target_loader = torch.utils.data.DataLoader(target_dataset,
    #                                 num_workers=3, batch_sampler=uniformbatchsampler)
    ##########################################################################################################################
    randombatchsampler = RandomBatchSampler(batch_size=min(args.batchsize, len(target_dataset)), len_imgs=len(target_dataset))
    target_loader = torch.utils.data.DataLoader(target_dataset,
                                    num_workers=3, batch_sampler=randombatchsampler)
    ########################################################################################################################
    # target_loader = torch.utils.data.DataLoader(target_dataset, batch_size=min(args.batchsize, len(target_dataset)),
    #                                 num_workers=3, shuffle=True, drop_last=True)
    target_loader_unl = torch.utils.data.DataLoader(target_dataset_unl,
                                    batch_size=args.batchsize, num_workers=3, shuffle=True, drop_last=True)
    target_loader_val = torch.utils.data.DataLoader(target_dataset_val, batch_size=min(args.batchsize, len(target_dataset_val)),
                                    num_workers=3, shuffle=False, drop_last=False)
    target_loader_test = torch.utils.data.DataLoader(target_dataset_test,
                                    batch_size=args.batchsize, num_workers=3, shuffle=False, drop_last=False)

    dataloaders['source'] = source_loader
    dataloaders['target_l'] = target_loader
    dataloaders['target_u'] = target_loader_unl
    dataloaders['val'] = target_loader_val
    dataloaders['test'] = target_loader_test

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

