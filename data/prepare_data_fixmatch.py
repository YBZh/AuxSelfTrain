import os
import random
import torch
from torchvision import transforms
from data.data_list import Imagelists_VISDA, return_classlist, Imagelists_VISDA_unl, Imagelists_VISDA_unl_paired
from data.randaugment import RandAugmentMC

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
    base_path = './data/txt/%s' % args.dataset
    root = args.datapath
    image_set_file_s = os.path.join(base_path, 'labeled_source_images_' + args.source + '.txt')
    image_set_file_t = os.path.join(base_path, 'labeled_target_images_' + args.target + '_%d.txt' % (args.num_labeled))
    #image_set_file_t_val = os.path.join(base_path, 'validation_target_images_' + args.target + '_3.txt') #### actually, we don't utilize the val dataset
    image_set_file_unl = os.path.join(base_path, 'unlabeled_target_images_' + args.target + '_%d.txt' % (args.num_labeled))

    transforms_train_weak, transforms_train_strong, transforms_test = _select_image_process(args.transform_type)
    ############ dataloader #############################
    source_dataset = Imagelists_VISDA_unl(image_set_file_s, root=root, transform=transforms_train_weak, transform2=transforms_train_strong, path=True)
    target_dataset = Imagelists_VISDA_unl(image_set_file_t, root=root, transform=transforms_train_weak, transform2=transforms_train_strong)

    target_dataset_unl = Imagelists_VISDA_unl(image_set_file_unl, root=root, transform=transforms_train_weak, transform2=transforms_train_strong, path=True)
    # target_dataset_unl_pseudo = Imagelists_VISDA_unl(image_set_file_unl, root=root, transform=transforms_train_weak, transform2=transforms_train_strong, path=True)
    target_dataset_test = Imagelists_VISDA(image_set_file_unl, root=root,  transform=transforms_test)
    class_list = return_classlist(image_set_file_s)
    print("%d classes in this dataset" % len(class_list))
    source_loader = torch.utils.data.DataLoader(source_dataset, batch_size=args.batchsize - int(args.batchsize * args.ratio_t),
                                                num_workers=args.num_workers, shuffle=True, drop_last=True, pin_memory=True)
    ##########################################################################################################################
    randombatchsampler = RandomBatchSampler(batch_size=int(args.batchsize * args.ratio_t), len_imgs=len(target_dataset))
    target_loader = torch.utils.data.DataLoader(target_dataset,
                                    num_workers=args.num_workers, batch_sampler=randombatchsampler, pin_memory=True)
    print('In each iteration: source labeleed data: %d, target labeled data: %d, target Unl data: %d' % (args.batchsize - int(args.batchsize * args.ratio_t), int(args.batchsize * args.ratio_t), int(args.batchsize * args.mu)))
    ########################################################################################################################
    # target_loader = torch.utils.data.DataLoader(target_dataset, batch_size=min(args.batchsize, len(target_dataset)),
    #                                 num_workers=3, shuffle=True, drop_last=True)

    randombatchsampler_unl = RandomBatchSampler(int(args.batchsize * args.mu), len_imgs=len(target_dataset_unl))
    target_loader_unl = torch.utils.data.DataLoader(target_dataset_unl,
                                    num_workers=args.num_workers, batch_sampler=randombatchsampler_unl, pin_memory=True)

    # target_loader_unl = torch.utils.data.DataLoader(target_dataset_unl,
    #                                 batch_size=args.batchsize * args.mu, num_workers=4, shuffle=True, drop_last=True)
    # target_loader_pseudo = torch.utils.data.DataLoader(target_dataset_unl_pseudo, batch_size=args.batchsize,
    #                                 num_workers=args.num_workers, shuffle=False, drop_last=False)
    target_loader_test = torch.utils.data.DataLoader(target_dataset_test,
                                    batch_size=args.batchsize, num_workers=args.num_workers, shuffle=False, drop_last=False, pin_memory=True)

    dataloaders['source'] = source_loader
    dataloaders['target_l'] = target_loader
    dataloaders['target_u'] = target_loader_unl
    # dataloaders['target_u_pseudo'] = target_loader_pseudo
    dataloaders['test'] = target_loader_test

    return dataloaders


def generate_dataloader_path(args):
    dataloaders = {}
    base_path = './data/txt/%s' % args.dataset
    root = args.datapath
    image_set_file_s = os.path.join(base_path, 'labeled_source_images_' + args.source + '.txt')
    image_set_file_t = os.path.join(base_path, 'labeled_target_images_' + args.target + '_%d.txt' % (args.num_labeled))
    image_set_file_t_val = os.path.join(base_path, 'validation_target_images_' + args.target + '_3.txt') #### actually, we don't utilize the val dataset
    image_set_file_unl = os.path.join(base_path, 'unlabeled_target_images_' + args.target + '_%d.txt' % (args.num_labeled))

    transforms_train_weak, transforms_train_strong, transforms_test = _select_image_process(args.transform_type)
    ############ dataloader #############################
    source_dataset = Imagelists_VISDA_unl(image_set_file_s, root=root, transform=transforms_train_weak, transform2=transforms_train_strong, path=True)
    target_dataset = Imagelists_VISDA_unl(image_set_file_t, root=root, transform=transforms_train_weak, transform2=transforms_train_strong, path=True)

    target_dataset_unl = Imagelists_VISDA_unl(image_set_file_unl, root=root, transform=transforms_train_weak, transform2=transforms_train_strong, path=True)
    target_dataset_val = Imagelists_VISDA(image_set_file_t_val, root=root, transform=transforms_test, path=True)
    target_dataset_test = Imagelists_VISDA(image_set_file_unl, root=root,  transform=transforms_test, path=True)
    class_list = return_classlist(image_set_file_s)
    print("%d classes in this dataset" % len(class_list))
    source_loader = torch.utils.data.DataLoader(source_dataset, batch_size=args.batchsize - int(args.batchsize * args.ratio_t),
                                                num_workers=args.num_workers, shuffle=True, drop_last=True)
    ##########################################################################################################################
    randombatchsampler = RandomBatchSampler(batch_size=int(args.batchsize * args.ratio_t), len_imgs=len(target_dataset))
    target_loader = torch.utils.data.DataLoader(target_dataset,
                                    num_workers=args.num_workers, batch_sampler=randombatchsampler)
    print('In each iteration: source labeleed data: %d, target labeled data: %d, target Unl data: %d' % (args.batchsize - int(args.batchsize * args.ratio_t), int(args.batchsize * args.ratio_t), args.batchsize * args.mu))
    ########################################################################################################################
    target_loader_proto = torch.utils.data.DataLoader(target_dataset, batch_size=min(args.batchsize, len(target_dataset)),
                                    num_workers=args.num_workers, shuffle=False, drop_last=False)

    randombatchsampler_unl = RandomBatchSampler(args.batchsize * args.mu, len_imgs=len(target_dataset_unl))
    target_loader_unl = torch.utils.data.DataLoader(target_dataset_unl,
                                    num_workers=args.num_workers, batch_sampler=randombatchsampler_unl)

    # target_loader_unl = torch.utils.data.DataLoader(target_dataset_unl,
    #                                 batch_size=args.batchsize * args.mu, num_workers=args.num_workers, shuffle=True, drop_last=True)
    target_loader_val = torch.utils.data.DataLoader(target_dataset_val, batch_size=min(args.batchsize, len(target_dataset_val)),
                                    num_workers=args.num_workers, shuffle=False, drop_last=False)
    target_loader_test = torch.utils.data.DataLoader(target_dataset_test,
                                    batch_size=args.batchsize, num_workers=args.num_workers, shuffle=False, drop_last=False)

    dataloaders['source'] = source_loader
    dataloaders['target_l'] = target_loader
    dataloaders['target_l_proto'] = target_loader_proto
    dataloaders['target_u'] = target_loader_unl
    dataloaders['val'] = target_loader_val
    dataloaders['test'] = target_loader_test

    return dataloaders


def generate_dataloader_mmd(args):
    dataloaders = {}
    base_path = './data/txt/%s' % args.dataset
    root = args.datapath
    image_set_file_s = os.path.join(base_path, 'labeled_source_images_' + args.source + '.txt')
    image_set_file_t = os.path.join(base_path, 'labeled_target_images_' + args.target + '_%d.txt' % (args.num_labeled))
    image_set_file_unl = os.path.join(base_path, 'unlabeled_target_images_' + args.target + '_%d.txt' % (args.num_labeled))
    transforms_train_weak, transforms_train_strong, transforms_test = _select_image_process(args.transform_type)

    source_dataset = Imagelists_VISDA(image_set_file_s, root=root, transform=transforms_train_weak, path=True)
    target_dataset_l = Imagelists_VISDA(image_set_file_t, root=root, transform=transforms_train_weak, path=False)
    target_dataset_u = Imagelists_VISDA(image_set_file_unl, root=root, transform=transforms_train_weak, path=True)

    source_loader = torch.utils.data.DataLoader(source_dataset,
                                    batch_size=args.batchsize, num_workers=args.num_workers, shuffle=True, drop_last=False, pin_memory=True)
    target_l_loader = torch.utils.data.DataLoader(target_dataset_l,
                                    batch_size=args.batchsize, num_workers=args.num_workers, shuffle=True, drop_last=False, pin_memory=True)
    target_u_loader = torch.utils.data.DataLoader(target_dataset_u,
                                    batch_size=args.batchsize, num_workers=args.num_workers, shuffle=True, drop_last=False, pin_memory=True)

    dataloaders['source'] = source_loader
    dataloaders['target_l'] = target_l_loader
    dataloaders['target_u'] = target_u_loader
    return dataloaders

def generate_dataloader_paired(args):
    dataloaders = {}
    base_path = './data/txt/%s' % args.dataset
    root = args.datapath
    image_set_file_s = os.path.join(base_path, 'labeled_source_images_' + args.source + '.txt')
    image_set_file_t = os.path.join(base_path, 'labeled_target_images_' + args.target + '_%d.txt' % (args.num_labeled))
    image_set_file_t_val = os.path.join(base_path, 'validation_target_images_' + args.target + '_3.txt') #### actually, we don't utilize the val dataset
    image_set_file_unl = os.path.join(base_path, 'unlabeled_target_images_' + args.target + '_%d.txt' % (args.num_labeled))

    transforms_train_weak, transforms_train_strong, transforms_test = _select_image_process(args.transform_type)
    ############ dataloader #############################
    # source_dataset = Imagelists_VISDA_unl(image_set_file_s, root=root, transform=transforms_train_weak, transform2=transforms_train_strong)
    # target_dataset = Imagelists_VISDA_unl(image_set_file_t, root=root, transform=transforms_train_weak, transform2=transforms_train_strong)
    st_dataset = Imagelists_VISDA_unl_paired(image_set_file_s, image_set_file_t,  root=root, transform=transforms_train_weak,
                                          transform2=transforms_train_strong)

    target_dataset_unl = Imagelists_VISDA_unl(image_set_file_unl, root=root, transform=transforms_train_weak, transform2=transforms_train_strong)
    target_dataset_val = Imagelists_VISDA(image_set_file_t_val, root=root, transform=transforms_test)
    target_dataset_test = Imagelists_VISDA(image_set_file_unl, root=root,  transform=transforms_test)
    class_list = return_classlist(image_set_file_s)
    print("%d classes in this dataset" % len(class_list))
    st_loader = torch.utils.data.DataLoader(st_dataset, batch_size=int(args.batchsize * 0.5),
                                                num_workers=args.num_workers, shuffle=True, drop_last=True)
    # source_loader = torch.utils.data.DataLoader(source_dataset, batch_size=args.batchsize - int(args.batchsize * args.ratio_t),
    #                                             num_workers=args.num_workers, shuffle=True, drop_last=True)
    ##########################################################################################################################
    # randombatchsampler = RandomBatchSampler(batch_size=int(args.batchsize * args.ratio_t), len_imgs=len(target_dataset))
    # target_loader = torch.utils.data.DataLoader(target_dataset,
    #                                 num_workers=args.num_workers, batch_sampler=randombatchsampler)
    print('In each iteration: source labeleed data: %d, target labeled data: %d, target Unl data: %d' % (args.batchsize - int(args.batchsize * args.ratio_t), int(args.batchsize * args.ratio_t), args.batchsize * args.mu))
    ########################################################################################################################
    # target_loader = torch.utils.data.DataLoader(target_dataset, batch_size=min(args.batchsize, len(target_dataset)),
    #                                 num_workers=3, shuffle=True, drop_last=True)

    randombatchsampler_unl = RandomBatchSampler(args.batchsize * args.mu, len_imgs=len(target_dataset_unl))
    target_loader_unl = torch.utils.data.DataLoader(target_dataset_unl,
                                    num_workers=args.num_workers, batch_sampler=randombatchsampler_unl)

    # target_loader_unl = torch.utils.data.DataLoader(target_dataset_unl,
    #                                 batch_size=args.batchsize * args.mu, num_workers=args.num_workers, shuffle=True, drop_last=True)
    target_loader_val = torch.utils.data.DataLoader(target_dataset_val, batch_size=min(args.batchsize, len(target_dataset_val)),
                                    num_workers=args.num_workers, shuffle=False, drop_last=False)
    target_loader_test = torch.utils.data.DataLoader(target_dataset_test,
                                    batch_size=args.batchsize, num_workers=args.num_workers, shuffle=False, drop_last=False)

    dataloaders['st'] = st_loader
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

