import os
import random
import torch
from torchvision import transforms
import torchvision.datasets as datasets
from data.folder_da import ImageFolder_Strong
from data.randaugment import RandAugmentMC

from PIL import Image
import os
import shutil

import ipdb
import numpy as np
from scipy.io import loadmat
from scipy import ndimage
import torch.utils.data as data


class Dataset(data.Dataset):
    """Args:
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """

    def __init__(self, data, label,
                 transform=None, target_transform=None, path=False, pseudo_strong=False):
        self.transform = transform
        self.target_transform = target_transform
        self.samples = torch.from_numpy(data).float()
        self.labels = torch.from_numpy(label)
        self.path = path
        self.pseudo_strong = pseudo_strong

    def __getitem__(self, index):
        """
         Args:
             index (int): Index
         Returns:
             tuple: (image, target) where target is index of the target class.
         """

        img, target = self.samples[index], self.labels[index]
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        # img = Image.fromarray(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        # if self.transform is not None:
        #     # img = self.transform(img)
        img_out = img / 255.0
        if self.pseudo_strong:  ### to match the normal image
            if self.path:
                return img_out, img_out, target, index, index
            else:
                return img_out, img_out, target
        else:
            if self.path:
                return img_out, target, index, index
            else:
                return img_out, target

    def __len__(self):
        return len(self.samples)



def shuffle(xs, ys):
    indices = list(range(len(xs)))
    np.random.shuffle(indices)
    return xs[indices], ys[indices]

def prepare_numpy_data_mnist(args):
    train_set = datasets.MNIST('./data', train=True, download=True)
    test_set = datasets.MNIST('./data', train=False, download=True)

    train_set_array = train_set.data.numpy()
    test_set_array = test_set.data.numpy()

    train_y_array = train_set.targets.numpy()
    test_y_array = test_set.targets.numpy()

    np.random.seed(args.seed)
    train_set_array, train_y_array = shuffle(train_set_array, train_y_array)

    return train_set_array[:50000], train_y_array[:50000], test_set_array, test_y_array

def sample_rotate_images(xs, start_angle, end_angle, re_size=28):
    new_xs = []
    num_points = xs.shape[0]
    for i in range(num_points):
        if start_angle == end_angle:
            angle = start_angle
        else:
            angle = np.random.uniform(low=start_angle, high=end_angle)
        # img = np.array(Image.fromarray(xs[i]).resize((re_size, re_size)))
        img = ndimage.rotate(xs[i], angle, reshape=False)
        img = img.reshape((1, re_size, re_size))
        # img = np.concatenate([img, img, img], 0)

        new_xs.append(img)

    return np.array(new_xs)


def continually_rotate_images(xs, start_angle, end_angle, re_size=28):
    new_xs = []
    num_points = xs.shape[0]
    for i in range(num_points):
        angle = float(end_angle - start_angle) / num_points * i + start_angle
        # img = np.array(Image.fromarray(xs[i]).resize((re_size, re_size)))

        img = ndimage.rotate(xs[i], angle, reshape=False)
        img = img.reshape((1, re_size, re_size))
        # img = xs[i].reshape((1, re_size, re_size))
        # # img = np.concatenate([img, img, img], 0)
        # img = ndimage.rotate(img, angle, reshape=False)
        new_xs.append(img)
    return np.array(new_xs)

def generate_dataloader(args):
    dataloaders = {}
    rotation_name = '_s_' + str(args.s_start) + 'to' + str(args.s_end) + '_t_' + str(args.t_start) + 'to' + str(args.t_end)

    saved_process_data = 'saved_processed_mnist' + rotation_name + '.pth.tar'
    if os.path.isfile(saved_process_data):
        mnist_pack = torch.load(saved_process_data)
        src_x, src_y, tar_x, tar_y, test_x, test_y = \
            mnist_pack['source_data'], mnist_pack['source_label'], mnist_pack['target_data'], mnist_pack['target_label'], mnist_pack['test_data'], mnist_pack['test_label']
    else:
        train_set_array, train_y_array, test_data_array, test_y_array = prepare_numpy_data_mnist(args)
        n_src = 25000

        src_x, src_y = train_set_array[: n_src], train_y_array[: n_src]
        src_x = continually_rotate_images(src_x, args.s_start, args.s_end)

        tar_x, tar_y = train_set_array[n_src:], train_y_array[n_src:]
        tar_x = continually_rotate_images(tar_x, args.t_start, args.t_end)

        test_x = continually_rotate_images(test_data_array, args.t_start, args.t_end)
        test_y = test_y_array

        torch.save({'source_data': src_x,
                    'source_label': src_y,
                    'target_data': tar_x,
                    'target_label': tar_y,
                    'test_data': test_x,
                    'test_label': test_y
                    }, saved_process_data)

    source_dataset = Dataset(src_x, src_y, pseudo_strong=True, path=True)
    target_dataset = Dataset(tar_x, tar_y, pseudo_strong=True, path=True)
    test_dataset = Dataset(test_x, test_y)

    source_loader = torch.utils.data.DataLoader(source_dataset, batch_size=args.batchsize,
                                                num_workers=args.num_workers, shuffle=True, drop_last=True)

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
    rotation_name = '_s_' + str(args.s_start) + 'to' + str(args.s_end) + '_t_' + str(args.t_start) + 'to' + str(args.t_end)

    saved_process_data = 'saved_processed_mnist' + rotation_name + '.pth.tar'
    if os.path.isfile(saved_process_data):
        mnist_pack = torch.load(saved_process_data)
        src_x, src_y, tar_x, tar_y, test_x, test_y = \
            mnist_pack['source_data'], mnist_pack['source_label'], mnist_pack['target_data'], mnist_pack['target_label'], mnist_pack['test_data'], mnist_pack['test_label']
    else:
        train_set_array, train_y_array, test_data_array, test_y_array = prepare_numpy_data_mnist(args)
        n_src = 25000

        src_x, src_y = train_set_array[: n_src], train_y_array[: n_src]
        src_x = continually_rotate_images(src_x, args.s_start, args.s_end)

        tar_x, tar_y = train_set_array[n_src:], train_y_array[n_src:]
        tar_x = continually_rotate_images(tar_x, args.t_start, args.t_end)

        test_x = continually_rotate_images(test_data_array, args.t_start, args.t_end)
        test_y = test_y_array

        torch.save({'source_data': src_x,
                    'source_label': src_y,
                    'target_data': tar_x,
                    'target_label': tar_y,
                    'test_data': test_x,
                    'test_label': test_y
                    }, saved_process_data)

    source_dataset = Dataset(src_x, src_y, pseudo_strong=True, path=True)
    target_dataset = Dataset(tar_x, tar_y, pseudo_strong=True, path=True)

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

