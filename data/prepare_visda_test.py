import os
import random
import torch
from torchvision import transforms
import torchvision.datasets as datasets
from data.folder_da import ImageFolder_Strong
from data.randaugment import RandAugmentMC

from data.vision import VisionDataset

from PIL import Image

import os
import os.path
import sys

import ipdb
########################### added part ###
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True  ## used to handle some error when loading the special images.

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




def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.
    Args:
        filename (string): path to a file
        extensions (tuple of strings): extensions to consider (lowercase)
    Returns:
        bool: True if the filename ends with one of given extensions
    """
    return filename.lower().endswith(extensions)


def is_image_file(filename):
    """Checks if a file is an allowed image extension.
    Args:
        filename (string): path to a file
    Returns:
        bool: True if the filename ends with a known image extension
    """
    return has_file_allowed_extension(filename, IMG_EXTENSIONS)


# def make_dataset(dir, class_to_idx, extensions=None, is_valid_file=None):
#     images = []
#     dir = os.path.expanduser(dir)
#     if not ((extensions is None) ^ (is_valid_file is None)):
#         raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")
#     if extensions is not None:
#         def is_valid_file(x):
#             return has_file_allowed_extension(x, extensions)
#     for target in sorted(class_to_idx.keys()):
#         d = os.path.join(dir, target)
#         if not os.path.isdir(d):
#             continue
#         for root, _, fnames in sorted(os.walk(d, followlinks=True)):
#             for fname in sorted(fnames):
#                 path = os.path.join(root, fname)
#                 if is_valid_file(path):
#                     item = (path, class_to_idx[target])
#                     images.append(item)
#
#     return images

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)

def make_dataset_from_text(root):
    images = []
    image_list_with_labels = os.path.join(root, 'image_list.txt')
    with open(image_list_with_labels) as f:
        line_example = f.readlines()
    f.close()
    for line in line_example:
        splits = line.split(' ')
        path = os.path.join(root, splits[0])
        label = int(splits[1])
        item = (path, label)
        images.append(item)

    return images

class DatasetFolder_visdatest(VisionDataset):
    """A generic data loader where the samples are arranged in this way: ::
        root/class_x/xxx.ext
        root/class_x/xxy.ext
        root/class_x/xxz.ext
        root/class_y/123.ext
        root/class_y/nsdf3.ext
        root/class_y/asd932_.ext
    Args:
        root (string): Root directory path.
        loader (callable): A function to load a sample given its path.
        extensions (tuple[string]): A list of allowed extensions.
            both extensions and is_valid_file should not be passed.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.
        is_valid_file (callable, optional): A function that takes path of a file
            and check if the file is a valid file (used to check of corrupt files)
            both extensions and is_valid_file should not be passed.
     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        samples (list): List of (sample path, class_index) tuples
        targets (list): The class_index value for each image in the dataset
    """

    def __init__(self, root, loader=default_loader, extensions=None, transform=None,
                 target_transform=None, path=False, is_valid_file=None):
        super(DatasetFolder_visdatest, self).__init__(root, transform=transform,
                                            target_transform=target_transform)
        # classes, class_to_idx = self._find_classes(self.root)
        # samples = make_dataset(self.root, class_to_idx, extensions, is_valid_file)
        samples = make_dataset_from_text(self.root)
        if len(samples) == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " + self.root + "\n"
                                "Supported extensions are: " + ",".join(extensions)))

        self.loader = loader
        self.extensions = extensions

        self.samples = samples
        self.targets = [s[1] for s in samples]
        self.path = path

    # def _find_classes(self, dir):
    #     """
    #     Finds the class folders in a dataset.
    #     Args:
    #         dir (string): Root directory path.
    #     Returns:
    #         tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.
    #     Ensures:
    #         No class is a subdirectory of another.
    #     """
    #     if sys.version_info >= (3, 5):
    #         # Faster and available in Python 3.5 and above
    #         classes = [d.name for d in os.scandir(dir) if d.is_dir()]
    #     else:
    #         classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    #     classes.sort()
    #     class_to_idx = {classes[i]: i for i in range(len(classes))}
    #     return classes, class_to_idx

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample_weak = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        if self.path:
            return sample_weak, target, index, path
        else:
            return sample_weak, target

    def __len__(self):
        return len(self.samples)


IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')




def generate_dataloader(args):
    dataloaders = {}
    base_path =  os.path.join(args.datapath, args.dataset)
    file_source = os.path.join(base_path, args.source)
    file_target = os.path.join(base_path, args.target)
    file_test = os.path.join(base_path, args.test)
    transforms_train_weak, transforms_train_strong, transforms_test = _select_image_process(args.transform_type)

    # source_dataset = ImageFolder_Strong(file_source, transforms_train_weak, transforms_train_strong, path=True)
    # target_dataset = ImageFolder_Strong(file_target, transforms_train_weak, transforms_train_strong, path=True)
    test_dataset = DatasetFolder_visdatest(file_test, transform=transforms_test)



    # source_loader = torch.utils.data.DataLoader(source_dataset, batch_size=args.batchsize,
    #                                             num_workers=args.num_workers, shuffle=True, drop_last=True)
    # print('number of target data is: %d' % (len(target_dataset)))
    # batch_size_unl = min(len(target_dataset), int(args.batchsize * args.mu))
    # randombatchsampler_unl = RandomBatchSampler(batch_size_unl, len_imgs=len(target_dataset))
    # target_loader = torch.utils.data.DataLoader(target_dataset,
    #                                 num_workers=args.num_workers, batch_sampler=randombatchsampler_unl)
    target_loader_test = torch.utils.data.DataLoader(test_dataset,
                                    batch_size=args.batchsize * args.mu, num_workers=args.num_workers, shuffle=False, drop_last=False)
    #
    # dataloaders['source'] = source_loader
    # dataloaders['target'] = target_loader
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

