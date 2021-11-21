import numpy as np
import os
import os.path
from PIL import Image
import random
import ipdb

def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def make_dataset_fromlist(image_list):
    with open(image_list) as f:
        image_index = [x.split(' ')[0] for x in f.readlines()]
    with open(image_list) as f:
        label_list = []
        selected_list = []
        for ind, x in enumerate(f.readlines()):
            label = x.split(' ')[1].strip()
            label_list.append(int(label))
            selected_list.append(ind)
        image_index = np.array(image_index)
        label_list = np.array(label_list)
    image_index = image_index[selected_list]
    return image_index, label_list





class Imagelists_VISDA(object):
    def __init__(self, image_list, root="./data/multi/",
                 transform=None, target_transform=None, path=False):
        imgs, labels = make_dataset_fromlist(image_list)
        self.imgs = imgs
        self.labels = labels
        self.transform = transform
        self.target_transform = target_transform
        self.loader = pil_loader
        self.root = root
        self.path = path


        self.num_class = len(return_classlist(image_list))
        self.category_index_list = []
        for i in range(self.num_class):
            list_temp = []
            for j in range(len(self.imgs)):
                if i == self.labels[j]:
                    list_temp.append(j)
            self.category_index_list.append(list_temp)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is
            class_index of the target class.
        """
        path = os.path.join(self.root, self.imgs[index])
        target = self.labels[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        if not self.path:
            return img, target
        else:
            return img, target, index, self.imgs[index]  ### the path of the selected images.

    def __len__(self):
        return len(self.imgs)


def return_classlist(image_list):
    with open(image_list) as f:
        label_list = []
        for ind, x in enumerate(f.readlines()):
            label = x.split(' ')[0].split('/')[-2]
            if label not in label_list:
                label_list.append(str(label))
    return label_list

class Imagelists_VISDA_unl(object):
    def __init__(self, image_list, root="./data/multi/",
                 transform=None, transform2=None, target_transform=None, path=False):
        imgs, labels = make_dataset_fromlist(image_list)
        self.imgs = imgs
        self.labels = labels
        self.transform = transform
        self.transform2 = transform2
        self.target_transform = target_transform
        self.loader = pil_loader
        self.root = root
        self.path = path

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is
            class_index of the target class.
        """
        path = os.path.join(self.root, self.imgs[index])
        target = self.labels[index]
        img = self.loader(path)
        if self.transform is not None:
            img1 = self.transform(img)
        if self.transform2 is not None:
            img2 = self.transform2(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        if not self.path:
            return img1, img2, target
        else:
            return img1, img2, target, index, self.imgs[index]  ### the path of the selected images.

    def __len__(self):
        return len(self.imgs)

class Imagelists_VISDA_unl_paired(object):
    def __init__(self, image_list_s, image_list_t, root="./data/multi/",
                 transform=None, transform2=None, target_transform=None, path=False):
        imgs_s, labels_s = make_dataset_fromlist(image_list_s)
        imgs_t, labels_t = make_dataset_fromlist(image_list_t)
        self.imgs_s = imgs_s
        self.labels_s = labels_s
        self.imgs_t = imgs_t
        self.labels_t = labels_t
        self.transform = transform
        self.transform2 = transform2
        self.target_transform = target_transform
        self.loader = pil_loader
        self.root = root
        self.path = path
        self.num_cate = len(np.unique(labels_t))
        self.index_per_category = []
        for i in range(self.num_cate):
            self.index_per_category.append([])
            for j in range(len(self.labels_t)):
                if labels_t[j] == i:
                    self.index_per_category[i].append(j)


    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is
            class_index of the target class.
        """
        path_s = os.path.join(self.root, self.imgs_s[index])
        target_s = self.labels_s[index]
        img_s = self.loader(path_s)

        index_t = random.choice(self.index_per_category[target_s])
        path_t = os.path.join(self.root, self.imgs_t[index_t])
        target_t = self.labels_t[index_t]
        assert target_s == target_t
        img_t = self.loader(path_t)

        if self.transform is not None:
            img1_s = self.transform(img_s)
            img1_t = self.transform(img_t)
        if self.transform2 is not None:
            img2_s = self.transform2(img_s)
            img2_t = self.transform2(img_t)
        if self.target_transform is not None:
            target_s = self.target_transform(target_s)
            target_t = self.target_transform(target_t)
        if not self.path:
            return img1_s, img2_s, target_s, img1_t, img2_t, target_t
        else:
            return img1_s, img2_s, target_s, self.imgs_s[index], img1_t, img2_t, target_t, index, self.imgs_t[index_t]  ### the path of the selected images.

    def __len__(self):
        return len(self.imgs_s)