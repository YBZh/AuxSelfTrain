from model.resnet import resnet34, resnet50,resnet101
from model.basenet import AlexNetBase, VGGBase, Predictor, Predictor_MME, Discriminator, Predictor_MMD, MNISTBase
from model.wideresnet import Wide_ResNet, Wide_ResNet_fixmatch
import torch

def construct_model(args):
    if args.net == 'resnet34':
        G = resnet34(args.bottle_neck)
        inc = 512
    elif args.net == 'resnet50':
        G = resnet50(args.bottle_neck)
        if args.bottle_neck:
            inc = 512
        else:
            inc = 2048
    elif args.net == 'resnet101':
        G = resnet101()
        inc = 2048
    elif args.net == "alexnet":
        G = AlexNetBase()
        inc = 4096
        raise NotImplementedError
    elif args.net == "vgg":
        G = VGGBase()
        inc = 4096
    elif args.net == "3layer":
        G = MNISTBase()
        inc = 512
    elif args.net == "wideresnet":
        G = Wide_ResNet()
        inc = 128
    elif args.net == "wideresnet_fixmatch":
        G = Wide_ResNet_fixmatch(filter=args.filter)
        inc = 128
    else:
        raise ValueError('Model cannot be recognized.')

    if args.MME_classifier:
        F = Predictor_MME(num_class=args.num_class, inc=inc, multi_fc=args.multi_fc)
    else:
        F = Predictor(num_class=args.num_class, inc=inc, multi_fc=args.multi_fc)

    return G, F

def construct_prototype(args):
    if args.net == 'resnet34':
        inc = 512
    elif args.net == 'resnet50':
        if args.bottle_neck:
            inc = 512
        else:
            inc = 2048
    elif args.net == 'resnet101':
        inc = 2048
    elif args.net == "alexnet":
        inc = 4096
        raise NotImplementedError
    elif args.net == "vgg":
        inc = 4096
    elif args.net == "3layer":
        inc = 512
    elif args.net == "wideresnet":
        inc = 128
    elif args.net == "wideresnet_fixmatch":
        inc = 128
    else:
        raise ValueError('Model cannot be recognized.')

    prototype_classifier = torch.cuda.FloatTensor(args.num_class, inc).fill_(0)
    prototype_classifier.requires_grad_(True)

    return prototype_classifier

def construct_discriminator(args):
    if args.net == 'resnet34':
        inc = 512
    elif args.net == 'resnet50':
        if args.bottle_neck:
            inc = 512
        else:
            inc = 2048
    elif args.net == 'resnet101':
        inc = 2048
    elif args.net == "alexnet":
        inc = 4096
        raise NotImplementedError
    elif args.net == "vgg":
        inc = 4096
    elif args.net == "wideresnet":
        inc = 128
    elif args.net == "wideresnet_fixmatch":
        inc = 128
    else:
        raise ValueError('Model cannot be recognized.')

    D = Discriminator(inc)

    return D