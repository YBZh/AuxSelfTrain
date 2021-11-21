from torchvision import models
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.autograd import Function


class GradReverse(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output * - ctx.alpha

        return output, None



class ZeroLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = 0.0

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output * 0.0

        return output, None

def grad_reverse(x, lambd=1.0):
    return GradReverse(lambd)(x)



def l2_norm(input):
    input_size = input.size()
    buffer = torch.pow(input, 2)

    normp = torch.sum(buffer, 1).add_(1e-10)
    norm = torch.sqrt(normp)

    _output = torch.div(input, norm.view(-1, 1).expand_as(input))

    output = _output.view(input_size)

    return output


class AlexNetBase(nn.Module):
    def __init__(self, pret=True):
        super(AlexNetBase, self).__init__()
        model_alexnet = models.alexnet(pretrained=pret)
        self.features = nn.Sequential(*list(model_alexnet.
                                            features._modules.values())[:])
        self.classifier = nn.Sequential()
        for i in range(6):
            self.classifier.add_module("classifier" + str(i),
                                       model_alexnet.classifier[i])
        self.__in_features = model_alexnet.classifier[6].in_features

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        # x = F.normalize(x)
        return x

    def output_num(self):
        return self.__in_features

class MNISTBase(nn.Module):
    def __init__(self, channels=32, num_classes=10, dataset='mnist'):
        super(MNISTBase, self).__init__()

        self.conv1 = nn.Conv2d(1, channels, kernel_size=5, stride=2, padding=2, bias=True)

        self.conv2 = nn.Conv2d(channels, channels, kernel_size=5, stride=2, padding=2, bias=True)
        self.conv3 = nn.Conv2d(channels, channels, kernel_size=5, stride=2, padding=2, bias=True)

        self.bn1 = nn.BatchNorm2d(channels)
        self.dropout = nn.Dropout(p=0.5)

        # self.linear = nn.Linear(512, num_classes)


    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.bn1(x)
        x = x.view(x.size(0), -1)
        # x = self.linear(x)

        return x

class VGGBase(nn.Module):
    def __init__(self, pret=True, no_pool=False):
        super(VGGBase, self).__init__()
        vgg16 = models.vgg16(pretrained=pret)
        self.classifier = nn.Sequential(*list(vgg16.classifier.
                                              _modules.values())[:-1])
        self.features = nn.Sequential(*list(vgg16.features.
                                            _modules.values())[:])
        # self.s = nn.Parameter(torch.FloatTensor([10]))

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 7 * 7 * 512)
        x = self.classifier(x)

        return x

class Mapping(nn.Module):
    def __init__(self, inc=4096):
        super(Mapping, self).__init__()
        self.fc = nn.Linear(inc, inc, bias=False)

    def forward(self, x):
        x = F.normalize(x)
        x_out = self.fc(x)
        return x_out


class Predictor(nn.Module):
    def __init__(self, num_class=64, inc=4096, multi_fc=False):
        super(Predictor, self).__init__()
        if multi_fc:
            self.fc = nn.Sequential(nn.Linear(inc, 512),
                                    nn.BatchNorm1d(512),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(512, num_class))
        else:
            self.fc = nn.Linear(inc, num_class)

    def forward(self, x):
        x_out = self.fc(x)
        return x_out


class Predictor_nograd(nn.Module):
    def __init__(self, num_class=64, inc=4096, multi_fc=False):
        super(Predictor_nograd, self).__init__()
        if multi_fc:
            self.fc = nn.Sequential(nn.Linear(inc, 512),
                                    nn.BatchNorm1d(512),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(512, num_class))
        else:
            self.fc = nn.Linear(inc, num_class)

    def forward(self, x):
        x = ZeroLayerF.apply(x, 0.0)
        x_out = self.fc(x)
        return x_out

class Predictor_MMD(nn.Module):
    def __init__(self, num_class=64, inc=4096, multi_fc=False):
        super(Predictor_MMD, self).__init__()
        if multi_fc:
            self.fc = nn.Sequential(nn.Linear(inc, 512),
                                    nn.BatchNorm1d(512),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(512, num_class))
        else:
            self.M = nn.Linear(inc, inc, bias=False)
            self.fc = nn.Linear(inc, num_class)

    def forward(self, x):
        mapped_x = self.M(x)
        x_out_mapped = self.fc(mapped_x)
        x_out = self.fc(x)
        return x_out_mapped, x_out



class Predictor_MME(nn.Module):
    def __init__(self, num_class=64, inc=4096, multi_fc=False, temp=0.05):
        super(Predictor_MME, self).__init__()
        self.multi_fc = multi_fc
        if multi_fc:
            self.fc1 = nn.Linear(inc, 512)
            self.fc2 = nn.Linear(512, num_class, bias=False)
        else:
            self.fc = nn.Linear(inc, num_class, bias=False)
        self.temp = temp

    def forward(self, x):
        if self.multi_fc:
            x = self.fc1(x)
            x = F.normalize(x)
            x_out = self.fc2(x) / self.temp
        else:
            x = F.normalize(x)
            x_out = self.fc(x) / self.temp
        return x_out



class Predictor_deep(nn.Module):
    def __init__(self, num_class=64, inc=4096, temp=0.05, normalize_temp=True):
        super(Predictor_deep, self).__init__()
        self.fc1 = nn.Linear(inc, 512)
        self.fc2 = nn.Linear(512, num_class, bias=False)
        self.num_class = num_class
        self.temp = temp
        self.normalize_temp = normalize_temp

    def forward(self, x, reverse=False, eta=0.1):
        x = self.fc1(x)
        if reverse:
            x = grad_reverse(x, eta)
        if self.normalize_temp:
            x = F.normalize(x)
            x_out = self.fc2(x) / self.temp
        else:
            x_out = self.fc2(x)
        return x_out




class Discriminator(nn.Module):
    def __init__(self, inc=4096):
        super(Discriminator, self).__init__()
        self.fc1_1 = nn.Linear(inc, 1)
        # self.fc1_1 = nn.Linear(inc, 512)
        # self.fc2_1 = nn.Linear(512, 1)


    def forward(self, x, lambd=1):
        x = GradReverse.apply(x, lambd)
        x_out = self.fc1_1(x)
        # x = F.relu(self.fc1_1(x))
        # x_out = self.fc2_1(x)

        return x_out

