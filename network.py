from torchvision.models import vgg16
import torch.nn as nn
import torch


def get_conv(start, end, model='vgg16'):
    conv1, conv2, conv3 = None, None, None
    if model == 'vgg16':
        net = vgg16(pretrained=True)
        return net.features[start:end]

    return None


class BackBone(nn.Module):
    def __init__(self, in_features, out_features):
        super(BackBone, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_features, out_features, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
        )

    def forward(self, x):
        return self.net(x)


class Feature_Extraction(nn.Module):
    def __init__(self, args):
        super(Feature_Extraction, self).__init__()
        self.layer1 = get_conv(0, 19, args.model)
        self.layer2 = get_conv(19, 26, args.model)
        self.layer3 = get_conv(26, 31, args.model)

        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=4, padding=0)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

    def forward(self, imgs):
        x1 = self.layer1(imgs)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        new_x1 = x3
        new_x2 = self.maxpool2(x2)
        new_x3 = self.maxpool4(x1)
        x = torch.cat([new_x1, new_x2, new_x3], dim=1)

        return x


class SAFF(nn.Module):
    def __init__(self, a=0.5, b=2, sigma=0.0001):
        super(SAFF, self).__init__()
        self.a = a
        self.b = b
        self.sigma = sigma

    def forward(self, x):
        """
        :param x: (n, c, h, w)
        :return:
        """
        n, K, h, w = x.shape
        S = x.sum(dim=1)  # n,h,w
        z = torch.sum(S ** self.a, dim=[1, 2])
        z = (z ** (1 / self.a)).view(n, 1, 1)
        S = (S / z) ** (1 / self.b)
        S = S.unsqueeze(1)
        new_x = (x * S).sum(dim=[2, 3])
        omg = (x > 0).sum(dim=[2, 3]) / (256 ** 2)
        omg_sum = omg.sum(dim=1).unsqueeze(1)
        omg = (K * self.sigma + omg_sum) / (self.sigma + omg)
        omg = torch.log(omg)
        x = omg * new_x
        return x


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.feature_extract = Feature_Extraction(args)
        self.saff = SAFF()

    def forward(self, img):

        x = self.feature_extract(img)
        x = self.saff(x)
        return x
