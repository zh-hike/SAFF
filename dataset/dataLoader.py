from torchvision.datasets import ImageFolder
import os
from torch.utils.data import DataLoader
from dataset.util import get_default_aug


class DL:
    def __init__(self, args):
        path = args.data_path
        aug = get_default_aug(args.dataset)
        if args.dataset == 'NWPU':
            path = os.path.join(path, 'NWPU-RESISC45')
        elif args.dataset == 'UC':
            path = os.path.join(path, 'UCMerced_LandUse/Images')
        elif args.dataset == 'SAR':
            path = os.path.join(path, 'SAR/')

        data = ImageFolder(path, transform=aug)
        self.dl = DataLoader(data, batch_size=args.batch_size, shuffle=True, drop_last=True)
