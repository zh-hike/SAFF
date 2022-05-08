import argparse
from Trainer import Trainer
import torch

parser = argparse.ArgumentParser("classifier")

parser.add_argument('--epochs', type=int, help="训练轮次", default=100)
parser.add_argument('--batch_size', type=int, help="批次大小", default=200)
parser.add_argument('--dataset', type=str, choices=['NWPU', 'UC', 'SAR'], default='NWPU')
parser.add_argument('--data_path', type=str, help="数据集所在路径", default='/users/zhhike/desktop/dataset/')
parser.add_argument('--model', type=str, choices=['vgg16'], default='vgg16')
parser.add_argument('--K', type=int, help="降为参数", default=784)
parser.add_argument('--ratio', type=float, help="训练比例", default=0.1)
parser.add_argument('--extract', action='store_true', help="是否进行特征提取")
parser.add_argument('--train', action='store_true', help="是否进行特征训练")

args = parser.parse_args()

if __name__ == "__main__":
    trainer = Trainer(args)

    with torch.no_grad():
        if args.extract:
            trainer.feature_extract()
        if args.train:
            trainer.train()
