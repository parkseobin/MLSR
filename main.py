import os
import tensorflow as tf
import argparse
from train import SRTrainer
from models import IDNModel
from dataset import Dataset



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training MLSR')
    parser.add_argument('--lr-beta', type=float, default=1e-6)
    parser.add_argument('--lr-alpha', type=float, default=1e-5)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--patch-size', type=int, default=512)
    parser.add_argument('--gradient-number', type=int, default=5)
    parser.add_argument('--log-step', type=int, default=50)
    parser.add_argument('--train-iteration', type=int, default=10000)
    parser.add_argument('--validation-step', type=int, default=500)
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--validation-directory', type=str, default='Urban100/train')
    parser.add_argument('--train-directory', type=str, default='Urban100/validation')
    parser.add_argument('--param-restore-path', type=str, default='checkpoint_x2')
    parser.add_argument('--param-save-path', type=str, default=None)
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES']= args.gpu
    print('>> using gpu: {}\n\n'.format(args.gpu))

    dataset = Dataset(args)
    trainer = SRTrainer(dataset, IDNModel, args)
    trainer.train()