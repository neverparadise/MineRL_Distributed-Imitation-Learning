import argparse
import os

parser = argparse.ArgumentParser(description='Argparse Tutorial')

parser.add_argument('--env', type=str, default="MineRLTreechop-v0", help='name of MineRLEnvironment')
parser.add_argument('--threshold', type=int, default=640, help='reward threshold for parsing demos')
parser.add_argument('--epochs', type=int, default=10, help='the number of training epochs')
parser.add_argument('--seq_len', type=int, default=400, help='sequence length for parsing demos')
parser.add_argument('--batch_size', type=int, default=32, help='batch_size for training')
parser.add_argument('--gamma', type=float, default=0.99, help='gamma for calculating return')
parser.add_argument('--model_name', type=str, default="pre_trained", help='pre_trained model name')

args = parser.parse_args()
config = {'env  ': args.env,
          'threshold': args.threshold,
          'epochs': args.epochs,
          'batch_size': args.batch_size,
          'seq_len': args.seq_len,
          'gamma': args.gamma,
          'model_name': args.model_name}


def print2(input1, **args):
    print(input1)
    print(type(args))
    for key, value in args.items():
        print(key, value)
print2(3, **config)

