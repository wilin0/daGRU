import argparse
import warnings
warnings.filterwarnings('ignore')
from exp import Exp


def create_parser():
    parser = argparse.ArgumentParser()
    # Set-up parameters
    parser.add_argument('--device', default='cuda', type=str, help='Name of device to use for tensor computations (cuda/cpu)')
    parser.add_argument('--res_dir', default='./results', type=str)
    parser.add_argument('--ex_name', default='Debug', type=str)
    parser.add_argument('--use_gpu', default=True, type=bool)
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--pretrained', default=True, type=bool)

    # dataset parameters
    parser.add_argument('--batch_size', default=16, type=int, help='Batch size')
    parser.add_argument('--val_batch_size', default=16, type=int, help='Batch size')
    parser.add_argument('--data_root', default='./data/')
    parser.add_argument('--dataname', default='mmnist', choices=['mmnist', 'taxibj'])
    parser.add_argument('--num_workers', default=8, type=int)

    # model parameters
    parser.add_argument('--in_shape', default=[20, 1, 64, 64], type=int, nargs='*') # [20, 1, 64, 64] for mmnist, [4, 2, 32, 32] for taxibj
    parser.add_argument('--num_hidden', default=[64, 64, 64, 64], type=int, nargs='*')
    parser.add_argument('--filter_size', default=5, type=int)
    parser.add_argument('--stride', default=1, type=int)
    parser.add_argument('--time_stride', default=3, type=int)
    parser.add_argument('--input_length', default=10, type=int)
    parser.add_argument('--num_layers', default=4, type=int)
    parser.add_argument('--patch_size', default=4, type=int)

    # Training parameters
    parser.add_argument('--epochs', default=51, type=int)
    parser.add_argument('--log_step', default=1, type=int)
    parser.add_argument('--lr', default=0.01, type=float, help='Learning rate')
    return parser


if __name__ == '__main__':
    args = create_parser().parse_args()
    config = args.__dict__

    exp = Exp(args)
    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>  start <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
    exp.train(args)
    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>> testing <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
    mse = exp.test(args)