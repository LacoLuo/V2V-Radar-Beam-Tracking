import argparse
from process import train_process
from config import configurations

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
            description="Train the model.")
    parser.add_argument(
            "-t", "--trn_data_path", required=True, type=str,
            help="path of training data")
    parser.add_argument(
            "-v", "--val_data_path", required=True, type=str,
            help="path of validation data")
    parser.add_argument(
            "-s", "--store_model_path", required=True, type=str,
            help="path of checkpoint")
    parser.add_argument(
            "-l", "--load_model_path", default=None, type=str,
            help="path of pretrained model")
    parser.add_argument(
            "-n", "--normalize", action='store_true',
            help="use normalization")
    parser.add_argument(
            "-f", "--feature", choices=["RD", "RA", "DA"], required=True, type=str,
            help="output radar feature: RD = Range-Doppler, RA = Range-Angle, DA = Doppler-Angle")
    parser.add_argument(
            "-x", "--x_size", required=True, type=int,
            help="size of input samples")
    args = parser.parse_args()

    config = configurations()
    config.trn_data_path = args.trn_data_path
    config.val_data_path = args.val_data_path
    config.store_model_path = args.store_model_path
    config.load_model_path = args.load_model_path
    config.normalize = args.normalize
    config.feature = args.feature
    config.x_size = args.x_size
    print('config:\n', vars(config))

    train_process(config)
