import os 
import argparse
from process import test_process
from config import configurations

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
            description="Beam prediction.")
    parser.add_argument(
            "-e", "--test_data_path", required=True, type=str,
            help="path of testing data")
    parser.add_argument(
            "-l", "--load_model_path", required=True, type=str,
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
    config.test_data_path = args.test_data_path
    config.load_model_path = args.load_model_path
    config.normalize = args.normalize
    config.feature = args.feature
    config.x_size = args.x_size
    print('config:\n', vars(config))
    test_process(config)
