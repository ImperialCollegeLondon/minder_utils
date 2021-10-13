import argparse
import sys
from .download.download import Downloader
from .formatting.formatting import Formatting
from .dataloader.dataloader import Dataloader
from .formatting.standardisation import standardise_activity_data
import numpy as np


def get_args(argv):
    parser = argparse.ArgumentParser(description='download and process DRI data')
    parser.add_argument('-download', type=lambda x: (str(x).lower() == 'true'),
                        default=True, help='download the data or not')
    parser.add_argument('-formatting', type=lambda x: (str(x).lower() == 'true'),
                        default=True, help='Categorize the data or not')
    parser.add_argument('-reload', type=lambda x: (str(x).lower() == 'true'),
                        default=True, help='Creating a new export request')
    return parser.parse_args(argv)


def process(args):
    if args.download:
        Downloader().export(args.reload)
    if args.formatting:
        return Formatting()


if __name__ == '__main__':
    args = get_args(sys.argv[1:])
    loader = process(args)
    dataloader = Dataloader(standardise_activity_data(loader.activity_data))
    a = dataloader.get_unlabelled_data()
    np.save('data/normalised/unlabelled.npy', a)
