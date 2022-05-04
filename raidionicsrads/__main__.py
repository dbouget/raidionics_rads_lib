import argparse
import os
import sys
import traceback
import logging
from raidionicsrads.compute import run_rads
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def path(string):
    if os.path.exists(string):
        return string
    else:
        sys.exit(f'File not found: {string}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', metavar='config', type=path, help='Path to the configuration file (*.ini)')

    argsin = sys.argv[1:]
    args = parser.parse_args(argsin)
    config_filename = args.config

    try:
        run_rads(config_filename=config_filename)
    except Exception as e:
        logging.error('{}'.format(traceback.format_exc()))


if __name__ == "__main__":
    main()

