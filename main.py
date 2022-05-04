import getopt
import os
import sys
import logging
import traceback
from raidionicsrads.compute import run_rads
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def main(argv):
    config_filename = None
    try:
        logging.basicConfig()
        logging.getLogger().setLevel(logging.DEBUG)
        opts, args = getopt.getopt(argv, "h:c:", ["Config="])
    except getopt.GetoptError:
        print('usage: main.py --Config <configuration_filepath>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('main.py --Config <configuration_filepath>')
            sys.exit()
        elif opt in ("-c", "--Config"):
            config_filename = arg

    try:
        run_rads(config_filename=config_filename)
    except Exception as e:
        logging.error('{}'.format(traceback.format_exc()))


if __name__ == "__main__":
    main(sys.argv[1:])

