import getopt
import os
import sys
import logging
import traceback
from raidionicsrads.compute import run_rads


def main(argv):
    config_filename = None
    try:
        logging.basicConfig()
        logging.getLogger().setLevel(logging.WARNING)
        opts, args = getopt.getopt(argv, "h:c:v:", ["Config=", "Verbose="])
    except getopt.GetoptError:
        print('usage: main.py --Config <configuration_filepath> (--Verbose <mode>)')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('main.py --Config <configuration_filepath> (--Verbose <mode>)')
            sys.exit()
        elif opt in ("-c", "--Config"):
            config_filename = arg
        elif opt in ("-v", "--Verbose"):
            if opt.lower() == 'debug':
                logging.getLogger().setLevel(logging.DEBUG)
            elif opt.lower() == 'info':
                logging.getLogger().setLevel(logging.INFO)
            elif opt.lower() == 'warning':
                logging.getLogger().setLevel(logging.WARNING)
            elif opt.lower() == 'error':
                logging.getLogger().setLevel(logging.ERROR)

    if not config_filename or not os.path.exists(config_filename):
        print('usage: main.py --Config <config_filepath> (--Verbose <mode>)')
        sys.exit(2)

    try:
        run_rads(config_filename=config_filename)
    except Exception as e:
        logging.error('{}'.format(traceback.format_exc()))


if __name__ == "__main__":
    main(sys.argv[1:])

