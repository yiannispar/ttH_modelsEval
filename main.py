# main.py
#
# author: I. Paraskevas <ioannis.paraskevas@cern.ch>
# created: May, 2022

import argparse
import utils
import draw
import configparser
import timeit

parser = argparse.ArgumentParser()
parser.add_argument('-o', type=str, help="output dir", default="")
args = parser.parse_args()
config = configparser.ConfigParser()

def main():

    config.read("config.ini")

    output_dir = args.o
    utils.check_if_output_dir_exists(output_dir)

    draw.make_plots(output_dir,config)

if __name__ == "__main__":
    start = timeit.default_timer()
    main()
    end = timeit.default_timer()
    exec_time = end - start
    hours, rem = divmod(exec_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print("Execution time = {:0>2}:{:0>2}:{:0>2}".format(int(hours),int(minutes),int(seconds)))
