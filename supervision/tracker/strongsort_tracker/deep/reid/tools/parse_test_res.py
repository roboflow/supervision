"""
This script aims to automate the process of calculating average results
stored in the test.log files over multiple splits.

How to use:
For example, you have done evaluation over 20 splits on VIPeR, leading to
the following file structure

log/
    eval_viper/
        split_0/
            test.log-xxxx
        split_1/
            test.log-xxxx
        split_2/
            test.log-xxxx
    ...

You can run the following command in your terminal to get the average performance:
$ python tools/parse_test_res.py log/eval_viper
"""
import argparse
import glob
import os
import re
from collections import defaultdict

import numpy as np

from supervision.tracker.strongsort_tracker.deep.reid.torchreid.utils import (
    check_isfile,
    listdir_nohidden,
)


def parse_file(filepath, regex_mAP, regex_r1, regex_r5, regex_r10, regex_r20):
    results = {}

    with open(filepath, "r") as f:
        lines = f.readlines()

        for line in lines:
            line = line.strip()

            match_mAP = regex_mAP.search(line)
            if match_mAP:
                mAP = float(match_mAP.group(1))
                results["mAP"] = mAP

            match_r1 = regex_r1.search(line)
            if match_r1:
                r1 = float(match_r1.group(1))
                results["r1"] = r1

            match_r5 = regex_r5.search(line)
            if match_r5:
                r5 = float(match_r5.group(1))
                results["r5"] = r5

            match_r10 = regex_r10.search(line)
            if match_r10:
                r10 = float(match_r10.group(1))
                results["r10"] = r10

            match_r20 = regex_r20.search(line)
            if match_r20:
                r20 = float(match_r20.group(1))
                results["r20"] = r20

    return results


def main(args):
    regex_mAP = re.compile(r"mAP: ([\.\deE+-]+)%")
    regex_r1 = re.compile(r"Rank-1  : ([\.\deE+-]+)%")
    regex_r5 = re.compile(r"Rank-5  : ([\.\deE+-]+)%")
    regex_r10 = re.compile(r"Rank-10 : ([\.\deE+-]+)%")
    regex_r20 = re.compile(r"Rank-20 : ([\.\deE+-]+)%")

    final_res = defaultdict(list)

    directories = listdir_nohidden(args.directory, sort=True)
    num_dirs = len(directories)
    for directory in directories:
        fullpath = os.path.join(args.directory, directory)
        filepath = glob.glob(os.path.join(fullpath, "test.log*"))[0]
        check_isfile(filepath)
        print(f"Parsing {filepath}")
        res = parse_file(filepath, regex_mAP, regex_r1, regex_r5, regex_r10, regex_r20)
        for key, value in res.items():
            final_res[key].append(value)

    print("Finished parsing")
    print(f"The average results over {num_dirs} splits are shown below")

    for key, values in final_res.items():
        mean_val = np.mean(values)
        print(f"{key}: {mean_val:.1f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=str, help="Path to directory")
    args = parser.parse_args()
    main(args)
