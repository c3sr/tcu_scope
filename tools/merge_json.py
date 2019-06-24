#! /usr/bin/env python3

import json
import glob
import sys


def main():
    if len(sys.argv) != 2:
        print("usage: merge_json directory")
        sys.exit(1)

    dir = sys.argv[1]
    accum = None
    files = glob.glob(dir + "/*json")
    for filepath in files:
        try:
            with open(filepath, "rb") as f:
                j = json.loads(f.read().decode("utf-8"))
                if accum == None:
                    accum = j
                else:
                    accum["benchmarks"] += j["benchmarks"]
        except:
            None

    with open(dir + ".json", 'w') as fp:
        json.dump(accum, fp, indent=4)


if __name__ == "__main__":
    main()
