import os
import pandas as pd
from glob import glob
from natsort import natsorted
from collections import OrderedDict
import json

def j2d(jpath):
    with open(jpath, encoding='utf-8') as j:
        d=json.load(j)
    return d

frozencfglist = natsorted(glob("../../analysis/files/results*.json"))

for cfg in frozencfglist:
    cmdpth = os.path.basename(os.path.splitext(cfg)[0])+"_download_ckpt.sh"
    print("Creating", cmdpth)
    d=j2d(cfg)
    with open(cmdpth, "w") as f:
        f.write("#!/usr/bin/env bash\n")
        f.write("\n")
        for s3ckpt in list(d['checkpoints'].keys()):
            localpath = s3ckpt.split("output/")[-1]
            f.write("mkdir -p {}\n".format(localpath))
            f.write("aws s3 sync {} {}\n".format(s3ckpt, localpath))
            f.write("\n")