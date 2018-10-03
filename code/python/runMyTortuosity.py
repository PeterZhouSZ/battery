import subprocess
import os
import time,sys

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description='Run tortuosity')
parser.add_argument('input', nargs='*', type=str,help='Input directory')
args = parser.parse_args()

root = Path(args.input[0]).resolve()
if(not root.is_dir()):
    print("Not a directory")
    exit()

dirs = sorted(root.rglob('*'))
dirs = [
    d for d in dirs if (d.is_dir() and len(sorted((d.glob("*.tiff")))) > 0)
]



if(os.name == 'nt'):
    toolPath = Path('../build/Release/batterytool.exe')    
    if(not toolPath.exists()):
        toolPath = Path('../build/Debug/batterytool.exe')
else:
    toolPath = Path('../build/batterytool')

solver = 'BICGSTABGPU'
directions = ['x','y','z','xneg','yneg','zneg']

for index, d in enumerate(dirs):

    print("Volume", index, "out of", len(dirs))

    outputfile = d / "measures.csv"
    if(outputfile.is_file()):
        outputfile.unlink()

    cmdArg = [str(toolPath)]

    cmdArg += [
             str(d) ,
            '-t'        
        ]

    cmdArg += ["--solver", solver]

    cmdArg += ["--rad"]

    cmdArg += ["-dall"]

    cmdArg += ['-o', str(outputfile)]

    print (" ".join(cmdArg))
    subprocess.call(cmdArg)
    #exit()




