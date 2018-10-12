import subprocess
import os
import time,sys

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description='Run tortuosity for .pos files')
parser.add_argument('input', nargs='*', type=str,help='Input directory')
args = parser.parse_args()

RESOLUTION = 256
N = -1

root = Path(args.input[0]).resolve()
if(not root.is_dir()):
    print("Not a directory")
    exit()

files = sorted(root.rglob('*.pos'))

if(os.name == 'nt'):
    toolPath = Path('../build/Release/batterytool.exe')    
    if(not toolPath.exists()):
        toolPath = Path('../build/Debug/batterytool.exe')
else:
    toolPath = Path('../build/batterytool')

solver = 'BICGSTABGPU'
directions = ['x','y','z','xneg','yneg','zneg']

files = files[0:N]

for index, d in enumerate(files):

    print("Volume", index, "out of", len(files))

    outputfile = d.parent / "measures.csv"    
    if(outputfile.is_file()):
        outputfile.unlink()

    cmdArg = [str(toolPath)]

    cmdArg += [
             str(d) ,
            '-t'        
        ]

    cmdArg += ["--solver", solver]

    cmdArg += ["--rad"]

    cmdArg += ["-dpos"]

    cmdArg += ["--sub", str(RESOLUTION)]

    cmdArg += ['-o', str(outputfile)]

    print (" ".join(cmdArg))
    subprocess.call(cmdArg)
    #exit()




