import subprocess
import os
import time,sys

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from PIL import Image


parser = argparse.ArgumentParser(description='Convert tiff')
parser.add_argument('input', nargs='*', type=str,help='Input directory')

args = parser.parse_args()


root = Path(args.input[0]).resolve()
if(not root.is_dir()):
    print("Not a directory")
    exit()

print("Converting ", root)

fs = sorted(root.rglob('*.tiff'))

for index, f in enumerate(fs):   
    
    newStem = f.stem
    try:
        i = int(newStem)
        newStem = str(i).zfill(5)
    except ValueError as verr:
        pass
    except Eception as ex:
        pass

    outf = f.with_name(newStem + ".tiff")        
    I = plt.imread(f)    
    img = Image.fromarray(I)

    
    img.save(outf)
    f.unlink()
    print(index, " out of ",len(fs))



