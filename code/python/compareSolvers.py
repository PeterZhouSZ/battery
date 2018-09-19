import subprocess
import os
import time,sys

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import csv

parser = argparse.ArgumentParser(description='Compare solvers.')
parser.add_argument('s',nargs='?', type=int,default=0, help='Start offset')
parser.add_argument('n',nargs='?', type=int,default=None, help='Number of dirs')
parser.add_argument('input', nargs='*', type=str,help='Dataset directory')
parser.add_argument('-o', dest='output',type=str, default='out.csv', help='Output file')
parser.add_argument('--plot',dest='plot', action='store_true')

args = parser.parse_args()



solvers = ['MGGPU','Eigen']
solverLabels = ['BiCGStab-GPU','BiCGStab-CPU']
sub = None
direction = 'x-'
verbose = True
solver = 'MGGPU'
volExport = False

subs = list(range(16,296,16))



if(args.plot):  

    plt.figure(figsize=(8, 6), dpi=80)
    
    solverTimes = {}

    for solver, label in zip(solvers,solverLabels):

        f = open(solver + '_' + args.output,'r') 
        reader = csv.reader(f)        
        
        dims = []
        times = []
        indices = []
        taus = []
        
        for i, row in enumerate(reader):            
            if(i == 0):                
                i_dimx = row.index('dimx')
                i_dimy = row.index('dimy')
                i_dimz = row.index('dimz')
                i_t = row.index('t')
                i_tau = row.index('tau')                
                i_dir = row.index('dir')
                i_porosity = row.index('porosity')
                continue
            
            for j, col in enumerate(row):
                row[j] = row[j].strip()
            
            voxels = int(row[i_dimx]) * int(row[i_dimy]) * int(row[i_dimz]);

            voxels = np.cbrt(voxels)
            dims.append(voxels)
            times.append(float(row[i_t]))
            indices.append(i - 1)
            taus.append(float(row[i_tau]))

        solverTimes[solver] = times
        
        f.close()
                
        plt.plot(dims, times, label=label)             


    #print(np.asarray(solverTimes['Eigen']) / np.asarray(solverTimes['MGGPU']) )
        
    plt.xticks(np.arange(0,256, step=32))
    plt.yticks(np.arange(0,256, step=20))
    
    plt.xlabel('Voxels^3')
    plt.ylabel('Time (s)')
    plt.grid(True)
    plt.legend()
    
    plt.title('Tortuosity Calculation (Ubuntu 18.04, GCC 7.3.0, CUDA 9.2)')
    plt.show()

    



    quit()


if(os.name == 'nt'):
    toolPath = Path('../build/Release/batterytool.exe')    
else:
    toolPath = Path('../build/batterytool')
    


if(len(args.input) == 0):
    if(os.name == 'nt'):
        root = 'D:/!battery/datasetNMC/'    
    else:
        root = Path('/media/vkrs/DATA/!battery/datasetNMC/')
else:
    root = Path(args.input[0])
    
dirs = sorted(root.glob('*'))






if(args.n == None): 
    args.n = len(dirs) - args.s

print("Found {} directories, using {} to {}".format(len(dirs), args.s, args.s + args.n))

dirs = dirs[ args.s : (args.s + args.n)]






for subdir in dirs:    



    targ = [str(toolPath)]

    targ += [
        str(subdir),
        '-t'        
    ]

    targ += [ '-d' + direction]
    if(verbose):
        targ += ['-v']

    

    

    if(sub):
        targ += ['--sub', str(sub)]

    if(volExport):
        targ += ['--volExport']
    
    


    #targ += ['--solver', solver]
    #targ += ['-o', solver + "_" + args.output]



    for _s in subs:
        for solver in solvers:
            finalArgs = targ + ['--sub', str(_s)] + ['--solver', solver] + ['-o', solver + "_" + args.output];
            
            print (" ".join(finalArgs))

            subprocess.call(
                finalArgs
            )

        
   
