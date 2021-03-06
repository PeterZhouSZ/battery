import subprocess
import os
import time,sys

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import csv
from scipy import stats
from scipy.optimize import curve_fit

parser = argparse.ArgumentParser(description='Compare solvers.')
parser.add_argument('input', nargs='*', type=str,help='Dataset directory')
parser.add_argument('s',nargs='?', type=int,default=0, help='Start offset')
parser.add_argument('n',nargs='?', type=int,default=None, help='Number of dirs')
parser.add_argument('-o', dest='output',type=str, default='out.csv', help='Output file')
parser.add_argument('--plot',dest='plot', action='store_true')

args = parser.parse_args()



solvers = ['BICGSTABGPU']#,'MGGPU','BICGSTABCPU']
solverLabels = ['BiCGStab-GPU']#,'MultiGrid-GPU','BiCGStab-CPU',]
#sub = None
#direction = 'neg'
directions = ['x','y','z','xneg','yneg','zneg']
verbose = True
solver = 'MGGPU'
volExport = False
rad = True
outputToSameFile = False

#subs = [296, 320, 380, 420]
#solvers = ['BICGSTABGPU']
subs = [128]





if(args.plot):  

    plt.figure(figsize=(8, 6), dpi=80)
    
    solverTimes = {}

    for solver, label in zip(solvers,solverLabels):

        filename = Path(solver + '_' + args.output)
        if(not filename.exists()):
            continue
        f = open(str(filename),'r')         
        reader = csv.reader(f)        
        
        dims = []
        times = []
        indices = []
        taus = []
        dirs  = []
        
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
            
            if(float(row[i_tau]) < 1):
                continue

            voxels = int(row[i_dimx]) * int(row[i_dimy]) * int(row[i_dimz]);
            voxels = np.cbrt(voxels)
            dims.append(voxels)
            dirs.append(row[i_dir])
            times.append(float(row[i_t]))
            indices.append(i - 1)
            taus.append(float(row[i_tau]))

        solverTimes[solver] = times
        
        f.close()

        x = dims
        y = times

        x,y = zip(*sorted(zip(x,y)))
        x = np.asarray(x)
        y = np.asarray(y)

        slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
        line = slope*x+intercept
       

                
        plt.scatter(x,y, label=label)         
        plt.plot(x,line)



    #print(np.asarray(solverTimes['Eigen']) / np.asarray(solverTimes['MGGPU']) )
        
    #plt.xticks(np.arange(0,256, step=32))
    
    plt.yticks(list(np.arange(0,500, step=100)) + [30])    
    #plt.axvline(x=230,color='gray',linestyle='--')
    #plt.axvline(x=300,color='gray',linestyle='--')

    #plt.xlim(1,2)

    plt.xlabel('Dimension along single axis [voxels^(1/3)] ')
    plt.ylabel('Time [s]')
    plt.grid(True)
    plt.legend()
    
    if(os.name == 'nt'):
        plt.title('Tortuosity Calculation (NMC Dataset)\n(Windows 8.1, MSVC 2017, CUDA 9.2)\n(Intel Xeon E5-1630 v3 @ 3.7Ghz, 16GB RAM DDR4, GTX Titan X 12GB)')
    else:
        plt.title('Tortuosity Calculation (Ubuntu 18.04, GCC 7.3.0, CUDA 9.2)')
    plt.show()

    



    quit()


if(os.name == 'nt'):
    toolPath = Path('../build/Release/batterytool.exe')    
    if(not toolPath.exists()):
        toolPath = Path('../build/Debug/batterytool.exe')
else:
    toolPath = Path('../build/batterytool')
    


if(len(args.input) == 0):
    #dataset
    if(True):
        if(os.name == 'nt'):
            root = Path('D:/!battery/datasetNMC/')    
        else:
            root = Path('/media/vkrs/DATA/!battery/datasetNMC/')
    else:
        root = Path("../../data/graphite")
else:
    root = Path(args.input[0])
    
dirs = sorted(root.glob('*'))
dirs = [d for d in dirs if d.is_dir()]





if(args.n == None): 
    args.n = len(dirs) - args.s

print("Found {} directories, using {} to {}".format(len(dirs), args.s, args.s + args.n))

dirs = dirs[ args.s : (args.s + args.n)]


for solver in solvers:
    for subdir in dirs:    



        targ = [str(toolPath)]

        targ += [
            str(subdir),
            '-t'        
        ]

        #targ += [ '-d' + direction]
        if(verbose):
            targ += ['-v']

        

        if(rad):
            targ += ['--rad']

        #if(sub):
            #targ += ['--sub', str(sub)]

        if(volExport):
            targ += ['--volExport']
        
        
        if(outputToSameFile):
            targ += ['-o', args.output]
        else:
            targ += ['-o', solver + "_" + args.output]

        targ += ['--solver', solver]
        #targ += ['-o', solver + "_" + args.output]
        
        for direction in directions:       
            finalArgs = targ + [ '-d' + direction]

            for sub in subs:     
                finalArgs = targ + [ '-d' + direction] + ['--sub', str(sub)]        
                print (" ".join(finalArgs))
                subprocess.call(
                    finalArgs
                )

            if(len(subs) == 0):
                print (" ".join(finalArgs))
                subprocess.call(
                    finalArgs
                )

    
   
