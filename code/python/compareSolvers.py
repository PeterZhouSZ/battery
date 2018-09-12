import subprocess

import os

import time,sys


t0 = time.time()




#root = '../../data/LCO/dataset/'
root = 'D:/!battery/datasetNMC/'
#f = open("output.csv",'w')

#print("Starting tau calculation for:")
dataDirs = []
for subdir, dirs, files in os.walk(root):
    if(subdir == root): continue    
    dataDirs += [subdir]

#dataDirs = dataDirs[:16]    
sub = 64


for subdir in dataDirs:    
    ##print(subdir)
#--sub " + str(sub) + "
    args = subdir + "/ -t -dneg -v -o out_NMC_MGGPU.csv  --solver MGGPU"    
    print(args)
    subprocess.call(
        '../bin/Release/batterytool.exe ' + args
    )

    #args = subdir + "/ -t -dneg --step 128 -v -o out_compare.csv --solver Eigen"
    # --sub 16 --volExport"
    #print(args)
    #sys.stdout.flush()
    #subprocess.call(
    #    '../bin/Release/batterytool.exe ' + args
    #)

#f.close()


t1 = time.time()
sys.stdout.write('Total time(min): %g\n' %((t1 - t0)/60))