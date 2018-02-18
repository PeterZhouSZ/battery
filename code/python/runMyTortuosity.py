import subprocess

import os

import time,sys


t0 = time.time()




root = '../../data/graphiteSections/dataset/'
f = open("output.csv",'w')

for subdir, dirs, files in os.walk(root):
    if(subdir == root): continue
    ##print(subdir)

    args = subdir + "/ -t -dneg --step 128 -v -o out.csv"#--sub 16"
    print(args)

    subprocess.call(
        '../bin/Release/batterytool.exe ' + args#,  stdout=f
    )

f.close()


t1 = time.time()
sys.stdout.write('Total time(min): %g\n' %((t1 - t0)/60))