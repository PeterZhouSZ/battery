import subprocess

import os

import time,sys


t0 = time.time()




root = '../../data/graphiteSections/dataset2/'
#f = open("output.csv",'w')

print("Starting tau calculation for:")
for subdir, dirs, files in os.walk(root):
    if(subdir == root): continue    
    print(subdir)

sys.stdout.flush()

for subdir, dirs, files in os.walk(root):
    if(subdir == root): continue
    ##print(subdir)

    args = subdir + "/ -t -dneg --step 128 -v -o out.csv --volExport "# --sub 16 --volExport"
    print(args)
    sys.stdout.flush()
    subprocess.call(
        '../bin/Release/batterytool.exe ' + args#,  stdout=f
    )

#f.close()


t1 = time.time()
sys.stdout.write('Total time(min): %g\n' %((t1 - t0)/60))