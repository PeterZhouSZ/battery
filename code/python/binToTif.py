import os, sys, time, psutil, math, cPickle
import argparse
import scipy.misc

from PIL import Image #install Pillow

optionparser = argparse.ArgumentParser(prog='Numpy to .tif convertor')
optionparser.add_argument('input', metavar = 'INPUT FILE')
optionparser.add_argument('-a', '--axis',  metavar = 'AXIS', help='x or y or z', default= 'x')
optionparser.add_argument('-o', '--output')


args = optionparser.parse_args(sys.argv[1:])
basename = os.path.basename(args.input)
filename = os.path.splitext(basename)[0]

if args.output == None:
	args.output = './' + filename + '/'
if args.output[-1] != '/' and args.output[-1] != '\\':
	args.output += '/'


if not os.path.isdir(args.output):
	os.makedirs(args.output)



print ("Reading " + args.input + " ...")
f = open(args.input, 'rb')
data = cPickle.load(f)
f.close()

print("Shape: " + str(data.shape))

def getSlice(i):
	if args.axis == 'x':
		return data[i,:,:]
	if args.axis == 'y':
		return data[:,i,:]
	if args.axis == 'z':
		return data[:,:,i]


N = data.shape[0]
if args.axis == 'y':
	N = data.shape[1]
elif args.axis == 'z':
	N = data.shape[2]


print ("Saving tiff ...")
for i in range(0,N):
	istr = "%04d" % (i,)		
	scipy.misc.imsave(args.output + filename + '_' + istr +  '.tiff', getSlice(i))
	if(i % (N/10) == 0):
		print(str(i*100.0 / N ) + '%')

print ('100%, done.')

