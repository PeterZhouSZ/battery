from fipy import Grid3D
from fipy.variables.cellVariable import CellVariable
from fipy.variables.faceVariable import FaceVariable
from fipy.boundaryConditions.fixedValue import FixedValue
from fipy.terms.diffusionTerm import DiffusionTerm
import numpy as np
import os, time, psutil, argparse, sys
#from fipy.solvers.trilinos.linearGMRESSolver import LinearGMRESSolver
from fipy.solvers.scipy.linearGMRESSolver import LinearGMRESSolver
from fipy.tools import numerix

# Import other functions
import os, sys, time, psutil, math, cPickle, math
from numpy import where, shape, average, unique

'''---------------------------------------------------------------------------------------'''
t0 = time.time()

## Start memory monitor:
process = psutil.Process(os.getpid())


# Process command line arguments.

optionparser = argparse.ArgumentParser(prog='Tortuosity Calculator')
optionparser.add_argument('input', metavar = 'INPUT FILE')
optionparser.add_argument('-o', '--output')

args = optionparser.parse_args(sys.argv[1:])
basename = os.path.basename(args.input)
filename = os.path.splitext(basename)[0]

if args.output == None:
	args.output = './Output'
# 	directoryPath = os.path.dirname(args.input)
# 	directoryList = directoryPath.split('/')
# 	directoryList[-4] = 'Output'
# 	args.output = '/'.join(directoryList)

	
if not os.path.isdir(args.output):
		os.makedirs(args.output)


# Read Tomography Data
#print ("Reading Data from", filename, '...')
t1 = time.time()
inputFile = open(args.input, 'rb')
intensity = cPickle.load(inputFile)
t2 = time.time()
#print 'Done Reading!'
inputFile.close()

## Create the cell variable for storing the phase information
nx = intensity.shape[2]
ny = intensity.shape[1]
nz = intensity.shape[0]
dx = 0.37e-6
dy = 0.37e-6
dz = 0.37e-6

newIntensity = np.zeros(nx*ny*nz)

mesh = Grid3D(dx, dy, dz, nx, ny, nz)

print (intensity.shape)
print ("Setting phase...")

# flatten 3D array to 1D
newIntensity = intensity.flatten()

# set phase
t3 = time.time()
phase=CellVariable(mesh, value = 0.0)
phase.setValue(1 - newIntensity/255) #Thresholding step. Values less than 255 are assigned 0.

t4 = time.time()
print ("Phase Set!")

# Set diffusivity.
print ("Setting D...")
t5 = time.time()
D = 1.0e-10*phase + 1.0e-7*(1.0-phase)
t6 = time.time()
print ("D Set!")

# Calculate porosity.
porosity = 1.0 - (np.shape(where(phase > 0)[0])[0] / float(np.shape(phase)[0]))

print ('solving...')
t7 = time.time()

x,y,z = mesh.cellCenters

# Boundary conditions
valueHigh = 1.0
valueLow = 0.0

# Define solution variable
concentration = CellVariable(name="solution variable", mesh=mesh, value=0.0)	
concentration.constrain(valueLow, mesh.facesLeft)
concentration.constrain(valueHigh, mesh.facesRight)

#Define and solve steady state diffusion equation
DiffusionTerm(coeff=D).solve(var=concentration, solver=LinearGMRESSolver(tolerance=1.0e-6, iterations=int(1e9)))


#Post-process to calculate tortuosity in x direction
concentration[where(phase == 1.0)[0]] = 0
dp = average(phase[where(x == min(x))])
dc1 = average(concentration.value[where(x == min(x))])
dc2 = average(concentration.value[where(x == min(x))])*(1.-dp)

xtortuosity = dx*porosity/dc1/(dx*nx)/2
xtortuosity2 = dx*porosity/dc2/(dx*nx)/2
t8 = time.time()


# Write results to file
results = open(args.output + '/' + filename +'.tor', 'w')
results.write('porosity %g' % porosity)
results.write('\nmesh_size %g %g %g' % (dx, dy, dz))
results.write('\nbounds %g %g %g' % (nx*dx, ny*dy, nz*dz))
results.write('\nmesh_elements %g %g %g' % (nx, ny, nz))
# results.write('\nparticle_surface_area %g' % parser._shapes.values()[0].getSurfaceArea())
# results.write('\nparticle_volume %g' % parser._shapes.values()[0].getVolume())
results.write('\ntortuosity %g %g %g' % (xtortuosity, xtortuosity, xtortuosity))
results.write('\ntortuosity2 %g %g %g' % (xtortuosity2, xtortuosity2, xtortuosity2))
# results.write('\nconcentrationShape %d' % len(concentration))
# results.write('\ndc1dc2 %g %g' % (dc1, dc2))
results.write('\ndp %g' % dp)
results.write('\ntime_tot %g' %((t8 - t0)/60))
results.write('\ntime_read %g' %((t2 - t1)/60))
results.write('\ntime_set_phase %g' %((t4 - t3)/60))
results.write('\ntime_set_D %g' %((t6 - t5)/60))
results.write('\ntime_sol %g' %((t8 - t7)/60))
results.write('\nmem_percent %g' %process.memory_percent())
results.close()
print ("Done!!!")