### Generation of Ellipsoids

The prograsms in this folder create volumes (stored as a series of .tiff images)
that contain ellipsoids with a controllable March-Dollase orientation distribution.

The most important files are GenerateDataset_[1-5] which generate volumes with various orientation distributions,
porosity (controlled via volume fraction), and surface areas.

The algorithm (and much of the code) is based off of the algorithm discussed in [Ghossein and Lévesque 2013], so some of the comments are written in French.

---------------

Elias Ghossein and Martin Lévesque. 2013. Random generation of periodic hard ellipsoids based on molecular dynamics: A computationally-efficient algorithm. J. Comput. Phys. 253 (November 2013), 471-490. DOI=http://dx.doi.org/10.1016/j.jcp.2013.07.004

