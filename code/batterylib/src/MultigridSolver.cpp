#include "MultigridSolver.h"

using namespace blib;

template class MultigridSolver<float>;
template class MultigridSolver<double>;


#include "JacobiSolver.h"
#include <cassert>
#include "../../batteryviz/src/utility/IOUtility.h"
#include <numeric>

#include <Eigen/SparseLU>
#include <Eigen/IterativeLinearSolvers>


template <typename T>
void addDebugChannel(Volume & vol, const Eigen::Matrix<T, Eigen::Dynamic, 1> & v, ivec3 dim, const std::string & name, int levnum = -1, bool normalize = false) {
		
	auto & c = vol.getChannel(vol.addChannel(dim, TYPE_FLOAT));
	
	Eigen::Matrix<T, Eigen::Dynamic, 1> tmp = v;
	if (normalize) {
		tmp = v.cwiseAbs();
		/*auto maxc = tmp.maxCoeff();
		auto minc = tmp.minCoeff();
		tmp = (tmp.array() - minc) / (maxc - minc);*/
		tmp *= T(10);
	}

	memcpy(c.getCurrentPtr().getCPU(), tmp.data(), dim.x * dim.z* dim.y * sizeof(float));
	c.getCurrentPtr().commit();

	char buf[24]; itoa(levnum, buf, 10);
	c.setName(name + buf);


}


template <typename T>
MultigridSolver<T>::MultigridSolver(bool verbose)
	: _verbose(verbose)
{
	
	_restrictOp = std::array<T, 27>{
		0,1,0 , 1,2,1 , 0,1,0,
		1,2,1 , 2,4,2 , 1,2,1,
		0,1,0 , 1,2,1 , 0,1,0
	};

	_interpOp = std::array<T, 27>{
		1,1,1, 2,4,2, 1,2,1,
		2,4,2, 4,8,4, 2,4,2,
		1,2,1, 2,4,2, 1,2,1
	};

	
}

template <typename T, typename K>
void conv3D(
	K * src, ivec3 srcDim,
	K * dest, ivec3 destDim,
	const std::array<T,27> & kernel
) {
	
	auto srcStride = { 1, srcDim.x, srcDim.x * srcDim.y };

	for (auto z = 0; z < destDim.z; z++) {
		for (auto y = 0; y < destDim.y; y++) {
			for (auto x = 0; x < destDim.x; x++) {

				auto destI = linearIndex(destDim, { x,y,z });			

				K val = K(0);
				T wsum = 0.0;
				
				for (auto k = 0; k < 3; k++) {
					for (auto j = 0; j < 3; j++) {
						for (auto i = 0; i < 3; i++) {				

							ivec3 srcPos = 2 * ivec3(x, y, z) + ivec3(i - 1, j - 1, k - 1);
							if (!isValidPos(srcDim, srcPos)) continue;
						
							const T & w = kernel[i + 3 * j + k * 9];
							if (w == T(0)) continue;

							auto srcI = linearIndex(srcDim, srcPos);
							val += w * src[srcI];
							wsum += w;
						}
					}
				}

				if (wsum > T(0))
					dest[destI] = val / wsum;
				else
					dest[destI] = T(0);

			}
		}
	}
}

template <typename T>
void restriction(
	T * src, ivec3 srcDim,
	T * dest, ivec3 destDim 	
) {
	assert(srcDim.x == destDim.x * 2);
	assert(srcDim.y == destDim.y * 2);
	assert(srcDim.z == destDim.z * 2);

	for (auto z = 0; z < destDim.z; z++) {
		for (auto y = 0; y < destDim.y; y++) {
			for (auto x = 0; x < destDim.x; x++) {
				ivec3 ipos = ivec3(x, y, z);
				auto srcI = linearIndex(srcDim, 2 * ipos);
				auto destI = linearIndex(destDim, ipos);
				dest[destI] = src[srcI];
			}
		}
	}


}


template <typename T>
void interpolation(
	T * src, ivec3 srcDim,
	T * dest, ivec3 destDim //higher	
) {

	

	for (auto z = 0; z < destDim.z; z++) {
		for (auto y = 0; y < destDim.y; y++) {
			for (auto x = 0; x < destDim.x; x++) {
				ivec3 ipos = { x,y,z };
				ivec3 iposSrc = ipos / 2;
				auto destI = linearIndex(destDim, ipos);
				auto srcI = linearIndex(srcDim, iposSrc);

				ivec3 srcStride = { 1, srcDim.x, srcDim.x * srcDim.y };
				if (x == destDim.x - 1) srcStride[0] = 0;
				if (y == destDim.y - 1) srcStride[1] = 0;
				if (z == destDim.z - 1) srcStride[2] = 0;

				const ivec3 mods = { x % 2, y % 2, z % 2 };
				const int modSum = mods.x + mods.y + mods.z;


				T val = T(0);
				
				//Colocated				
				if (modSum == 0) {					
					val = src[srcI];
				}
				//Edge interpolation
				else if (modSum == 1) {
					int index = (mods[0] == 1) ? 0 : ((mods[1] == 1) ? 1 : 2);
					
					ivec3 dx = ivec3(0);
					dx[index] = 1;

					//assert(srcI + srcStride[index] == linearIndex(srcDim, iposSrc + dx));

					val = (
						src[srcI] +
						src[srcI + srcStride[index]]
						) * T(1.0 / 2.0);


				}
				//Face interpolation
				else if (modSum == 2) {
					int coIndex = (mods[0] == 0) ? 0 : ((mods[1] == 0) ? 1 : 2);
					int i0 = (coIndex + 1) % 3;
					int i1 = (coIndex + 2) % 3;		

					val = (
						src[srcI] +
						src[srcI + srcStride[i1]] +
						src[srcI + srcStride[i0]] +
						src[srcI + srcStride[i0] + srcStride[i1]]
						) * T(1.0 / 4.0);
				}
				//Cube interpolation  //modSum == 3
				else {
					val = (
						src[srcI] +
						src[srcI + srcStride[0]] +
						src[srcI + srcStride[1]] +
						src[srcI + srcStride[0] + srcStride[1]] +
						src[srcI + srcStride[2]] +
						src[srcI + srcStride[1] + srcStride[2]] +
						src[srcI + srcStride[0] + srcStride[2]] +
						src[srcI + srcStride[0] + srcStride[1] + srcStride[2]]
						) * T(1.0 / 8.0);					
				}


				dest[destI] = val;				

			}
		}
	}

}

template <typename T>
void interp3D(
	T * src, ivec3 srcDim,
	T * dest, ivec3 destDim //higher	
) {
		

	for (auto z = 0; z < destDim.z; z++) {
		for (auto y = 0; y < destDim.y; y++) {
			for (auto x = 0; x < destDim.x; x++) {
				auto destI = linearIndex(destDim, { x,y,z });

				ivec3 srcStride = { 1, srcDim.x, srcDim.x * srcDim.y };
				if (x == destDim.x-1) srcStride[0] = 0;
				if (y == destDim.y-1) srcStride[1] = 0;
				if (z == destDim.z-1) srcStride[2] = 0;


				vec3 sPos = vec3(x, y, z) * 0.5f;
				ivec3 ipos = ivec3(sPos);
				vec3 fract = glm::fract(sPos);

				auto srcI = linearIndex(srcDim, ipos);

				assert(srcI >= 0 && srcI < srcDim.x * srcDim.x * srcDim.y);
							
				T vals[8] = {
					src[srcI],														//0
					src[srcI + srcStride[0]],										//1
					src[srcI +					srcStride[1]],						//2
					src[srcI + srcStride[0] +	srcStride[1]],						//3
					src[srcI									+ srcStride[2]],	//4
					src[srcI + srcStride[0]						+ srcStride[2]],	//5
					src[srcI +					srcStride[1]	+ srcStride[2]],	//6
					src[srcI + srcStride[0] +	srcStride[1]	+ srcStride[2]]		//7
				};

				T x0 = glm::mix(
					glm::mix(vals[0], vals[2], fract.y), //0,0,0 vs 0,1,0
					glm::mix(vals[4], vals[6], fract.y), //0,0,1 vs 0,1,1
					fract.z
				);
				T x1 = glm::mix(
					glm::mix(vals[1], vals[3], fract.y), //1,0,0 vs 1,1,0
					glm::mix(vals[5], vals[7], fract.y), //1,0,1 vs 1,1,1
					fract.z
				);

				float val = glm::mix(x0, x1, fract.x);

				dest[destI] = val;

			}
		}
	}



}

template <typename T>
bool MultigridSolver<T>::prepare(
	Volume & v, 
	const uchar * D, ivec3 origDim, Dir dir, T d0, T d1,
	uint levels
){


	

	bool res = true;

	_lv = levels;

	//Prepare lin. system for all levels
	_A.resize(_lv);
	_f.resize(_lv);
	_x.resize(_lv);
	_tmpx.resize(_lv);
	_r.resize(_lv);
	_dims.resize(_lv);

	//Generate D volume for smaller resolutions
	std::vector<std::vector<float>> Dlevels(_lv);
	std::vector<std::vector<float>> Dlevels_interp(_lv);
	

	const size_t origTotal = origDim.x * origDim.y * origDim.z;

	size_t cntd0 = 0;
	for (auto i = 0; i < origTotal; i++) {
		if (D[i] == 0) cntd0++;
	}
	_porosity = cntd0 / T(origTotal);

	//Generate first level of D
	{
		Dlevels.front().resize(origTotal);
		_dims.front() = origDim;

		for (auto i = 0; i < origTotal; i++) {
			Dlevels.front()[i] = (D[i] == 0) ? d0 : d1;
		}

		res &= prepareAtLevel(Dlevels[0].data(), _dims[0], dir, 0);


		Dlevels_interp[0].resize(origTotal, 0.0f);
	}

	//Generate rest of levels, based on higher resolution
	for (uint i = 1; i < _lv; i++) {
		const int divFactor = (1 << i);
		const ivec3 dim = origDim / divFactor;
		const size_t total = dim.x * dim.y * dim.z;

		_dims[i] = dim;
		Dlevels[i].resize(total);
		conv3D(Dlevels[i - 1].data(), _dims[i - 1], Dlevels[i].data(), _dims[i], _restrictOp);
		//restriction(Dlevels[i - 1].data(), _dims[i - 1], Dlevels[i].data(), _dims[i]);
		
		Dlevels_interp[i].resize(total, 0.0f);
		

		res &= prepareAtLevel(Dlevels[i].data(), _dims[i], dir, i);
	}




	for (uint i = 0; i < _lv; i++) {

		//addDebugChannel<T>(v, Dlevels[i].data(), _dims[i], "D ", i, false);

		

		/*if(i < _lv - 1){
			interp3D(Dlevels[i + 1].data(), _dims[i + 1], Dlevels_interp[i].data(), _dims[i]);		
			auto id = v.addChannel(_dims[i], TYPE_FLOAT);
			auto & c = v.getChannel(id);
			memcpy(c.getCurrentPtr().getCPU(), Dlevels_interp[i].data(), _dims[i].x * _dims[i].y * _dims[i].z * sizeof(float));
			c.getCurrentPtr().commit();
		}*/



	}

	if (_verbose) {
		std::cout << "Prepared multigrid with L = " << _lv << " res: " << origDim.x << ", "<< origDim.y << ", " << origDim.z << std::endl;
	}
	
	return res;
}

template <typename T>
bool MultigridSolver<T>::prepareAtLevel(
	const float * D, ivec3 dim, Dir dir,
	uint level
)
{
	using vec3 = glm::tvec3<T, glm::highp>;
		
	assert(dim.x > 0 && dim.y > 0 && dim.z > 0);


	uint dirPrimary = getDirIndex(dir);
	uint dirSecondary[2] = { (dirPrimary + 1) % 3, (dirPrimary + 2) % 3 };
	const T highConc = 1.0f;
	const T lowConc = 0.0f;
	const T concetrationBegin = (getDirSgn(dir) == 1) ? highConc : lowConc;
	const T concetrationEnd = (getDirSgn(dir) == 1) ? lowConc : highConc;

	size_t N = dim.x * dim.y * dim.z;
	size_t M = N;

	


	
	//D getter
	const auto getD = [&dim,D](ivec3 pos) {			
		return D[linearIndex(dim, pos)];
	};

	const auto sample = [&getD, dim, D](ivec3 pos, Dir dir) {
		const int k = getDirIndex(dir);
		assert(pos.x >= 0 && pos.y >= 0 && pos.z >= 0);
		assert(pos.x < dim.x && pos.y < dim.y && pos.z < dim.z);

		ivec3 newPos = pos;
		int sgn = getDirSgn(dir);
		if (pos[k] + sgn < 0 || pos[k] + sgn >= dim[k]) {
			//do nothing
		}
		else {
			newPos[k] += getDirSgn(dir);
		}
		return getD(newPos);
	};	


	//Default spacing
	const vec3 h_general = { 1.0f / (dim.x + 1), 1.0f / (dim.y + 1), 1.0f / (dim.z + 1) };

	size_t d0Count = 0;
	size_t d1Count = 0;

		
	
	_f[level].resize(M);
	_x[level].resize(N);
	_A[level].resize(M, N);
	_A[level].reserve(Eigen::VectorXi::Constant(N, 7));
	


	//Build matrix
	for (auto z = 0; z < dim.z; z++) {
		for (auto y = 0; y < dim.y; y++) {
			for (auto x = 0; x < dim.x; x++) {

				const ivec3 ipos = { x,y,z };
				auto i = linearIndex(dim, ipos);

				vec3 hneg = h_general;
				vec3 hpos = h_general;

				//Adjust spacing on boundary
				for (auto k = 0; k < 3; k++) {
					if (ipos[k] == 0)
						hneg[k] *= 1.0 / sqrt(2);//0.5;					
					else if (ipos[k] >= dim[k] - 1)
						hpos[k] *= 1.0 / sqrt(2); //0.5;
				}
				const vec3 invHneg = vec3(1.0) / hneg;
				vec3 invHneg2 = invHneg*invHneg;
				const vec3 invHpos = vec3(1.0) / hpos;
				vec3 invHpos2 = invHpos*invHpos;

				T bval = 0.0f;
				T vals[7] = { 0.0f,0.0f ,0.0f ,0.0f ,0.0f ,0.0f ,0.0f };
				size_t colId[7];
				colId[0] = i - dim.y*dim.x;
				colId[1] = i - dim.x;
				colId[2] = i - 1;
				colId[3] = i;
				colId[4] = i + 1;
				colId[5] = i + dim.x;
				colId[6] = i + dim.y*dim.x;

				T Di = getD(ipos);
				


				auto Dvec = vec3(Di);

				auto Dneg = (vec3(
					sample(ipos, X_NEG),
					sample(ipos, Y_NEG),
					sample(ipos, Z_NEG)
				) + vec3(Dvec)) * T(0.5) * invHneg2;

				auto Dpos = (vec3(
					sample(ipos, X_POS),
					sample(ipos, Y_POS),
					sample(ipos, Z_POS)
				) + vec3(Dvec)) * T(0.5) * invHpos2;

				T diagVal = 0.0f;

				//(is first order accurate von neumann for secondary axes)
				for (auto k = 0; k < 3; k++) {
					if (ipos[k] > 0)
						diagVal += -Dneg[k];
					if (ipos[k] < dim[k] - 1)
						diagVal += -Dpos[k];
				}

				//Boundary conditions
				if (ipos[dirPrimary] == 0) {
					bval = -Dneg[dirPrimary] * concetrationBegin;
					diagVal -= Dneg[dirPrimary];

				}
				else if (ipos[dirPrimary] == dim[dirPrimary] - 1) {
					bval = -Dpos[dirPrimary] * concetrationEnd;
					diagVal -= Dpos[dirPrimary];
				}


				if (z > 0) vals[0] = Dneg.z;
				if (y > 0) vals[1] = Dneg.y;
				if (x > 0) vals[2] = Dneg.x;
				vals[3] = diagVal;
				if (x < dim.x - 1) vals[4] = Dpos.x;
				if (y < dim.y - 1) vals[5] = Dpos.y;
				if (z < dim.z - 1) vals[6] = Dpos.z;

				_f[level][i] = bval;

				//initial guess
				if (getDirSgn(dir) == 1)
					_x[level][i] = 1.0f - (ipos[dirPrimary] / T(dim[dirPrimary] + 1));
				else
					_x[level][i] = (ipos[dirPrimary] / T(dim[dirPrimary] + 1));


				for (auto k = 0; k < 7; k++) {
					if (colId[k] < 0 || colId[k] >= N || vals[k] == 0.0f) continue;
					_A[level].insert(i, colId[k]) = vals[k];
				}

			}
		}
	}


	//_porosity = T(d0Count) / T(d0Count + d1Count);

	_A[level].makeCompressed();

	//_solver.compute(_A);

	return true;
}






template <typename T>
T MultigridSolver<T>::solve(Volume &vol, T tolerance, size_t maxIterations)
{

	//_verbose = false;

	std::vector<Vector> & r = _r;		
	std::vector<Vector> & f = _f;
	std::vector<Vector> & v = _x;
	std::vector<Vector> & tmpx = _tmpx;
	std::vector<SparseMat> & A = _A;

	for (auto i = 0; i < _lv; i++) {
		r[i].resize(v[i].size());
		tmpx[i].resize(v[i].size());
	}


	const int preN = 8;
	const int postN = 8;
	const int lastLevel = _lv - 1;

	Eigen::SparseLU<SparseMat> exactSolver;
	//Eigen::BiCGSTAB<SparseMat> exactSolver;
	//exactSolver.setTolerance(tolerance / 2.0f);

	exactSolver.compute(A[lastLevel]);
	
		
	auto tabs = [](int n){
		return std::string(n, '\t');
	};

	
	//maxIterations V cycle
	for (auto k = 0; k < maxIterations; k++) {
		
		/*		
			SMOOTHING DISABLED
		*/
		
		
		for (auto i = 0; i < lastLevel; i++) {

			//Initial guess is zero for all except first level
			if (i > 0) v[i].setZero();
			
			if (_verbose) {
				std::cout << tabs(i) << "Level: " << i << " dim: " << _dims[i].x << ", " << _dims[i].y << ", " << _dims[i].z << std::endl;
				std::cout << tabs(i) << "Pre-smoothing: ";
			}

			//Pre smoothing, v[i] is initial guess & result
			//Residual saved in r[i]
			T err = solveJacobi(A[i], f[i], v[i], tmpx[i], r[i], tolerance, preN, false);
			if(preN == 0) r[i] = f[i] - A[i]*v[i];
			
			if (_verbose) {
				std::cout << err << std::endl;				

				addDebugChannel(vol, _r[i], _dims[i], "Pre ", i, true);				
			}

			
			if (_verbose) {
				std::cout << tabs(i) << "Restriction r[" << i << "]" << " to " << "f["<< (i+1) << "]" << std::endl;
			}
			//Restrict residual r[i] to f[i+1] and use it as f[i+1] in next iteration
			conv3D(r[i].data(), _dims[i], f[i + 1].data(), _dims[i + 1], _restrictOp);
			//restriction(r[i].data(), _dims[i], f[i + 1].data(), _dims[i + 1]);
						
		}

		if (_verbose) {
			std::cout << tabs(lastLevel) << "Exact solve at Level " << lastLevel << std::endl;
		}		
		//Direct solver
		v[lastLevel] = exactSolver.solve(f[lastLevel]);

		//Iterative solver
		//v[lastLevel].setZero();
		//v[lastLevel] = exactSolver.solveWithGuess(f[lastLevel], v[lastLevel]);

		//Iterative Jacobi
		//T errExact = solveJacobi(A[lastLevel], f[lastLevel], v[lastLevel], tmpx[lastLevel], r[lastLevel], 1e-6, 512, true);
		//std::cout << "exact error" << errExact << std::endl;

		if (_verbose) {
			addDebugChannel(vol, v[lastLevel], _dims[lastLevel], "Exact solution", lastLevel, true);
		}

		//Up step
		for (int i = lastLevel - 1; i >= 0; i--) {

			if (_verbose) {
				std::cout << tabs(i) << "Interpolation from v[" << (i+1) << "]" << " to " << "Iv[" << (i+1) << "]" << std::endl;
			}
			//Interpolate v[i+1] to temp[i]
			interpolation<T>(v[i + 1].data(), _dims[i + 1], tmpx[i].data(), _dims[i]);

			if (_verbose) {
				std::cout << tabs(i) << "Correction v[" << i << "]" << " += " << "Iv[" << (i+1) << "]" << std::endl;
			}
			//Add temp to v[i]
			v[i] += tmpx[i];

			if (_verbose) {				
				std::cout << tabs(i) << "Post-smoothing: ";
			}

			//Post smoothing, v[i] is initial guess & result		
			T err = solveJacobi(A[i], f[i], v[i], tmpx[i], r[i], tolerance, postN, false);
			if (postN == 0) r[i] = f[i] - A[i]*v[i];

			if (_verbose) {	
				std::cout << err << std::endl;
				//addDebugChannel(vol, r[i], _dims[i], "Post ", i, true);
				addDebugChannel(vol, tmpx[i], _dims[i], "Interpolated corr ", i, true);
				
			}
		}


		T err = sqrt(r[0].squaredNorm() / f[0].squaredNorm());
		std::cout << "k = " << k << " err: " << err << std::endl;		

		if (err < tolerance || isinf(err) || isnan(err))
			return err;

		//std::cout << "k = " << k << ", res: " << << std::endl;
	
	}




	//Steps
	//0.0: define Ax = b on finest grid (done)
	//0.05 define Ax = b on multi grid
	//0.1: define initial guess of u ~ x 
	//1. Smooth error e
		//using restriction of residual to a coarser grid
		//1.1 calculate residual on fine grid
		//1.2 restrict to coarser grid (avg?)
		//1.3 Solve equation Ae = r on the coarser (or coarsest only?) grid 
			//(gives offset of x_i to reach x, on coarser grid) (pre smoothing??)
	//2. Propagate coarse erorr e to finer grid
		//2.1 Interpolate to finer grid
		//2.2 Smooth with iterative solver (Ae = r again?) (post smoothing)


	//Pseudocode
	//A, b
	// init guess u

	//Presmooth
	//e = jacobi(A,r,e,n) //Solve Ae = r, n iterations	
	//


	//Stopping tolerance
		//(norm of residual) / (norm of rhs)

	//residual definition
	//residual	|	r = A * x_i - b //difference of mult. result
	//error		|	e  = x_i - x //error of solution
	//			| x_i = e +x, r = A(e+x) - b = Ae + Ax - b = r -> (Ax - b = 0) -> Ae = r
	//			|	Ae = r
	//


	//Todos:
		//generate _A/2^level

	return 0.0f;

}


template <typename T>
BLIB_EXPORT bool blib::MultigridSolver<T>::resultToVolume(VolumeChannel & vol)
{

	assert(_x.size() > 0);

	void * destPtr = vol.getCurrentPtr().getCPU();

	//Copy directly if same type
	if ((std::is_same<float, T>::value && vol.type() == TYPE_FLOAT) ||
		(std::is_same<double, T>::value && vol.type() == TYPE_DOUBLE)) {
		memcpy(destPtr, _x[0].data(), _x[0].size() * sizeof(T));
	}
	else {
		if (vol.type() == TYPE_FLOAT) {
			Eigen::Matrix<float, Eigen::Dynamic, 1> tmpX = _x[0].cast<float>();
			memcpy(destPtr, tmpX.data(), tmpX.size() * sizeof(float));
		}
		else if (vol.type() == TYPE_DOUBLE) {
			Eigen::Matrix<double, Eigen::Dynamic, 1> tmpX = _x[0].cast<double>();
			memcpy(destPtr, tmpX.data(), tmpX.size() * sizeof(double));
		}
		else {
			return false;
		}
	}

	return true;
}





template <typename T>
BLIB_EXPORT bool blib::MultigridSolver<T>::generateErrorVolume(Volume & v)
{

	//assert(std::is_same<float, T>::value == true);

	for (uint i = 0; i < _lv; i++) {		
		auto id = v.addChannel(_dims[i], TYPE_FLOAT);
		auto & c = v.getChannel(id);
		memcpy(c.getCurrentPtr().getCPU(), _r[i].data(), _dims[i].x * _dims[i].y * _dims[i].z * sizeof(float));
		c.getCurrentPtr().commit();				
	}

	return true;
}



template <typename T>
BLIB_EXPORT T blib::MultigridSolver<T>::tortuosity(const VolumeChannel & mask, Dir dir)
{
	assert(mask.type() == TYPE_UCHAR || mask.type() == TYPE_CHAR);
	const auto dim = mask.dim();

	const T * concData = _x[0].data();
	const uchar * cdata = (uchar *)mask.getCurrentPtr().getCPU();

	const int primaryDim = getDirIndex(dir);
	const int secondaryDims[2] = { (primaryDim + 1) % 3, (primaryDim + 2) % 3 };


	int n = dim[secondaryDims[0]] * dim[secondaryDims[1]];
	int k = (getDirSgn(dir) == -1) ? 0 : dim[primaryDim] - 1;

	/*
	Calculate average in low concetration plane
	*/
	bool zeroOutPart = true;
	std::vector<T> isums(dim[secondaryDims[0]], T(0));

	#pragma omp parallel for
	for (auto i = 0; i < dim[secondaryDims[0]]; i++) {
		T jsum = T(0);
		for (auto j = 0; j < dim[secondaryDims[1]]; j++) {
			ivec3 pos;
			pos[primaryDim] = k;
			pos[secondaryDims[0]] = i;
			pos[secondaryDims[1]] = j;

			if (zeroOutPart && cdata[linearIndex(dim, pos)] == 0)
				jsum += concData[linearIndex(dim, pos)];
		}
		isums[i] = jsum;
	}

	T sum = std::accumulate(isums.begin(), isums.end(), T(0));

	double dc = sum / n;
	double dx = 1.0f / (dim[primaryDim] + 1);
	double tau = /*dx * */_porosity / (dc * /*dx **/ dim[primaryDim] * 2);

	if (_verbose) {
		std::cout << "dc: " << dc << std::endl;
		std::cout << "porosity: " << _porosity << std::endl;
		std::cout << "tau: " << tau << std::endl;
	}

	return tau;
}
