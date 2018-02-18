#include "MultigridSolver.h"

using namespace blib;

template class MultigridSolver<float>;
template class MultigridSolver<double>;

#include <Eigen/SparseLU>
#include "JacobiSolver.h"


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
	_rhs.resize(_lv);
	_x.resize(_lv);
	_dims.resize(_lv);

	//Generate D volume for smaller resolutions
	std::vector<std::vector<float>> Dlevels(_lv);
	std::vector<std::vector<float>> Dlevels_interp(_lv);
	

	const size_t origTotal = origDim.x * origDim.y * origDim.z;
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
		
		Dlevels_interp[i].resize(total, 0.0f);
		

		res &= prepareAtLevel(Dlevels[i].data(), _dims[i], dir, i);
	}




	for (uint i = 0; i < _lv; i++) {

		{
			auto id = v.addChannel(_dims[i], TYPE_FLOAT);
			auto & c = v.getChannel(id);
			memcpy(c.getCurrentPtr().getCPU(), Dlevels[i].data(), _dims[i].x * _dims[i].y * _dims[i].z * sizeof(float));
			c.getCurrentPtr().commit();
		}

		if(i < _lv - 1){
			interp3D(Dlevels[i + 1].data(), _dims[i + 1], Dlevels_interp[i].data(), _dims[i]);		
			auto id = v.addChannel(_dims[i], TYPE_FLOAT);
			auto & c = v.getChannel(id);
			memcpy(c.getCurrentPtr().getCPU(), Dlevels_interp[i].data(), _dims[i].x * _dims[i].y * _dims[i].z * sizeof(float));
			c.getCurrentPtr().commit();
		}



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

		
	
	_rhs[level].resize(M);
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

				_rhs[level][i] = bval;

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
T MultigridSolver<T>::solve(T tolerance, size_t maxIterations)
{


	std::vector<Vector> r(_lv);	
	std::vector<Vector> tmpx(_lv); //temp vecs
	std::vector<Vector> & f = _rhs;
	std::vector<Vector> & v = _x;
	std::vector<SparseMat> & A = _A;

	for (auto i = 0; i < _lv; i++) {
		r[i].resize(_x[i].size());
		tmpx[i].resize(_x[i].size());
	}


	Eigen::SparseLU<SparseMat> exactSolver;
	exactSolver.compute(A.back());
	
	
	

	

	//maxIterations = 0;
	//maxIterations V cycle
	for (auto k = 0; k < maxIterations; k++) {
		
		std::cout << "k = " << k << std::endl;
		
		for (auto i = 0; i < _lv - 1; i++) {

			if (i > 0) v[i].setZero();
			//Pre smoothing, v[i] is initial guess & result
			//Residual saved in r[i]
			solveJacobi(A[i], f[i], v[i], tmpx[i], r[i], tolerance, 4, true);
			
			//Restrict residual r[i] and use it as f[i+1]
			conv3D(r[i].data(), _dims[i], f[i + 1].data(), _dims[i + 1], _restrictOp);
						
		}

		std::cout << "exact solve" << std::endl;
		/*std::cout << v[_lv - 1];*/
		v[_lv - 1] = exactSolver.solve(f[_lv - 1]);

		//Up step
		for (int i = _lv - 2; i >= 0; i--) {

			//Interpolate v[i+1] to temp
			interp3D<T>(v[i + 1].data(), _dims[i + 1], tmpx[i].data(), _dims[i]);

			//Add temp to v[i]
			v[i] += tmpx[i];
			
			//Post smoothing, v[i] is initial guess & result		
			solveJacobi(A[i], f[i], v[i], tmpx[i], r[i], tolerance, 4, true);
		}

		std::cout << "k = " << k << ", res: " << sqrt(v[0].squaredNorm() / f[0].squaredNorm()) << std::endl;
	
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

