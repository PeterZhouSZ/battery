#include "DiffusionSolver.h"

#include <chrono>
#include <iostream>
#include <fstream>

#include <Eigen/IterativeLinearSolvers>
#include <Eigen/SparseLU>

#include <omp.h>
#include <stack>

#include "PrimitiveTypes.h"
#include "JacobiSolver.h"

//#define DS_LINSYS_TO_FILE

using namespace blib;

template class DiffusionSolver<float>;
template class DiffusionSolver<double>;



enum NeighType {
	NODE, 
	DIRICHLET,
	VON_NEUMANN
};

#define NO_NODE (size_t(0)-1)

struct Neigh {
	union {
		int index; //if node type
		float value; //if vonneumman -> add to diagonal, if dirichlet -> subtract from b
	};
	NeighType type;
};

struct Node {
	Node(ivec3 pos_) : pos(pos_) {}
	ivec3 pos;  //for debug
	Neigh neigh[6];
};




void buildNodeList(std::vector<Node> & nodeList, std::vector<size_t> & indices, const VolumeChannel & c, size_t nodeIndex_) {

	const auto & dim = c.dim();
	const size_t stride[3] = { 1, dim.x, dim.x*dim.y };
	const uchar * cdata = (uchar *)c.getCurrentPtr().getCPU();
	//todo add bound.cond here
	/*if (curPos.x < 0 || curPos.y < 0 || curPos.z < 0 ||
		curPos.x >= dim.x || curPos.y >= dim.y || curPos.z >= dim.z
		) return;*/

	
	
	

	std::stack<ivec3> stackNodePos;
	

	for (auto y = 0; y < dim.y; y++) {
		for (auto z = 0; z < dim.z; z++) {
			stackNodePos.push({ 0,y,z });
		}
	}
	while (!stackNodePos.empty()) {

		ivec3 nodePos = stackNodePos.top();		
		stackNodePos.pop();

		auto i = linearIndex(dim, nodePos);

	
		

		//Node at d1, skip
		if (indices[i] == NO_NODE) {
			if(cdata[i] != 0)
				continue;

			nodeList.push_back(Node(nodePos));
			indices[i] = nodeList.size() - 1;
		}


		assert(cdata[i] == 0);

		size_t nodeIndex = indices[i];
		

		auto * n = &nodeList[nodeIndex];
		

		//Visited
		//if (indices[i] != NO_NODE) return;

		const float conc[2] = { 0.0f,1.0f };
		const int dirichletDir = 0; //x dir

		const float highConc = 1.0f;
		const float lowConc = 0.0f;
		const float d0 = 0.001f;

		static const ivec3 dirVecs[6] = {
			ivec3(1,0,0),
			ivec3(-1,0,0),
			ivec3(0,1,0),
			ivec3(0,-1,0),
			ivec3(0,0,1),
			ivec3(0,0,-1)
		};

		for (auto k = 0; k < (3); k++) {
			for (auto dir = 0; dir < 2; dir++) {
				const int neighDir = 2 * k + dir;


				//Box domain conditions
				if ((dir == 0 && n->pos[k] == dim[k] - 1) ||
					(dir == 1 && n->pos[k] == 0)) {
					//Dirichlet  
					if (k == dirichletDir) {
						n->neigh[neighDir].type = DIRICHLET;
						n->neigh[neighDir].value = conc[dir]; //x1 or x0 conc
					}
					else {
						n->neigh[neighDir].type = VON_NEUMANN;
						n->neigh[neighDir].value = d0;
					}
				}
				//
				else {

					const ivec3 dirVec = dirVecs[2 * k + dir];
					const int thisStride = -(dir * 2 - 1) * int(stride[k]); // + or -, stride in k dim

					if (cdata[i + thisStride] == 0) {

						n->neigh[neighDir].type = NODE;

						int ni = linearIndex(dim, n->pos + dirVec);
						if (indices[ni] == NO_NODE) {
							nodeList.push_back(Node(n->pos + dirVec));
							indices[ni] = nodeList.size() - 1;
							n = &nodeList[nodeIndex]; //refresh pointer if realloc happened
							n->neigh[neighDir].index = indices[ni];							
							stackNodePos.push(n->pos + dirVec);
						}
						else {
							n->neigh[neighDir].index = indices[ni];
						}


					}
					else {
						n->neigh[neighDir].type = VON_NEUMANN;
						n->neigh[neighDir].value = d0;
					}
				}
			}

		}

	}

/*

	if (n.pos.x == 0) {
		n.neigh[X_NEG].type = DIRICHLET;
		n.neigh[X_NEG].value = highConc;
	}
	else {
		if (cdata[i - stride[0]] == 0){
			int ni = linearIndex(dim, n.pos + ivec3(-1, 0, 0));
			if (indices[ni] == NO_NODE) {
				nodeList.push_back(Node(n.pos + ivec3(-1, 0, 0)));
				indices[ni] = nodeList.size() - 1;
				buildNodeList(nodeList, indices, c, nodeList.back());
			}
			n.neigh[X_NEG].index = indices[ni];
			n.neigh[X_NEG].type = NODE;
		}
		else {
			n.neigh[X_NEG].type = VON_NEUMANN;
			n.neigh[X_NEG].value = d0;
		}
	}

	if (n.pos.x == dim.x - 1) {
		n.neigh[X_POS].type = DIRICHLET;
		n.neigh[X_POS].value = lowConc;
	}
	else {
		if (cdata[i + stride[0]] == 0) {
			int ni = linearIndex(dim, n.pos + ivec3(1, 0, 0));
			if (indices[ni] == NO_NODE) {
				nodeList.push_back(Node(n.pos + ivec3(1, 0, 0)));
				indices[ni] = nodeList.size() - 1;
				buildNodeList(nodeList, indices, c, nodeList.back());
			}
			n.neigh[X_POS].index = indices[ni];
			n.neigh[X_POS].type = NODE;
		}
		else {
			n.neigh[X_POS].type = VON_NEUMANN;
			n.neigh[X_POS].value = d0;
		}
	}
*/


/*

	if (n.pos.x == dim.x - 1) {
		n.neigh[X_NEG].type = DIRICHLET;
		n.neigh[X_NEG].value = lowConc;
	}
	else {
		

	}*/

	





}


template <typename T>
DiffusionSolver<T>::DiffusionSolver(bool verbose)
	: _verbose(verbose)	 
{
	
}

template <typename T>
DiffusionSolver<T>::~DiffusionSolver()
{

}


template <typename T>
bool DiffusionSolver<T>::prepare(VolumeChannel & volChannel, Dir dir, float d0, float d1)
{

	using vec3 = glm::tvec3<T, glm::highp>;
	
	auto dim = volChannel.dim();
	assert(volChannel.type() == TYPE_UCHAR);
	assert(dim.x > 0 && dim.y > 0 && dim.z > 0);
	
	uint dirPrimary = getDirIndex(dir);
	uint dirSecondary[2] = { (dirPrimary + 1) % 3, (dirPrimary + 2) % 3 };
	const T highConc = 1.0f;
	const T lowConc = 0.0f;
	const T concetrationBegin = (getDirSgn(dir) == 1) ? highConc : lowConc;
	const T concetrationEnd = (getDirSgn(dir) == 1) ? lowConc : highConc;

	size_t N = dim.x * dim.y * dim.z;
	size_t M = N; 

	_rhs.resize(M);	
	_x.resize(N);

	std::vector<Eigen::Triplet<T>> triplets(M);
	
	//D data
	const uchar * D = (uchar *)volChannel.getCurrentPtr().getCPU();	
	
	//D getter
	const auto getD = [dim, D, d0, d1](ivec3 pos) {
		return (D[linearIndex(dim, pos)] == 0) ? d0 : d1;
	};

	const auto sample = [&getD, dim, D, d0, d1](ivec3 pos, Dir dir) {
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
						hneg[k] *= 0.5;					
					else if (ipos[k] >= dim[k] - 1)
						hpos[k] *= 0.5;					
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

				auto Dcur = vec3(getD(ipos));				
				auto Dneg = (vec3(
					sample(ipos, X_NEG),
					sample(ipos, Y_NEG),
					sample(ipos, Z_NEG)
				) + vec3(Dcur)) * T(0.5) * invHneg2;

				auto Dpos = (vec3(
					sample(ipos, X_POS),
					sample(ipos, Y_POS),
					sample(ipos, Z_POS)
				) + vec3(Dcur)) * T(0.5) * invHpos2;
				
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
					bval = -T(0.5) * sample(ipos, getDir(dirPrimary, -1)) * concetrationBegin * invHneg2[dirPrimary];
					T c = T(0.5) * invHneg2[dirPrimary] * sample(ipos, getDir(dirPrimary, -1));;
					diagVal -= c;

				}
				else if (ipos[dirPrimary] == dim[dirPrimary] - 1) {
					bval = -T(0.5) * sample(ipos, getDir(dirPrimary, +1)) * concetrationEnd  * invHpos2[dirPrimary];
					T c = T(0.5)* invHpos2[dirPrimary] * sample(ipos, getDir(dirPrimary, +1));
					diagVal -= c;

				}


				if (z > 0) vals[0] = Dneg.z;
				if (y > 0) vals[1] = Dneg.y;
				if (x > 0) vals[2] = Dneg.x;
				vals[3] = diagVal;
				if (x < dim.x - 1) vals[4] = Dpos.x;
				if (y < dim.y - 1) vals[5] = Dpos.y;
				if (z < dim.z - 1) vals[6] = Dpos.z;

				_rhs[i] = bval;

				//initial guess
				if (getDirSgn(dir) == 1)
					_x[i] = 1.0f - (ipos[dirPrimary] / T(dim[dirPrimary] + 1));
				else
					_x[i] = (ipos[dirPrimary] / T(dim[dirPrimary] + 1));

				
				for (auto k = 0; k < 7; k++) {
					if (colId[k] < 0 || colId[k] >= N || vals[k] == 0.0f) continue;
					triplets.push_back(Eigen::Triplet<T>(i, colId[k], vals[k]));					
				}

			}
		}
	}


	//Todo fill directly
	_A.resize(M, N);
	_A.setFromTriplets(triplets.begin(), triplets.end());
	_A.makeCompressed();	



	_solver.compute(_A);

	return true;

}



template <typename T>
T DiffusionSolver<T>::solve(float tolerance, size_t maxIterations, size_t iterPerStep)
{
	
	iterPerStep = std::min(iterPerStep, maxIterations);
	_solver.setTolerance(tolerance);
	_solver.setMaxIterations(iterPerStep);

	T err = std::numeric_limits<T>::max();

	for (auto i = 0; i < maxIterations; i += iterPerStep) {

		_x = _solver.solveWithGuess(_rhs, _x);
		err = _solver.error();

		if (_verbose) {
			std::cout << "i:" << i << ", estimated error: " << err << std::endl;			
		}

		if (err <= tolerance)
			break;		
	}


	return err;

}


template <typename T>
bool DiffusionSolver<T>::resultToVolume(VolumeChannel & vol)
{

	void * destPtr = vol.getCurrentPtr().getCPU();

	//Copy directly if same type
	if ((std::is_same<float, T>::value && vol.type() == TYPE_FLOAT) ||
		(std::is_same<double, T>::value && vol.type() == TYPE_DOUBLE)) {
		memcpy(destPtr, _x.data(), _x.size() * sizeof(T));
	}
	else {
		if (vol.type() == TYPE_FLOAT) {
			Eigen::Matrix<float, Eigen::Dynamic, 1> tmpX = _x.cast<float>();
			memcpy(destPtr, tmpX.data(), tmpX.size() * sizeof(float));
		}
		else if (vol.type() == TYPE_DOUBLE) {
			Eigen::Matrix<double, Eigen::Dynamic, 1> tmpX = _x.cast<double>();
			memcpy(destPtr, tmpX.data(), tmpX.size() * sizeof(double));
		}
		else {
			return false;
		}
	}	
	
	return true;


}



template <typename T>
bool DiffusionSolver<T>::solve(
	VolumeChannel & volChannel, 
	VolumeChannel * outVolume, 
	Dir dir, 
	float d0,
	float d1,
	float tolerance	
)
{	
	using vec3 = glm::tvec3<T,glm::highp>;
	

	const auto & c = volChannel;
	auto dim = c.dim();
	
	uint dirPrimary = getDirIndex(dir);
	uint dirSecondary[2] = { (dirPrimary + 1) % 3, (dirPrimary + 2) % 3 };


	const T highConc = 1.0f;
	const T lowConc = 0.0f;

	const T concetrationBegin = (getDirSgn(dir) == 1) ? highConc : lowConc;
	const T concetrationEnd = (getDirSgn(dir) == 1) ? lowConc : highConc;

	
	if(_verbose)
		std::cout << "DIM " << dim.x << ", " << dim.y << ", " << dim.z << " nnz:" << dim.x*dim.y*dim.z / (1024 * 1024.0f) << "M" << std::endl;

	//Matrix dimensions	
	size_t N = dim.x * dim.y * dim.z; //cols
	size_t M = N; //rows
	//size_t nnz = _maxElemPerRow * M; //total elems
	

	//Reallocate if needed
	/*if (nnz > _nnz) {
		//Update dims
		_nnz = nnz;
		_M = M;
		_N = N;	
	}*/

	


	

	auto start0 = std::chrono::system_clock::now();
	

	std::vector<Eigen::Triplet<T>> triplets;
	Eigen::Matrix<T, Eigen::Dynamic, 1> b(M);
	b.setZero();
	Eigen::Matrix<T, Eigen::Dynamic, 1> X(N);

			

	volChannel.getCurrentPtr().retrieve();
	assert(volChannel.type() == TYPE_UCHAR);
	uchar * D = (uchar *)volChannel.getCurrentPtr().getCPU();

	const auto linIndex = [dim](int x, int y, int z) -> size_t {
		return x + dim.x * y + dim.x * dim.y * z;
	};


	const auto getD = [D,d0,d1, &linIndex](ivec3 pos) {
		return (D[linIndex(pos.x, pos.y, pos.z)] == 0) ? d0 : d1;
	};

	const auto sample = [&getD, dim, D, d0, d1](ivec3 pos, Dir dir/*, vec3 invH2*/) {

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


	const vec3 h_general = { 1.0f / (dim.x + 1), 1.0f / (dim.y + 1), 1.0f / (dim.z + 1) };		

	int curRowPtr = 0;
	for (auto z = 0; z < dim.z; z++) {
		for (auto y = 0; y < dim.y; y++) {
			for (auto x = 0; x < dim.x; x++) {

				const ivec3 ipos = { x,y,z };

				vec3 hneg = h_general;
				vec3 hpos = h_general;

				//Spacing on boundary
				for (auto k = 0; k < 3; k++) {					
					if (ipos[k] == 0) {
						hneg[k] *= 0.5;
					}
					else if (ipos[k] >= dim[k] - 1) {
						hpos[k] *= 0.5;
					}
				}				

				const vec3 invHneg = vec3(1.0) / hneg;
				vec3 invHneg2 = invHneg*invHneg;
				const vec3 invHpos = vec3(1.0) / hpos;
				vec3 invHpos2 = invHpos*invHpos;				

				auto i = linIndex(x, y, z);

				T bval = 0.0f;
				T vals[7] = { 0.0f,0.0f ,0.0f ,0.0f ,0.0f ,0.0f ,0.0f };
				int colId[7];
				colId[0] = i - dim.y*dim.x;
				colId[1] = i - dim.x;
				colId[2] = i - 1;

				colId[3] = i;

				colId[4] = i + 1;
				colId[5] = i + dim.x;
				colId[6] = i + dim.y*dim.x;



				auto Dcur = vec3(getD(ipos)); // vec x vec product

				auto DAtNeg = vec3(
					sample(ipos, X_NEG),
					sample(ipos, Y_NEG),
					sample(ipos, Z_NEG)
				);
				auto Dneg = (DAtNeg + vec3(Dcur)) * T(0.5) * invHneg2;

				auto Dpos = (vec3(
					sample(ipos, X_POS),
					sample(ipos, Y_POS),
					sample(ipos, Z_POS)
				) + vec3(Dcur)) * T(0.5) * invHpos2;

			
				


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
								
					bval =  -T(0.5) * sample(ipos, getDir(dirPrimary, -1)) * concetrationBegin * invHneg2[dirPrimary];

				
					T c = T(0.5)* invHneg2[dirPrimary] * sample(ipos, getDir(dirPrimary, -1));;
					diagVal -= c;
					
				}
				else if (ipos[dirPrimary] == dim[dirPrimary] - 1) {
					
					bval = -T(0.5) * sample(ipos, getDir(dirPrimary, +1)) * concetrationEnd  * invHpos2[dirPrimary];
					
					T c = T(0.5)* invHpos2[dirPrimary] * sample(ipos, getDir(dirPrimary, +1));
					diagVal -= c;
					
				}

				assert(diagVal != 0.0f);

				if (z > 0) vals[0] = Dneg.z;
				

				if (y > 0) vals[1] = Dneg.y;
			

				if (x > 0) vals[2] = Dneg.x;
				

				if (x < dim.x - 1) vals[4] = Dpos.x;
			
				if (y < dim.y - 1) vals[5] = Dpos.y;
			
				if (z < dim.z - 1) vals[6] = Dpos.z;
			

				

				
				

				vals[3] = diagVal;
				
				


				

				

					b[i] = bval;

					//initial guess
					if (getDirSgn(dir) == 1)
						X[i] = 1.0f - (ipos[dirPrimary] / T(dim[dirPrimary] + 1));
					else
						X[i] = (ipos[dirPrimary] / T(dim[dirPrimary] + 1));
				
			
		
				int inRow = 0;
				for (auto k = 0; k < 7; k++) {					
					if(vals[k] == 0.0f) continue;

					triplets.push_back(Eigen::Triplet<T>(i, colId[k], vals[k]));

					
					inRow++;
				}


				curRowPtr += inRow;
			}
		}
	}


	Eigen::SparseMatrix<T,Eigen::RowMajor> A(M, N);
	A.setFromTriplets(triplets.begin(), triplets.end());
	A.makeCompressed();

	



#ifdef DS_LINSYS_TO_FILE
	{
		std::ofstream f("A.txt");
		f << A.toDense();
		f.close();
	}
	{
		std::ofstream f("b.vec");
		f << b;
		f.close();
	}
#endif

	

	auto end0 = std::chrono::system_clock::now();

	std::chrono::duration<double> elapsed_seconds0 = end0 - start0;
	if (_verbose) {
		std::cout << "prep elapsed time: " << elapsed_seconds0.count() << "s\n";
	}




	
	auto start = std::chrono::system_clock::now();

	omp_set_num_threads(8);
	Eigen::setNbThreads(8);


	//auto Xinit = X;

	//solveJacobi<float>(A, b, X, tolerance, 150000, true);
	

	//X = Xinit;
	
	Eigen::BiCGSTAB<Eigen::SparseMatrix<T,Eigen::RowMajor>> stab;
	//Eigen::ConjugateGradient<Eigen::SparseMatrix<float>> stab;		
	//Eigen::SparseLU<Eigen::SparseMatrix<float>> stab;		
	//stab.analyzePattern(A);
	//stab.factorize(A);
	//x = stab.solve(b);
	//x = stab.compute(A).solve(b);

	const int maxIter = 1500;
	const int iterPerStep = 100;

	stab.setTolerance(tolerance);
	stab.setMaxIterations(iterPerStep);
	stab.compute(A);





	for (auto i = 0; i < maxIter; i += iterPerStep) {
		X = stab.solveWithGuess(b, X);
		double er = stab.error();
		if (_verbose) {

			
			//memcpy(outVolume->getCurrentPtr().getCPU(), X.data(), X.size() * sizeof(float));
			std::cout << "i:" << i << ", estimated error: " << er << std::endl;
			if (std::is_same<float, T>::value)
			{
				memcpy(outVolume->getCurrentPtr().getCPU(), X.data(), X.size() * sizeof(float));
				float tau = tortuosityCPU(
					volChannel,
					*outVolume,
					dir
				);
				std::cout << "tau = " << tau << "\n";
			}
		}
		if (er <= tolerance)
			break;
	}	




	auto end = std::chrono::system_clock::now();

	if (_verbose) {
		std::chrono::duration<double> elapsed_seconds = end - start;
		std::cout << "#iterations:     " << stab.iterations() << std::endl;
		std::cout << "estimated error: " << stab.error() << std::endl;
		std::cout << "solve host elapsed time: " << elapsed_seconds.count() << "s\n";				
		std::cout << "host avg conc: " << X.mean() << std::endl;
		std::cout << "tolerance: " << tolerance << std::endl;

		std::cout << "res.norm : " << (b - A*X).norm()  << std::endl;
		

	}

#ifdef DS_LINSYS_TO_FILE
	{
		std::ofstream f("x.vec");
		f << X;
		f.close();
	}
#endif






	
	

	if (outVolume) {
#ifdef DS_USEGPU
		memcpy(outVolume->getCurrentPtr().getCPU(), resDeviceX.data(), resDeviceX.size() * sizeof(float));
#else		

		//if using float
		if (std::is_same<float, T>::value) {
			memcpy(outVolume->getCurrentPtr().getCPU(), X.data(), X.size() * sizeof(float));
		}
		//conv from double
		else {
			Eigen::Matrix<float, Eigen::Dynamic, 1> xfloat = X.cast<float>();
			memcpy(outVolume->getCurrentPtr().getCPU(), xfloat.data(), xfloat.size() * sizeof(float));
		}
#endif
	
	
	}
	
	
	
	return true;
}











template <typename T>
BLIB_EXPORT bool DiffusionSolver<T>::solveWithoutParticles(
	VolumeChannel & volChannel, 
	VolumeChannel * outVolume, 
	float d0,
	float d1,
	float tolerance
)
{
#ifdef DS_USEGPU
	assert(false); // CPU only
#endif

	

	const auto & c = volChannel;
	auto dim = c.dim();

	if (_verbose)
		std::cout << "DIM " << dim.x << ", " << dim.y << ", " << dim.z << " nnz:" << dim.x*dim.y*dim.z / (1024 * 1024.0f) << "M" << std::endl;


	const float highConc = 1.0f;
	const float lowConc = 0.0f;
	//const float d0 = 0.001f;

	//Max dimensions
	size_t maxN = dim.x * dim.y * dim.z; //cols
	size_t maxM = maxN; //rows	


	auto start0 = std::chrono::system_clock::now();

	std::vector<Node> nodeList;
	nodeList.reserve(maxN); 
	std::vector<size_t> indices(volChannel.dim().x*volChannel.dim().y*volChannel.dim().z, 0-1);
	buildNodeList(nodeList, indices, volChannel, 1);


	float porosity = 0.0f;
	{
		const uchar * cdata = (uchar *)volChannel.getCurrentPtr().getCPU();
		size_t cnt = 0;
		for (auto i = 0; i < volChannel.getCurrentPtr().byteSize() / sizeof(uchar); i++) {
			if (cdata[i] == 0)
				cnt++;
		}
		porosity = (cnt / float(maxN));
	}
	
	if (_verbose)
	{		
		std::cout << "occupancy (d0, direct): " << 100.0f * porosity << std::endl;
	}

	if (_verbose)
		std::cout << "occupancy (d0): " << 100.0f * nodeList.size() / float(dim.x*dim.y*dim.z) << "%  (" << nodeList.size() / (1024 * 1024.0f) << "M)" << std::endl;

	auto end0= std::chrono::system_clock::now();
	std::chrono::duration<double> elapsed_seconds0 = end0 - start0;
	if (_verbose) {
		std::cout << "prep elapsed time: " << elapsed_seconds0.count() << "s\n";
	}



	////////////////////////////

	size_t M = nodeList.size();
	size_t N = M;

	const vec3 h = { 1.0f / (dim.x + 1), 1.0f / (dim.y + 1), 1.0f / (dim.z + 1) };
	const vec3 invH = { 1.0f / h.x, 1.0f / h.y, 1.0f / h.z };
	const vec3 invH2 = { invH.x*invH.x,invH.y*invH.y,invH.z*invH.z };


	std::vector<Eigen::Triplet<T>> triplets;
	
	Eigen::Matrix<T, Eigen::Dynamic, 1> b(M);
	b.setZero();
	Eigen::Matrix<T, Eigen::Dynamic, 1> x(N);

	size_t row = 0;
	for (auto & n : nodeList) {

		auto Dcur = vec3(d0) * invH2; // vec x vec product
		auto Dpos = Dcur;
		auto Dneg = Dcur;

		auto diagVal = -(Dpos.x + Dpos.y + Dpos.z + Dneg.x + Dneg.y + Dneg.z);
		float bval = 0.0f;

		int k = 0;
		for (auto & neigh : n.neigh) {
			float dval = (k % 2 == 0) ? Dpos[k / 2] : Dneg[k / 2];

			if (neigh.type == NODE) {
				triplets.push_back(Eigen::Triplet<T>(row, neigh.index, 
					dval
					));
			}
			else if (neigh.type == DIRICHLET) {
				bval -= d0 * neigh.value * invH2[k / 3];
			}
			else if (neigh.type == VON_NEUMANN) {
				diagVal += dval;
			}
			k++;
		}

		b[row] = bval;

		//initial guess
		x[row] = 1.0f - (n.pos.x / float(dim.x + 1));
		
		triplets.push_back(Eigen::Triplet<T>(row, row, diagVal));
		
		row++;
	}

	auto start = std::chrono::system_clock::now();
	Eigen::SparseMatrix<T, Eigen::RowMajor> A(M, N);
	A.setFromTriplets(triplets.begin(), triplets.end());
	A.makeCompressed();


	omp_set_num_threads(8);
	Eigen::setNbThreads(8);

	/*Eigen::VectorXf res;
	for (auto i = 0; i < 1024; i++) {
		jacobi(A, b, x);
		res = A*b - x;
		std::cout << "jacobi res: " << res.mean() << std::endl;
	}*/

	Eigen::BiCGSTAB<Eigen::SparseMatrix<T, Eigen::RowMajor>> stab;	

	
	stab.setTolerance(tolerance);
	stab.compute(A);
	

	const int maxIter = 3000;
	const int iterPerStep = 100;

	stab.setMaxIterations(iterPerStep);

	for (auto i = 0; i < maxIter; i += iterPerStep) {
		x = stab.solveWithGuess(b, x);		
		float er = stab.error();		
		if (_verbose) {			
			std::cout << "i:" << i << ", estimated error: " << er << std::endl;
		}
		if (er <= tolerance)
			break;
	}	
	
	auto end = std::chrono::system_clock::now();


	if (_verbose) {
		std::chrono::duration<double> elapsed_seconds = end - start;
		std::cout << "#iterations:     " << stab.iterations() << std::endl;
		std::cout << "estimated error: " << stab.error() << std::endl;
		std::cout << "solve host elapsed time: " << elapsed_seconds.count() << "s\n";
		std::cout << "host avg conc: " << x.mean() << std::endl;
		std::cout << "tolerance " << tolerance << std::endl;

	}

	//Convert solution to volume
	{
		float * concData = (float *)outVolume->getCurrentPtr().getCPU();
		const uchar * cdata = (uchar *)volChannel.getCurrentPtr().getCPU();
		outVolume->clearCurrent();
		int nodeIndex = 0;
		for (auto & n : nodeList) {
			auto i = linearIndex(dim, n.pos);
			concData[i] = x[nodeIndex];			
			nodeIndex++;

		}
	}

	return true;
}

template <typename T>
BLIB_EXPORT T blib::DiffusionSolver<T>::tortuosityCPU(const VolumeChannel & mask, const VolumeChannel & concetration, Dir dir)
{
	
	
	assert(mask.type() == TYPE_UCHAR || mask.type() == TYPE_CHAR);
	assert(concetration.type() == TYPE_FLOAT);
	assert(mask.dim().x == concetration.dim().x);
	assert(mask.dim().y == concetration.dim().y);
	assert(mask.dim().z == concetration.dim().z);
	const auto dim = mask.dim();
	
	const auto totalElem = dim.x * dim.y * dim.z;


	const float * concData = (float *)concetration.getCurrentPtr().getCPU();
	const uchar * cdata = (uchar *)mask.getCurrentPtr().getCPU();

	int zeroElem = 0;
	for (auto i = 0; i < totalElem; i++) {
		zeroElem += (cdata[i] == 0) ? 1 : 0;
	}
	double porosity = zeroElem / float(totalElem);


	

	const int primaryDim = getDirIndex(dir);
	const int secondaryDims[2] = { (primaryDim + 1) % 3, (primaryDim + 2) % 3 };
	
	double sum = 0.0f;
	//float sumHigh = 0.0f;
	int n = 0;	
	int k = (getDirSgn(dir) == -1) ? 0 : dim[primaryDim] - 1;
	int kHigh = (getDirSgn(dir) == 1) ? 0 : dim[primaryDim] - 1;

	bool zeroOutPart = true;

	for (auto i = 0; i < dim[secondaryDims[0]]; i++) {
		for (auto j = 0; j < dim[secondaryDims[1]]; j++) {
			ivec3 pos;
			pos[primaryDim] = k;
			pos[secondaryDims[0]] = i;
			pos[secondaryDims[1]] = j;


			if(zeroOutPart && cdata[linearIndex(dim, pos)] == 0)
				sum += concData[linearIndex(dim, pos)];
			
			/*pos[primaryDim] = kHigh;
			sumHigh += concData[linearIndex(dim, pos)];*/

			n++;
		}
	}

	//const float d0 = 0.001f;
	

	double dc = sum / n;
	double dx = 1.0f / (dim[primaryDim] + 1);
	double tau = /*dx * */porosity / (dc * /*dx **/ dim[primaryDim] * 2); /// (dx * (dim[primaryDim]+1) );


	std::cout << "dc: " << dc << std::endl;

	/*float avgJ = sum / n;
	//float h = 1.0f / dim[primaryDim];	

	float high = 1.0f;
	float low = 0.0f;
	float dc = high - low;
	float dx = 1.0f;

	float tau = d0 * porosity * dc / (avgJ * dx);

	std::cout << "AvgJ: " << avgJ << std::endl;
	std::cout << "Deff: " << (avgJ * dx) / dc << std::endl;*/
	std::cout << "porosity: " << porosity << std::endl;

	//float t = (d0 * (1.0f * porosity) / dc);
	//float t2 = h * porosity / dc / 1.0f / 2.0f;

	return tau;

	

	
}


