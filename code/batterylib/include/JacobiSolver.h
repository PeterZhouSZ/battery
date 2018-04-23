#pragma once



#include <iostream>
#include <omp.h>


#include <Eigen/Eigen>
#include "Types.h"

namespace blib {

	template <typename T>
	void solveJacobi(
		const Eigen::SparseMatrix<T, Eigen::RowMajor> & A,
		const Eigen::Matrix<T, Eigen::Dynamic, 1> & b,
		Eigen::Matrix<T, Eigen::Dynamic, 1> & x,
		float tolerance = 1e-4,
		size_t maxIter = 20000,
		bool verbose = false

	) {
		Eigen::Matrix<T, Eigen::Dynamic, 1> xprime(x.size());
		Eigen::Matrix<T, Eigen::Dynamic, 1> res(x.size());

		maxIter = (maxIter / 2) * 2; //divisible by two


		float bsqnorm = b.squaredNorm();
		float tol2 = tolerance * tolerance * bsqnorm;

		for (auto i = 0; i < maxIter; ++i) {

			auto & curX = (i % 2 == 0) ? x : xprime;
			auto & nextX = (i % 2 == 0) ? xprime : x;

			res = b - A*curX;

			float err = res.squaredNorm();// / bsqnorm;

										  //float err = res.mean();

			if (verbose && i % 128 == 0) {

				float tol_error = sqrt(err / bsqnorm);

				std::cout << "jacobi i: " << i << " err: " << err << ", tol_error: " << tol_error << std::endl;
			}

			if (err <= tol2) {


				if (verbose) {
					float tol_error = sqrt(err / bsqnorm);
					std::cout << "solved " << err << " <= " << tol2 << std::endl;
					std::cout << "tol_error" << tol_error << std::endl;
				}

				if (&x != &curX) {
					std::swap(x, curX);
				}
				break;
			}
			jacobiStep(A, b, curX, nextX);

		}

	}

	template <typename T>
	T solveJacobi(
		const Eigen::SparseMatrix<T, Eigen::RowMajor> & A,
		const Eigen::Matrix<T, Eigen::Dynamic, 1> & b,
		Eigen::Matrix<T, Eigen::Dynamic, 1> & x,
		Eigen::Matrix<T, Eigen::Dynamic, 1> & xprime,
		Eigen::Matrix<T, Eigen::Dynamic, 1> & residual,
		float tolerance = 1e-4,
		size_t maxIter = 20000,
		bool verbose = false
	) {
		assert(xprime.size() == x.size());
		assert(residual.size() == x.size());

		maxIter = ((maxIter + 1) / 2) * 2; //divisible by two


		T bsqnorm = b.squaredNorm();
		T tol2 = tolerance * tolerance * bsqnorm;
		T tol_error = T(1);

		for (auto i = 0; i < maxIter; ++i) {

			auto & curX = (i % 2 == 0) ? x : xprime;
			auto & nextX = (i % 2 == 0) ? xprime : x;

			residual = b - A*curX;

			float err = residual.squaredNorm();// / bsqnorm;

										  //float err = res.mean();
			tol_error = sqrt(err / bsqnorm);

			if (verbose && i % 128 == 0) {
				std::cout << "jacobi i: " << i << " err: " << err << ", tol_error: " << tol_error << std::endl;
			}

			if (tol_error <= tolerance) {
				if (verbose) {
					std::cout << "solved " << tol_error << " <= " << tolerance << std::endl;
				}

				if (&x != &curX) {
					std::swap(x, curX);
				}
				break;
			}
			jacobiStep(A, b, curX, nextX);

		}

		if (verbose) {
			if (tol_error > tolerance)
				std::cout << "jacobi not converged" << tol_error << " vs " << tolerance << std::endl;
		}

		return tol_error;

	}

	template <typename T>
	void jacobiStep(
		const Eigen::SparseMatrix<T, Eigen::RowMajor> & A,
		const Eigen::Matrix<T, Eigen::Dynamic, 1> & b,
		Eigen::Matrix<T, Eigen::Dynamic, 1> & x,
		Eigen::Matrix<T, Eigen::Dynamic, 1> & xnew
	) {

#ifndef _DEBUG
#pragma omp parallel for
#endif
		for (auto i = 0; i < A.rows(); i++) {

			T sum = 0.0f;
			T diag = 0.0f;
			for (Eigen::SparseMatrix<T, Eigen::RowMajor>::InnerIterator it(A, i); it; ++it) {
				auto  j = it.col();
				if (j == i) {
					diag = it.value();
					continue;
				}
				sum += it.value() * x[j];
			}

			/*if (diag == 0.0f) {
				char k;
				k = 0;
			}*/
			xnew[i] = (b[i] - sum) / diag;
		}
	}


	template <typename T>
	T solveGaussSeidel(
		const Eigen::SparseMatrix<T, Eigen::RowMajor> & A,
		const Eigen::Matrix<T, Eigen::Dynamic, 1> & b,
		Eigen::Matrix<T, Eigen::Dynamic, 1> & x,
		Eigen::Matrix<T, Eigen::Dynamic, 1> & residual,
		ivec3 dim,
		float tolerance = 1e-4,
		size_t maxIter = 20000,
		bool verbose = false
	) {

		assert(residual.size() == x.size());

		T bsqnorm = b.squaredNorm();
		T tol_error = T(1);

		Eigen::Matrix<T, Eigen::Dynamic, 1> nres = residual;

		Eigen::Matrix<T, Eigen::Dynamic, 1> xprime = x;

		for (auto i = 0; i < maxIter; ++i) {
			//gaussSeidelStep(A, b, x);
			/*for (auto k = 0; k < 6; k+=2) {
				gaussSeidelStepZebra(A, b, x, dim, Dir(k));
			}*/


		
			

			if (true) {
				for (auto k = 0; k < 1; k++) {
					gaussSeidelStepLineZebra<T, 0, 1, false>(A, b, x, dim);
					gaussSeidelStepLineZebra<T, 0, 1, true>(A, b, x, dim);
					gaussSeidelStepLineZebra<T, 1, 1, false>(A, b, x, dim);
					gaussSeidelStepLineZebra<T, 1, 1, true>(A, b, x, dim);
					gaussSeidelStepLineZebra<T, 2, 1, false>(A, b, x, dim);
					gaussSeidelStepLineZebra<T, 2, 1, true>(A, b, x, dim);
				}				
			}



			/*for (auto i = 0; i < 256; i++) {
				gaussSeidelBoundaries(A, b, x, dim);
			}*/

			if (false) {

				int iter = 512;
				{
					const int dir = 0;
					for (auto plane = 0; plane < dim[dir]; plane++) {
						jacobiPlane<T, dir>(A, b, x, xprime, dim, plane, iter);
					}
					
				}

				//std::cout << "x " << (x - xprime).squaredNorm() << std::endl;
				{
					const int dir = 1;
					for (auto plane = 0; plane < dim[dir]; plane++) {
						jacobiPlane<T, dir>(A, b, x, xprime, dim, plane, iter);
					}
				}

				//std::cout << "y " << (x - xprime).squaredNorm() << std::endl;
				{
					const int dir = 2;
					for (auto plane = 0; plane < dim[dir]; plane++) {
						jacobiPlane<T, dir>(A, b, x, xprime, dim, plane, iter);
					}
				}

				//std::cout << "z " << (x - xprime).squaredNorm() << std::endl;

				//std::cout << "======" << std::endl;
			}




/*

			gaussSeidelStepLineZebra<T, 0, -1, false>(A, b, x, dim);
			gaussSeidelStepLineZebra<T, 0, -1, true>(A, b, x, dim);
			gaussSeidelStepLineZebra<T, 1, -1, false>(A, b, x, dim);
			gaussSeidelStepLineZebra<T, 1, -1, true>(A, b, x, dim);
			gaussSeidelStepLineZebra<T, 2, -1, false>(A, b, x, dim);
			gaussSeidelStepLineZebra<T, 2, -1, true>(A, b, x, dim);
			*/


			//gaussSeidelStep(A, b, x);
			//gaussSeidelStepZebra(A, b, x, dim, X_POS);

			
			

			//gaussSeidelStepParallel(A, b, x, dim);

			residual = b - A*x;

			//Ordered GS
			/*{
				nres = residual.cwiseAbs() / residual.lpNorm<1>();

				struct E {
					T val;
					int index;
				};
				std::vector<E> indices;
				for (auto i = 0; i < nres.size(); i++) {
					indices.push_back({ nres[i],i });
				}
				std::sort(indices.begin(), indices.end(), [](const E & a, const E & b) { return a.val < b.val; });

				for (auto & e : indices) {
					gaussSeidelRelax(A, b, x, e.index);
				}

				residual = b - A*x;
			}
*/

			



			float err = residual.squaredNorm();
			tol_error = sqrt(err / bsqnorm);

			if (verbose) {
				std::cout << "gauss seidel i: " << i << " err: " << err << ", tol_error: " << tol_error << std::endl;
			}

			if (tol_error <= tolerance) {
				if (verbose) {
					std::cout << "solved " << tol_error << " <= " << tolerance << std::endl;
				}
				break;
			}

		}

		if (verbose) {
			if (tol_error > tolerance)
				std::cout << "jacobi not converged" << tol_error << " vs " << tolerance << std::endl;
		}

		return tol_error;
	}


	template <typename T, int dir> 
	void jacobiPlane(
		const Eigen::SparseMatrix<T, Eigen::RowMajor> & A,
		const Eigen::Matrix<T, Eigen::Dynamic, 1> & b,
		Eigen::Matrix<T, Eigen::Dynamic, 1> & x,
		Eigen::Matrix<T, Eigen::Dynamic, 1> & xprime,
		ivec3 dim,
		int planeInDir,
		int iterations
	) {
		const int secDirs[2] = {
			(dir + 1) % 3,
			(dir + 2) % 3
		};

		Eigen::Matrix<T, Eigen::Dynamic, 1> * X[2] = {
			&x, &xprime
		};

		if (iterations % 2 == 1) iterations++;

		for (auto iter = 0; iter < iterations; iter++) {

			auto & x0 = *X[iter % 2];
			auto & x1 = *X[(iter + 1) % 2];

			#pragma omp parallel for
			for (auto i = 0; i < dim[secDirs[0]]; i++) {
				for (auto j = 0; j < dim[secDirs[1]]; j++) {
					ivec3 vox;
					vox[dir] = planeInDir;
					vox[secDirs[0]] = i;
					vox[secDirs[1]] = j;


					auto row = linearIndex(dim, vox);
					T sum = T(0);
					T diag = T(0);
					for (Eigen::SparseMatrix<T, Eigen::RowMajor>::InnerIterator it(A, row); it; ++it) {
						auto  col = it.col();
						if (col == row) {
							diag = it.value();
							continue;
						}
						sum += it.value() * x0[col];
					}

					x1[row] = (b[row] - sum) / diag;

				}
			}
		}		

	}

	


	template <typename T>
	void gaussSeidelStep(
		const Eigen::SparseMatrix<T, Eigen::RowMajor> & A,
		const Eigen::Matrix<T, Eigen::Dynamic, 1> & b,
		Eigen::Matrix<T, Eigen::Dynamic, 1> & x
	) {

		for (auto i = 0; i < A.rows(); i++) {
		//for (auto i = A.rows(); i >= 0; i--) {
			T sum = T(0);
			T diag = T(0);
			for (Eigen::SparseMatrix<T, Eigen::RowMajor>::InnerIterator it(A, i); it; ++it) {
				auto  j = it.col();
				if (j == i) {
					diag = it.value();
					continue;
				}
				sum += it.value() * x[j];
			}			
			x[i] = (b[i] - sum) / diag;
		}


	}


	template <typename T>
	void gaussSeidelBoundaries(
		const Eigen::SparseMatrix<T, Eigen::RowMajor> & A,
		const Eigen::Matrix<T, Eigen::Dynamic, 1> & b,
		Eigen::Matrix<T, Eigen::Dynamic, 1> & X,
		ivec3 dim
	) {

		gaussSeidelBoundary<T, 0, -1>(A, b, X, dim);
		gaussSeidelBoundary<T, 0, 1>(A, b, X, dim);
		gaussSeidelBoundary<T, 1, -1>(A, b, X, dim);
		gaussSeidelBoundary<T, 1, 1>(A, b, X, dim);
		gaussSeidelBoundary<T, 2, -1>(A, b, X, dim);
		gaussSeidelBoundary<T, 2, 1>(A, b, X, dim);

		/*//Xneg
		gaussSeidelBoundary<T, 0, -1, false>(A, b, X, dim);
		gaussSeidelBoundary<T, 0, -1, true>(A, b, X, dim);
		//xpos
		gaussSeidelBoundary<T, 0, 1, false>(A, b, X, dim);
		gaussSeidelBoundary<T, 0, 1, true>(A, b, X, dim);

		//yneg
		gaussSeidelBoundary<T, 1, -1, false>(A, b, X, dim);
		gaussSeidelBoundary<T, 1, -1, true>(A, b, X, dim);
		//ypos
		gaussSeidelBoundary<T, 1, 1, false>(A, b, X, dim);
		gaussSeidelBoundary<T, 1, 1, true>(A, b, X, dim);

		//zneg
		gaussSeidelBoundary<T, 2, -1, false>(A, b, X, dim);
		gaussSeidelBoundary<T, 2, -1, true>(A, b, X, dim);
		//zpos
		gaussSeidelBoundary<T, 2, 1, false>(A, b, X, dim);
		gaussSeidelBoundary<T, 2, 1, true>(A, b, X, dim);*/
	}

	template <typename T, int dir, int sgn>
	void gaussSeidelBoundary(
		const Eigen::SparseMatrix<T, Eigen::RowMajor> & A,
		const Eigen::Matrix<T, Eigen::Dynamic, 1> & b,
		Eigen::Matrix<T, Eigen::Dynamic, 1> & X,
		ivec3 dim
	) {
		const int secDirs[2] = {
			(dir + 1) % 3,
			(dir + 2) % 3
		};
		for (auto i = 0; i < dim[secDirs[0]]; i++) {
			for (auto j = 0; j < dim[secDirs[1]]; j++) {
				ivec3 vox;
				auto dirBegin = (sgn == 1) ? (dim[dir] - 1) : 0;
				vox[dir] = dirBegin;
				vox[secDirs[0]] = i;
				vox[secDirs[1]] = j;

				//layers from boundary
				for (auto k = 0; k < 1; k++) {
					vox[dir] = dirBegin + k*(-sgn);

					auto row = linearIndex(dim, vox);

					T sum = T(0);
					T diag = T(0);
					for (Eigen::SparseMatrix<T, Eigen::RowMajor>::InnerIterator it(A, row); it; ++it) {
						auto  col = it.col();
						if (col == row) {
							diag = it.value();
							continue;
						}
						sum += it.value() * X[col];
					}

					X[row] = (b[row] - sum) / diag;

				}
			}
		}
	
	}

	template <typename T, int dir, int sgn, bool alternate>
	void gaussSeidelBoundaryAlternating(
		const Eigen::SparseMatrix<T, Eigen::RowMajor> & A,
		const Eigen::Matrix<T, Eigen::Dynamic, 1> & b,
		Eigen::Matrix<T, Eigen::Dynamic, 1> & X,
		ivec3 dim
	) {

		
		const int secDirs[2] = {
			(dir + 1) % 3,
			(dir + 2) % 3
		};

		

		for (auto i = 0; i < dim[secDirs[0]] / 2; i++) {
			for (auto j = 0; j < dim[secDirs[1]]; j++) {
				ivec3 vox;
				vox[dir] = (sgn == 1) ? (dim[dir]-1) : 0;
				vox[secDirs[0]] = i * 2;
				vox[secDirs[1]] = j;

				if (!alternate)
					vox[secDirs[0]] += j % 2;
				else
					vox[secDirs[0]] += 1 - j % 2;

				auto row = linearIndex(dim, vox);

				T sum = T(0);
				T diag = T(0);
				for (Eigen::SparseMatrix<T, Eigen::RowMajor>::InnerIterator it(A, row); it; ++it) {
					auto  col = it.col();
					if (col == row) {
						diag = it.value();
						continue;
					}
					sum += it.value() * X[col];
				}

				X[row] = (b[row] - sum) / diag;

			}
		}

	
	}

	template<typename T>
	void gaussSeidelRelax(
		const Eigen::SparseMatrix<T, Eigen::RowMajor> & A,
		const Eigen::Matrix<T, Eigen::Dynamic, 1> & b,
		Eigen::Matrix<T, Eigen::Dynamic, 1> & X,
		size_t row
	) {
		
		T sum = T(0);
		T diag = T(0);
		for (Eigen::SparseMatrix<T, Eigen::RowMajor>::InnerIterator it(A, row); it; ++it) {
			auto  col = it.col();
			if (col == row) {
				diag = it.value();
				continue;
			}
			sum += it.value() * X[col];
		}

		X[row] = (b[row] - sum) / diag;
	}


	template <typename T, int dir, int sgn, bool alternate>
	void gaussSeidelStepLineZebra(
		const Eigen::SparseMatrix<T, Eigen::RowMajor> & A,
		const Eigen::Matrix<T, Eigen::Dynamic, 1> & b,
		Eigen::Matrix<T, Eigen::Dynamic, 1> & X,
		ivec3 dim
	) {


		int primDim = dim[dir];
		const int secDirs[2] = {
			(dir + 1) % 3,
			(dir + 2) % 3
		};

		
		for (auto i = 0; i < dim[secDirs[0]] / 2; i++) {
			for (auto j = 0; j < dim[secDirs[1]]; j++) {

				ivec3 vox;
				vox[secDirs[0]] = i * 2;
				vox[secDirs[1]] = j;
/*

				if (vox[secDirs[0]] == 0 && secDirs[0] == 1) continue;
				if (vox[secDirs[0]] == dim[secDirs[0]] - 1 && secDirs[0] == 1) continue;

				if (vox[secDirs[1]] == 0 && secDirs[1] == 1) continue;
				if (vox[secDirs[1]] == dim[secDirs[1]] - 1 && secDirs[1] == 1) continue;

				if (vox[secDirs[0]] == 0 && secDirs[0] == 2) continue;
				if (vox[secDirs[0]] == dim[secDirs[0]] - 1 && secDirs[0] == 2) continue;

				if (vox[secDirs[1]] == 0 && secDirs[1] == 2) continue;
				if (vox[secDirs[1]] == dim[secDirs[1]] - 1 && secDirs[1] == 2) continue;*/

				if (!alternate)
					vox[secDirs[0]] += j % 2;
				else 
					vox[secDirs[0]] += 1 - j % 2;

				const int begin = (sgn == -1) ? int(primDim) - 1 : 0;
				const int end = (sgn == -1) ? -1 : int(primDim);

				for (auto k = begin; k != end; k+=sgn) {

					ivec3 voxi = { vox.x, vox.y, vox.z };
					voxi[dir] = k;


					auto row = linearIndex(dim, voxi);

					

					T sum = T(0);
					T diag = T(0);
					for (Eigen::SparseMatrix<T, Eigen::RowMajor>::InnerIterator it(A, row); it; ++it) {
						auto  col = it.col();
						if (col == row) {
							diag = it.value();
							continue;
						}
						sum += it.value() * X[col];
					}

					X[row] = (b[row] - sum) / diag;

					/*if (row == 0) {
						printf("i0 ... %f, %f\n", sum, X[row]);
					}*/

				}

			}		
		}



	}




	template <typename T>
	void gaussSeidelStepZebra(
		const Eigen::SparseMatrix<T, Eigen::RowMajor> & A,
		const Eigen::Matrix<T, Eigen::Dynamic, 1> & b,
		Eigen::Matrix<T, Eigen::Dynamic, 1> & X,
		ivec3 dim,
		Dir dir
	) {

		
		ivec3 begin = { 0,0,0 };
		ivec3 step = { 1,1,1 };
		ivec3 end = dim;

		int k = getDirIndex(dir);
		int sgn = getDirSgn(dir);

		step[k] = 2;
		if (sgn == -1) {			
			begin[k] = dim[k] - 1;
			end[k] = 0;			
			step[k] = -step[k];
		}

		
		for (auto d = 0; d < 2; d++){

			
			begin[k] += sgn * d; 
			//end[k] += sgn * d;

			//#pragma omp parallel for

			
			
			//for (auto zi = 0; zi < dim.z; zi += step.z) {
				//z = begin.z + zi*sgn;
			for (auto z = begin.z; z != end.z && z != end.z + step.z / 2; z += step.z) {
				for (auto y = begin.y; y != end.y && y != end.y + step.y / 2; y += step.y) {
					for (auto x = begin.x; x != end.x && x != end.x + step.x / 2; x += step.x) {

						auto i = linearIndex(dim, { x,y,z });

						T sum = T(0);
						T diag = T(0);
						for (Eigen::SparseMatrix<T, Eigen::RowMajor>::InnerIterator it(A, i); it; ++it) {
							auto  j = it.col();
							if (j == i) {
								diag = it.value();
								continue;
							}
							sum += it.value() * X[j];
						}
						X[i] = (b[i] - sum) / diag;
					}
				}
			}






		}


		

}



	template <typename T>
	void gaussSeidelStepParallel(
		const Eigen::SparseMatrix<T, Eigen::RowMajor> & A,
		const Eigen::Matrix<T, Eigen::Dynamic, 1> & b,
		Eigen::Matrix<T, Eigen::Dynamic, 1> & X,
		ivec3 dim
	) {


		for (auto k = 0; k < 2; k++) {

			#pragma omp parallel for
			for (auto z = 0; z < dim.z; z++) {
				for (auto y = 0; y < dim.y; y++) {
					int xstart = ((y + z) % 2 == k) ? 0 : 1;
					for (auto x = xstart; x < dim.x; x += 2) {
						assert((x + y + z) % 2 == k);

						auto i = linearIndex(dim, { x,y,z });

						T sum = T(0);
						T diag = T(0);
						for (Eigen::SparseMatrix<T, Eigen::RowMajor>::InnerIterator it(A, i); it; ++it) {
							auto  j = it.col();
							if (j == i) {
								diag = it.value();
								continue;
							}							
							sum += it.value() * X[j];
						}
						X[i] = (b[i] - sum) / diag;

					}
				}
			}

		}

	}

}
