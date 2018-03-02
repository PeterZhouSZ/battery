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
			gaussSeidelStep(A, b, curX, nextX);

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

		for (auto i = 0; i < maxIter; ++i) {
			//gaussSeidelStep(A, b, x);
			for (auto k = 0; k < 6; k+=2) {
				gaussSeidelStepZebra(A, b, x, dim, Dir(k));
			}
			//gaussSeidelStepParallel(A, b, x, dim);

			residual = b - A*x;

			float err = residual.squaredNorm();
			tol_error = sqrt(err / bsqnorm);

			if (verbose && i % 128 == 0) {
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


	template <typename T>
	void gaussSeidelStep(
		const Eigen::SparseMatrix<T, Eigen::RowMajor> & A,
		const Eigen::Matrix<T, Eigen::Dynamic, 1> & b,
		Eigen::Matrix<T, Eigen::Dynamic, 1> & x
	) {

		for (auto i = 0; i < A.rows(); i++) {
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
