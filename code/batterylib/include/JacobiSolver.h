#pragma once



#include <iostream>
#include <omp.h>

#include <Eigen/Eigen>

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
void solveJacobi(
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

	maxIter = (maxIter / 2) * 2; //divisible by two


	float bsqnorm = b.squaredNorm();
	float tol2 = tolerance * tolerance * bsqnorm;

	for (auto i = 0; i < maxIter; ++i) {

		auto & curX = (i % 2 == 0) ? x : xprime;
		auto & nextX = (i % 2 == 0) ? xprime : x;

		residual = b - A*curX;

		float err = residual.squaredNorm();// / bsqnorm;

									  //float err = res.mean();
		float tol_error = sqrt(err / bsqnorm);

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
void jacobiStep(
	const Eigen::SparseMatrix<T, Eigen::RowMajor> & A,
	const Eigen::Matrix<T, Eigen::Dynamic, 1> & b,
	Eigen::Matrix<T, Eigen::Dynamic, 1> & x,
	Eigen::Matrix<T, Eigen::Dynamic, 1> & xnew
) {


#pragma omp parallel for
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
