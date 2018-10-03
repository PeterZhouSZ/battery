#include "BICGSTABGPU.h"

#include "DataPtr.h"
#include "cuda/Volume.cuh"
#include "cuda/LinearSys.cuh"
#include "cuda/CudaUtility.h"

#include "Volume.h"
#include "Timer.h"

#include <iostream>

#ifdef DEBUG_SOLVER
#define DMSG(x) std::cout << x << std::endl;
#else
#define DMSG(x) {}
#endif

namespace blib {
	

	template <typename T>
	BICGSTABGPU<T>::BICGSTABGPU(bool verbose) : _volume(std::make_unique<Volume>()), Solver(verbose)
	{

	}



	template <typename T>
	bool BICGSTABGPU<T>::prepare(const PrepareParams & params)
	{

		assert(params.mask && params.output);

		if (!params.mask) return false;
		if (!params.output) return false;

		_volume->clear();
		

		_params = params;		

		const auto dim = _params.mask->dim();
		const size_t N = dim.x * dim.y * dim.z;


		auto &v = *_volume;

		{			
			_domain = v.getCUDAVolume(v.addChannel(dim, primitiveTypeof<T>(), false, "domain"));
			_f = v.getCUDAVolume(v.addChannel(dim, primitiveTypeof<T>(), false, "f"));							
			_x = params.output->getCUDAVolume();			 

			T zero = T(0);
			launchClearKernel(TYPE_DOUBLE, _x->surf, _x->res, &zero);

			if (_verbose)
				std::cout << "Allocated _domain, _f & _x" << std::endl;
		}
		

		{		
			
			const size_t reduceN = Volume_Reduce_RequiredBufferSize(N);
			_auxReduceBuffer = std::make_shared<DataPtr>();
			_auxReduceBuffer->allocDevice(reduceN, sizeof(T));
			_auxReduceBuffer->allocHost();

			if (_verbose)
				std::cout << "Allocated aux buffer " << reduceN / float(1024 * 1024) << "MB" << std::endl;
		}


		/*
		Generate continous domain 
		*/
		{
			auto MGmask = _params.mask->getCUDAVolume();
			blib::CUDATimer tGenDomain(true);

			LinearSys_GenerateDomain(*MGmask, _params.d0, _params.d1, *_domain);
			
			if(_verbose)
				std::cout << "Gen domain " << tGenDomain.time() << "s" << std::endl;
		}

		/*
			Setup and send system parameters to GPU
		*/		
		{
			LinearSys_SysParams sysp;
			sysp.highConc = double(1.0);
			sysp.lowConc = double(0.0);
			sysp.concetrationBegin = (getDirSgn(params.dir) == 1) ? sysp.highConc : sysp.lowConc;
			sysp.concetrationEnd = (getDirSgn(params.dir) == 1) ? sysp.lowConc : sysp.highConc;

			sysp.cellDim[0] = params.cellDim.x;
			sysp.cellDim[1] = params.cellDim.y;
			sysp.cellDim[2] = params.cellDim.z;
			sysp.faceArea[0] = sysp.cellDim[1] * sysp.cellDim[2];
			sysp.faceArea[1] = sysp.cellDim[0] * sysp.cellDim[2];
			sysp.faceArea[2] = sysp.cellDim[0] * sysp.cellDim[1];

			sysp.dirPrimary = getDirIndex(params.dir);
			sysp.dirSecondary = make_uint2((sysp.dirPrimary + 1) % 3, (sysp.dirPrimary + 2) % 3);

			sysp.dir = params.dir;

			if (!LinearSys_commitSysParams(sysp)) {
				return false;
			}

			if (_verbose)
				std::cout << "Commit sys params" << std::endl;
		}


		/*
		Generate A
		*/
		{	
			_A = std::make_shared<DataPtr>();
			_A->allocDevice(N, sizeof(CUDA_Stencil_7));

			CUDATimer t(true);
			LinearSys_GenerateSystem(
				*_domain, 
				static_cast<CUDA_Stencil_7*>(_A->gpu), 
				*_f
			);

			if(_verbose)
				std::cout << "Gen Ax = b: " << t.time() << "s" << std::endl;
		}

		//Release domain - not needed for solving
		{
			v.removeChannel(_domain->ID);
			_domain = nullptr;		
		}

		//Allocate solver vectors
		{
			_temp = v.getCUDAVolume(v.addChannel(dim, primitiveTypeof<T>(), false, "temp"));			
			_rhat0 = v.getCUDAVolume(v.addChannel(dim, primitiveTypeof<T>(), false, "rhat0"));
			_r = v.getCUDAVolume(v.addChannel(dim, primitiveTypeof<T>(), false, "r"));
			_p = v.getCUDAVolume(v.addChannel(dim, primitiveTypeof<T>(), false, "p"));
			_v = v.getCUDAVolume(v.addChannel(dim, primitiveTypeof<T>(), false, "v"));
			_s = v.getCUDAVolume(v.addChannel(dim, primitiveTypeof<T>(), false, "s"));
			_t = v.getCUDAVolume(v.addChannel(dim, primitiveTypeof<T>(), false, "t"));
			_y = v.getCUDAVolume(v.addChannel(dim, primitiveTypeof<T>(), false, "y"));
			_z = v.getCUDAVolume(v.addChannel(dim, primitiveTypeof<T>(), false, "z"));
			_ainvert = v.getCUDAVolume(v.addChannel(dim, primitiveTypeof<T>(), false, "ainvert"));			
		}


		{
			CUDATimer t(true);

			LinearSys_InvertSystemDiagTo(static_cast<CUDA_Stencil_7*>(_A->gpu), *_ainvert);

			if (_verbose)
				std::cout << "1.0 / Diag(A) " << t.time() << "s" << std::endl;
		}

		return true;
	}



	template <typename T>
	BLIB_EXPORT typename Solver<T>::Output BICGSTABGPU<T>::solve(const SolveParams & solveParams)
	{
		Solver<T>::Output out;
		out.error = 0;
		out.iterations = 0;
		out.status = SOLVER_STATUS_NO_CONVERGENCE;

		const uint3 res = _x->res;
		auto SqNorm = [&](CUDA_Volume & vol) {
			return Volume_SquareNorm(res, vol, _auxReduceBuffer->gpu, _auxReduceBuffer->cpu);
		};

		auto DotProduct = [&](CUDA_Volume & vol0, CUDA_Volume & vol1) {
			T result;
			Volume_DotProduct(vol0, vol1, *_temp, _auxReduceBuffer->gpu, _auxReduceBuffer->cpu, &result);
			return result;
		};


		Profiler profiler;

		CUDA_Stencil_7 * A = static_cast<CUDA_Stencil_7 *>(_A->gpu);
		auto & r = *_r;
		auto & r0 = *_rhat0;
		auto & rhs = *_f;
		auto & x = *_x;
		auto & ainvert = *_ainvert;

		T zero = 0.0;
		T tol = solveParams.tolerance;
		T tol_error = 1.0;

		size_t maxIters = solveParams.maxIter;
		size_t n = res.x*res.y*res.z;

		//1. Residual r
		CUDA_TIMER(LinearSys_Residual(A, x, rhs, r),"residual",profiler);

		//2. Choose rhat0 ..
		CUDA_TIMER(launchCopyKernel(TYPE_DOUBLE, res, r.surf, r0.surf),"copy", profiler);

		T r0_sqnorm = SqNorm(r0);
		T rhs_sqnorm = SqNorm(rhs);

		//Trivial solution
		if (rhs_sqnorm == 0) {
			launchClearKernel(TYPE_DOUBLE, x.surf, res, &zero);
			out.error = T(0);			
			out.status = Solver::SOLVER_STATUS_SUCCESS;
			return out;
		}

		T rho = 1;
		T alpha = 1;
		T w = 1;

		auto & p = *_p;
		auto & v = *_v;
		launchClearKernel(TYPE_DOUBLE, v.surf, res, &zero);
		launchClearKernel(TYPE_DOUBLE, p.surf, res, &zero);

		auto & y = *_y;
		auto & z = *_z;
		auto & s = *_s;
		auto & t = *_t;

		T tol2 = tol*tol*rhs_sqnorm;
		T eps2 = Eigen::NumTraits<T>::epsilon() * Eigen::NumTraits<T>::epsilon();

		size_t i = 0;
		size_t restarts = 0;		

		


		T rsqNorm = 0;
		CUDA_TIMER(rsqNorm = SqNorm(r), "sqNorm", profiler);
		while (rsqNorm > tol2 && i < maxIters)
		{
			DMSG(i);
			DMSG("\t" << "sq r: " << SqNorm(*r));

			T rho_old = rho;

			CUDA_TIMER(rho = DotProduct(r0, r), "dot", profiler);

			DMSG("\t" << " rho: " << rho);

			if (abs(rho) < eps2*r0_sqnorm)
			{
				DMSG("\t" << " restart ");
				CUDA_TIMER(LinearSys_Residual(A, x, rhs, r), "residual", profiler); 
				CUDA_TIMER(launchCopyKernel(TYPE_DOUBLE, res, r.surf, r0.surf), "copy", profiler); 
				CUDA_TIMER(rho = r0_sqnorm = SqNorm(r), "sqNorm", profiler); 
				if (restarts++ == 0)
					i = 0;
			}

			T beta = (rho / rho_old) * (alpha / w);
			DMSG("\t" << " beta: " << beta);

			
			//p = r + beta * (p - w * v);
			CUDA_TIMER(launchAPlusBetaBGammaPlusC(TYPE_DOUBLE, res, p.surf, v.surf, r.surf, -w, beta), "launchAPlusBetaBGammaPlusC", profiler);

			DMSG("\t" << " psq: " << SqNorm(p));

			//Preconditioner
			{								
				CUDA_TIMER(launchMultiplyKernel(TYPE_DOUBLE, res, ainvert.surf, p.surf, y.surf), "multiply", profiler);
				DMSG("\t" << " ysq: " << SqNorm(y));
				CUDA_TIMER(LinearSys_MatrixVectorProduct(A, y, v), "matvec", profiler);
				DMSG("\t" << " vsq: " << SqNorm(v));
			}


			
			CUDA_TIMER(alpha = rho / DotProduct(r0, v), "dot", profiler);
			DMSG("\t" << " alpha: " << alpha << " = " << rho << "/" << DotProduct(r0, v));

			//s = r - alpha  v;
			CUDA_TIMER(launchAddAPlusBetaB(TYPE_DOUBLE, res, r.surf, v.surf, s.surf, -alpha), "launchAddAPlusBetaB", profiler);
			DMSG("\t" << " ssq: " << SqNorm(s));

			//Preconditioner
			{
				CUDA_TIMER(launchMultiplyKernel(TYPE_DOUBLE, res, ainvert.surf, s.surf, z.surf), "multiply", profiler);
				DMSG("\t" << " zsq: " << SqNorm(z));

				CUDA_TIMER(LinearSys_MatrixVectorProduct(A, z, t), "matvec", profiler);
				DMSG("\t" << " tsq: " << SqNorm(t));
			}


			T tmp;
			CUDA_TIMER(tmp = SqNorm(t), "sqNorm", profiler);

			if (tmp > T(0)) {				
				CUDA_TIMER(w = DotProduct(t, s) / tmp, "dot", profiler);
			}
			else {
				w = T(0);
			}
			DMSG("\t" << " w: " << w);

			
			//x += alpha * y + w * z;
			CUDA_TIMER(launchABC_BetaGamma(TYPE_DOUBLE, res, x.surf, y.surf, z.surf, alpha, w), "launchABC_BetaGamma", profiler);
			DMSG("\t" << " xsq: " << SqNorm(x));
			
			//r = s - w * t; C = A + beta * B
			CUDA_TIMER(launchAddAPlusBetaB(TYPE_DOUBLE, res, s.surf, t.surf, r.surf, -w), "launchAddAPlusBetaB", profiler);

			CUDA_TIMER(rsqNorm = SqNorm(r), "sqNorm", profiler);
			DMSG("\t" << " rsq: " << SqNorm(r));

			++i;
		}

		CUDA_TIMER(tol_error = sqrt(SqNorm(r) / rhs_sqnorm), "sqNorm", profiler);
		_iterations += i;

		profiler.stop();


		if (_verbose) {
			std::cout << "Iterations: " << _iterations << " / " << solveParams.maxIter << std::endl;
			std::cout << profiler.summary();
		}

		out.error = tol_error;
		out.iterations = _iterations;
		out.status = (tol_error < solveParams.tolerance) ? SOLVER_STATUS_SUCCESS : SOLVER_STATUS_NO_CONVERGENCE;
		
		return out;
	}


	

	template class BICGSTABGPU<double>;
	template class BICGSTABGPU<float>;
}
