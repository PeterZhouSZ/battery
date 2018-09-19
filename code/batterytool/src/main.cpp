#include <iostream>

#include <args.h>

#include <batterylib/include/DiffusionSolver.h>
#include <batterylib/include/MGGPU.h>
#include <batterylib/include/VolumeIO.h>

#include <batterylib/src/cuda/CudaUtility.h>


#include <chrono>
#include <numeric>

#if defined(__GNUC__)
    #include <experimental/filesystem>
#else
	#include <filesystem>
#endif

#include <fstream>

namespace fs = std::experimental::filesystem;

using namespace std;
using namespace blib;


std::string tmpstamp(const std::string & format /*= "%Y_%m_%d_%H_%M_%S"*/)
{
	char buffer[256];
	std::time_t now = std::time(NULL);
	std::tm * ptm = std::localtime(&now);
	std::strftime(buffer, 256, format.c_str(), ptm);
	return std::string(buffer);
}

args::ArgumentParser parser("Battery tool", "Vojtech Krs (2018) vkrs@purdue.edu");
args::HelpFlag help(parser, "help", "", { 'h', "help" });

/*
IO
*/
args::Positional<std::string> argInput(parser, "input", "Input file", args::Options::Required);
args::ValueFlag<std::string> argOutput(parser, "output", "Output file", { 'o', "output" }, "");
args::ValueFlag<uint> argSubvolume(parser, "subvolume", "Sub Volume", { "sub" }, 0);
args::ValueFlag<std::string> argPrecision(parser, "precision", "Precision (float|double)", { 'p', "prec" }, "double");

args::Group group(parser, "Tortuosity:", args::Group::Validators::AtLeastOne);

args::Flag argTau(group, "t", "Tortuosity", { 't', "tau" });
args::ValueFlag<std::string> argTauDir(group, "string", "Direction (x|y|z)|all|pos|neg", { 'd', "dir" }, "x-");
args::ValueFlag<int> argTol(group, "tolerance", "Tolerance 1e-{tol}", { "tol" }, 6);
args::ValueFlag<int> argMaxIterations(group, "maxIterations", "Max Iterations", { "iter" }, 10000);
args::ValueFlag<int> argStep(group, "step", "Step", { "step" }, 250);
args::ValueFlag<std::string> argSolver(group, "string", "Solver (MGGPU|Eigen)", { "solver" }, "MGGPU");
args::Flag argVolumeExport(group, "Volume export", "Concetration volume export", { "volExport" });
args::Flag argVerbose(group, "v", "Verbose", { 'v', "verbose" });


/*
	Extract what directions should be calculated from flagStr
	all - positive and negative
	pos, neg - three directions
	x and/or y and/or z
	default: x positive 
*/
std::vector<Dir> getDirs(std::string & flagStr) {

	if (flagStr == "all")
		return { X_POS, X_NEG,Y_POS,Y_NEG,Z_POS,Z_NEG};
	if(flagStr == "pos")
		return { X_POS, Y_POS, Z_POS};
	if (flagStr == "neg")
		return { X_NEG, Y_NEG, Z_NEG };


	std::vector<Dir> arr;
	for (auto & c : flagStr) {
		if (c == 'x') arr.push_back(X_POS);
		else if (c == 'y') arr.push_back(Y_POS);
		else if (c == 'z') arr.push_back(Z_POS);
	}
	

	if (arr.empty())
		return { X_POS };

	return arr;
}


std::string dirString(Dir d){
	switch(d){
		case X_POS: return "X_POS";
		case X_NEG: return "X_NEG";
		case Y_POS: return "Y_POS";
		case Y_NEG: return "Y_NEG";
		case Z_POS: return "Z_POS";
		case Z_NEG: return "Z_NEG";
		case DIR_NONE: return "DIR_NONE";
	}

	return "Undefined direction";
}

template <typename T>
bool tortuosity() {

	blib::VolumeChannel::enableOpenGLInterop = false;
	

	if (argVerbose) {
		std::cout << "Precision: " << argPrecision.Get() << std::endl;
		std::cout << "Tolerance: " << argTol.Get() << std::endl;
		std::cout << "MaxIterations: " << argMaxIterations.Get() << " intermediate steps: " << argStep.Get() << std::endl;
	}


	/*
	Load input volume
	*/
	blib::Volume volume;
	uint IDMask = 0;
	uint IDConc = 1;
	try {
		IDMask = volume.emplaceChannel(
			blib::loadTiffFolder(argInput.Get().c_str(), true)
		);
		volume.binarize(IDMask, 1.0f);

		IDConc = volume.addChannel(volume.getChannel(IDMask).dim(), TYPE_FLOAT);

	}
	catch (const char * ex) {
		std::cerr << "Failed to load: ";
		std::cerr << ex << std::endl;
		return false;
	}

	blib::VolumeChannel & c = volume.getChannel(IDMask);

	//Resize volume if desired
	if (argSubvolume.Get() != 0) {
		auto dim = c.dim();
		blib::ivec3 dd = glm::min(dim, blib::ivec3(argSubvolume.Get()));
		c.resize({ 0,0,0 }, blib::ivec3(dd));
	}

	//Solution for export
	

	if (argVerbose) {
		auto dim = c.dim();
		std::cout << "Resolution: " << dim.x << " x " << dim.y << " x " << dim.z <<
			" = " << dim.x*dim.y*dim.z << " voxels "
			<< "(" << (dim.x*dim.y*dim.z) / (1024 * 1024.0f) << "M)" << std::endl;


	}


	std::vector<Dir> dirs = getDirs(argTauDir.Get());
	std::vector<T> taus(dirs.size());
	T porosity;
	std::vector<double> times(dirs.size());

	T d0 = 1.0f;
	T d1 = 0.001f;

	
	const bool bicg = true;
	const bool runAllSolvers = false;
	const size_t maxIterations = argMaxIterations.Get();
	

	const bool verbose = argVerbose.Get();
	const bool verboseDebug = false;
	
	blib::DiffusionSolver<T> solverEigen(verbose);
	blib::MGGPU<T> solverMGGPU;


//#define KERNEL_PROFILE


#ifdef KERNEL_PROFILE
	{
		typename blib::MGGPU<T>::PrepareParams p;
		{
			p.dir = dirs[0];
			p.d0 = d0;
			p.d1 = d1;
			auto maxDim = std::max(c.dim().x, std::max(c.dim().y, c.dim().z));
			auto minDim = std::min(c.dim().x, std::min(c.dim().y, c.dim().z));
			auto exactSolveDim = 4;
			p.cellDim = blib::vec3(1.0 / maxDim);
			p.levels = std::log2(minDim) - std::log2(exactSolveDim) + 1;
		}	
		solverMGGPU.bicgPrep(c, p, volume);

		typename blib::MGGPU<T>::SolveParams sp;
		solverMGGPU.profile();
		exit(0);		
	}
#endif


	for (auto i = 0; i < dirs.size(); i++) {
		auto dir = dirs[i];
		if (argVerbose) {
			std::cout << "Direction " << dir << std::endl;
		}

		auto t0 = std::chrono::system_clock::now();

		T tol = T(pow(10.0, -argTol.Get()));

		//Prepare linear system
		if (argSolver.Get() == "Eigen" || runAllSolvers) {			
			solverEigen.prepare(
				c,
				dir,
				d0,
				d1,
				true
			);

			solverEigen.solve(tol, maxIterations,  argStep.Get());

			taus[i] = solverEigen.tortuosity(c, dir);
			porosity = solverEigen.porosity();
		}

		if (argSolver.Get() == "MGGPU" || runAllSolvers) {
			typename blib::MGGPU<T>::PrepareParams p;
			{
				p.dir = dir;
				p.d0 = d0;
				p.d1 = d1;				
				auto maxDim = std::max(c.dim().x, std::max(c.dim().y, c.dim().z));
				auto minDim = std::min(c.dim().x, std::min(c.dim().y, c.dim().z));
				auto exactSolveDim = 4;
				p.cellDim = blib::vec3(1.0 / maxDim);
				p.levels = std::log2(minDim) - std::log2(exactSolveDim) + 1;								
			}			
			if(bicg)
				solverMGGPU.bicgPrep(c, p, volume);
			else
				solverMGGPU.prepare(c, p, volume);

			typename blib::MGGPU<T>::SolveParams sp;
			sp.tolerance = tol;
			sp.verbose = verbose;
			sp.verboseDebug = verboseDebug;
			sp.maxIter = maxIterations;
			
			if (bicg) {
				T err = solverMGGPU.bicgSolve(sp);
				std::cout << err << std::endl;
			}
			else
				solverMGGPU.solve(sp);

			taus[i] = solverMGGPU.tortuosity();
			porosity = solverMGGPU.porosity();
		}		

		//Calculate tortuosity and porosity				
		auto t1 = std::chrono::system_clock::now();


		std::chrono::duration<double> dt = t1 - t0;
		times[i] = dt.count();

		if (argVerbose) {
			std::cout << "Elapsed: " << dt.count() << "s (" << dt.count() / 60.0 << "m)" << std::endl;
		}

		//Export calculated concetration volume
		if (argVolumeExport && argSolver.Get() == "Eigen") {
			const std::string exportPath = (argInput.Get() + std::string("/conc_dir_") + char(char(dir) + '0')
				+ std::string("_") + tmpstamp("%Y_%m_%d_%H_%M_%S") 
				+ std::string(".vol"));			
			solverEigen.resultToVolume(volume.getChannel(IDConc));
			bool res = blib::saveVolumeBinary(exportPath.c_str(), volume.getChannel(IDConc));
			if (argVerbose) {
				std::cout << "export to" << (exportPath) << std::endl;
			}
		}

	}



	/*
	Output result
	*/
	{
		bool isNewFile = true;

		//Check if output file exists
		if(argOutput){
			std::ifstream f(argOutput.Get());	
			if(f.is_open())
				isNewFile = false;
		}

		//If output file, open it
		std::ofstream outFile;

		if (argOutput) {
			outFile.open(argOutput.Get(), std::ios::app);
		}
		//Choose output stream
		std::ostream & os = (outFile.is_open()) ? outFile : std::cout;

		//Header 
		if(isNewFile){
			os << "path,porosity,dir,tau,t,dimx,dimy,dimz,solver" << '\n';
		}

		for(auto i =0 ; i < taus.size(); i++){
			os << "'" << fs::absolute(fs::path(argInput.Get())) << "'" << ",\t";

			os << porosity << ",\t";

			os << dirString(dirs[i]) << ",\t";
			
			os << taus[i] << ",\t";
			

			//double avgTime = std::accumulate(times.begin(), times.end(), 0.0) / times.size();
			os << times[i] << ",\t";
			//os << avgTime << ",\t";

			auto dim = c.dim();
			os << dim.x << ",\t" << dim.y << ",\t" << dim.z;


			os << ",\t" << argSolver.Get();

			os << "\n";
		}
	}

	return true;

}



int main(int argc, char **argv){

		
	try{
		parser.ParseCLI(argc, argv);
	}	
	catch (args::Help){
		std::cout << parser;
		return 0;
	}
	catch (args::Error e) {
		std::cerr << e.what() << std::endl;
		std::cerr << parser;
		return 1;
	}	

	

	bool res = true;

	/*if (argPrecision.Get() == "float")
		res &= tortuosity<float>();
	else
		res &= tortuosity<double>();*/


	//cudaVerify();

	res &= tortuosity<double>();


	
	if (res) return 0;
	return 2;
}