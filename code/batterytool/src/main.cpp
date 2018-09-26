#include <iostream>

#include <args.h>

#include <batterylib/include/VolumeMeasures.h>
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
args::ValueFlag<std::string> argSolver(group, "string", "Solver (BICGSTABGPU|BICGSTABCPU|MGGPU)", { "solver" }, "BICGSTABGPU");
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

	if (flagStr == "xneg")
		return { X_NEG};
	if (flagStr == "yneg")
		return { Y_NEG };
	if (flagStr == "zneg")
		return { Z_NEG };


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

		if(argVolumeExport.Get())
			IDConc = volume.addChannel(volume.getChannel(IDMask).dim(), ((std::is_same<T, float>()) ? TYPE_FLOAT : TYPE_DOUBLE) );

	}
	catch (const char * ex) {
		std::cerr << "Failed to load: ";
		std::cerr << ex << std::endl;
		return false;
	}

	blib::VolumeChannel & c = volume.getChannel(IDMask);
	c.getCurrentPtr().allocCPU();
	c.getCurrentPtr().retrieve();


	

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
	
	std::vector<double> times(dirs.size());	

	blib::DiffusionSolverType solverType = DSOLVER_BICGSTABGPU;
	if (argSolver.Get() == "Eigen")
		solverType = DSOLVER_EIGEN;
	else if (argSolver.Get() == "MGGPU")
		solverType = DSOLVER_MGGPU;

	TortuosityParams tp;
	tp.verbose = argVerbose.Get();
	tp.coeffs = { 1.0, 0.001 };
	tp.maxIter = argMaxIterations.Get();
	tp.tolerance = double(pow(10.0, -argTol.Get()));

	tp.porosity = getPorosity<double>(c);
	tp.porosityPrecomputed = true;

	//Get area density
	c.getCurrentPtr().createTexture();
	T areaDensity = getReactiveAreaDensity<T>(c, c.dim(), 0.1f, 1.0f);
	

	for (auto i = 0; i < dirs.size(); i++) {
		

		tp.dir = dirs[i];

		if (argVerbose) {
			std::cout << "Direction " << tp.dir << std::endl;
		}

		auto t0 = std::chrono::system_clock::now();

		blib::VolumeChannel *outPtr = (argVolumeExport.Get()) ? &volume.getChannel(IDConc) : nullptr;		
		T tau = getTortuosity<T>(c, tp, solverType, outPtr);		

		taus[i] = tau;

		

		//Calculate tortuosity and porosity				
		auto t1 = std::chrono::system_clock::now();


		std::chrono::duration<double> dt = t1 - t0;
		times[i] = dt.count();

		if (argVerbose) {
			std::cout << "Elapsed: " << dt.count() << "s (" << dt.count() / 60.0 << "m)" << std::endl;
		}

		//Export calculated concetration volume
		if (argVolumeExport.Get()) {
			const std::string exportPath = (argInput.Get() + std::string("/conc_dir_") + char(char(tp.dir) + '0')
				+ std::string("_") + tmpstamp("%Y_%m_%d_%H_%M_%S") 
				+ std::string(".vol"));			
			
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
			os << "path,porosity,dir,tau,alpha,t,dimx,dimy,dimz,solver" << '\n';
		}

		for(auto i =0 ; i < taus.size(); i++){
			os << "'" << fs::absolute(fs::path(argInput.Get())) << "'" << ",\t";

			os << tp.porosity << ",\t";

			os << dirString(dirs[i]) << ",\t";
			
			os << taus[i] << ",\t";

			os << areaDensity << ",\t";
			

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