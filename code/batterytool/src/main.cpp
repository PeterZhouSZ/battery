#include <iostream>

#include <args.h>

#include <batterylib/include/DiffusionSolver.h>
#include <batterylib/include/VolumeIO.h>

#include <GLFW/glfw3.h>

#include <chrono>
#include <numeric>
#include <filesystem>
#include <fstream>
namespace fs = std::experimental::filesystem;

using namespace std;

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

	/*int dirSign = +1;
	char dirChar = argTauDir.Get()[0];
	if (argTauDir.Get().length() > 1) {
	if (argTauDir.Get()[1] == '-')
	dirSign = -1;
	}
	int dirIndex = (dirChar == 'x') ? 0 : ((dirChar == 'y') ? 1 : 2);
	Dir tauDir = getDir(dirIndex, dirSign);*/
	

}

int main(int argc, char **argv){

	using T = float;

	args::ArgumentParser parser("Battery tool", "Vojtech Krs (2018) vkrs@purdue.edu");
	args::HelpFlag help(parser, "help", "", { 'h', "help" });
	

	args::Positional<std::string> argInput(parser, "input", "Input file", args::Options::Required);
	args::ValueFlag<std::string> argOutput(parser, "output", "Output file", { 'o', "output" }, "");

	
	args::Group group(parser, "Output (select at least one):", args::Group::Validators::AtLeastOne);
	
	args::Flag argTau(group, "t", "Tortuosity", { 't', "tau", "tortuosity" });
	args::ValueFlag<std::string> argTauDir(group, "string", "Direction (x|y|z)|all|pos|neg", { 'd', "dir" }, "x-");
	args::ValueFlag<int> argTol(group, "tolerance", "Tolerance 1e-k", {"tol"}, 6);
	args::ValueFlag<int> argMaxIterations(group, "maxIterations", "Max Iterations", {"iter"}, 10000);

	args::ValueFlag<uint> argSubvolume(group, "subvolume", "Sub Volume", { "sub" }, 0);

	


	args::Flag argPorosity(group, "p", "Porosity", { 'p', "porosity" });
	args::Flag argTime(group, "time", "Time", { "time" });
	args::Flag argVerbose(group, "v", "Verbose", { 'v', "verbose" });
	args::CompletionFlag completion(parser, { "complete" });
	
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

	
	
	if (argVerbose) {
		//std::cout << "Direction: " << argTauDir.Get() << " (" << int(tauDir) << ")" << std::endl;
		std::cout << "Tolerance: " << argTol.Get() << std::endl;
		std::cout << "MaxIterations: " << argMaxIterations.Get() << std::endl;

	}


	
	
	


	 
	


	// Volume uses glTexture3D -> need opengl context
	///todo create context without a window
	glfwInit();
	auto wh = glfwCreateWindow(1, 1, "", NULL,NULL);
	glfwMakeContextCurrent(wh);	 
	

	


	
	blib::Volume volume;
	

	uint IDMask = 0;
	try {		
		IDMask = volume.emplaceChannel(
			blib::loadTiffFolder(argInput.Get().c_str(), false)
		);
	}
	catch (const char * ex) {
		std::cerr << "Failed to load: ";
		std::cerr << ex << std::endl;		
		return 1;
	}

	blib::VolumeChannel & c = volume.getChannel(IDMask);

	if (argSubvolume.Get() != 0) {
		int newDim = argSubvolume.Get();
		auto dim = c.dim();
		if (newDim > dim.x || newDim > dim.y || newDim > dim.z) {
			std::cerr << "Subvolume larger than volume " << dim.x << " x " << dim.y << " x " << dim.z << std::endl;
			return 1;
		}
		c.resize({ 0,0,0 }, blib::ivec3(newDim));
	}

	if (argVerbose) {
		auto dim = c.dim();
		std::cout << "Resolution: " << dim.x << " x " << dim.y << " x " << dim.z <<
			" = " << dim.x*dim.y*dim.z << " voxels "
			<< "(" << (dim.x*dim.y*dim.z) / (1024 * 1024.0f) << "M)" << std::endl;

	
	}

	
	std::vector<Dir> dirs = getDirs(argTauDir.Get());

	std::vector<T> taus(dirs.size());
	std::vector<double> times(dirs.size());
	
	T d0 = 1.0f;
	T d1 = 0.001f;

	blib::DiffusionSolver<T> solver(argVerbose);

	T porosity;

	for (auto i = 0; i < dirs.size(); i++) {
		auto dir = dirs[i];
		if (argVerbose) {
			std::cout << "Direction " << dir << std::endl;
		}

		auto t0 = std::chrono::system_clock::now();
		solver.prepare(
			c,
			dir,
			d0,
			d1
		);
		
		T tol = T(pow(10.0, -argTol.Get()));
		solver.solve(tol, argMaxIterations.Get());		

		taus[i] = solver.tortuosity(c, dir);
		porosity = solver.porosity();	

		auto t1 = std::chrono::system_clock::now();

		std::chrono::duration<double> dt = t1 - t0;
		times[i] = dt.count();
	}

	double avgTime = std::accumulate(times.begin(), times.end(),0.0) / times.size();


	{
		//If output file, open it
		std::ofstream outFile;
		if (argOutput) {
			outFile.open(argOutput.Get());
		}
		//Choose output stream
		std::ostream & os = (outFile.is_open()) ? outFile : std::cout;

		os << "'" << fs::absolute(fs::path(argInput.Get())) << "'" << ",\t";

		os << porosity << ",\t";
		for (auto & tau : taus) {
			std::cout << tau << ",\t";
		}

		os << avgTime << ",\t";

		auto dim = c.dim();
		os << dim.x << ",\t" << dim.y << ",\t" << dim.z;

		os << "\n";
	}

	
	return 0;

}