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

template <typename T>
bool tortuosity() {

	


	if (argVerbose) {
		std::cout << "Precision: " << argPrecision.Get() << std::endl;
		std::cout << "Tolerance: " << argTol.Get() << std::endl;
		std::cout << "MaxIterations: " << argMaxIterations.Get() << " intermediate steps: " << argStep.Get() << std::endl;
	}


	//VolumeChannel.DataPtr uses glTexture3D -> need opengl context
	///todo create context without a window or implement dataptr without gl interop
	glfwInit();
	auto wh = glfwCreateWindow(1, 1, "", NULL, NULL);
	glfwMakeContextCurrent(wh);


	/*
	Load input volume
	*/
	blib::Volume volume;
	uint IDMask = 0;
	try {
		IDMask = volume.emplaceChannel(
			blib::loadTiffFolder(argInput.Get().c_str(), true)
		);
		volume.binarize(IDMask, 1.0f);

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
	uint concChannel = volume.addChannel(c.dim(), TYPE_FLOAT);

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

	blib::DiffusionSolver<T> solver(argVerbose);
	for (auto i = 0; i < dirs.size(); i++) {
		auto dir = dirs[i];
		if (argVerbose) {
			std::cout << "Direction " << dir << std::endl;
		}

		auto t0 = std::chrono::system_clock::now();
		//Prepare linear system
		solver.prepare(
			c,
			dir,
			d0,
			d1
		);

		T tol = T(pow(10.0, -argTol.Get()));
		//Solve to desired tolerance
		solver.solve(tol, argMaxIterations.Get(), argStep.Get());

		//Calculate tortuosity and porosity
		taus[i] = solver.tortuosity(c, dir);
		porosity = solver.porosity();
		auto t1 = std::chrono::system_clock::now();


		std::chrono::duration<double> dt = t1 - t0;
		times[i] = dt.count();

		if (argVerbose) {
			std::cout << "Elapsed: " << dt.count() << "s (" << dt.count() / 60.0 << "m)" << std::endl;
		}

		//Export calculated concetration volume
		if (argVolumeExport) {
			const std::string exportPath = (argInput.Get() + std::string("/conc_dir") + char(char(dir) + '0') + std::string(".vol"));
			solver.resultToVolume(volume.getChannel(concChannel));
			bool res = blib::saveVolumeBinary(exportPath.c_str(), volume.getChannel(concChannel));
			if (argVerbose) {
				std::cout << "export to" << (exportPath) << std::endl;
			}
		}

	}



	/*
	Output result
	*/
	{
		//If output file, open it
		std::ofstream outFile;
		if (argOutput) {
			outFile.open(argOutput.Get(), std::ios::app);
		}
		//Choose output stream
		std::ostream & os = (outFile.is_open()) ? outFile : std::cout;

		os << "'" << fs::absolute(fs::path(argInput.Get())) << "'" << ",\t";

		os << porosity << ",\t";
		for (auto & tau : taus) {
			os << tau << ",\t";
		}

		double avgTime = std::accumulate(times.begin(), times.end(), 0.0) / times.size();
		os << avgTime << ",\t";

		auto dim = c.dim();
		os << dim.x << ",\t" << dim.y << ",\t" << dim.z;

		os << "\n";
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

	if (argPrecision.Get() == "float")
		res &= tortuosity<float>();
	else
		res &= tortuosity<double>();


	
	if (res) return 0;
	return 2;
}