#include "VolumeIO.h"


#include "tinytiff/tinytiffreader.h"


#if defined(__GNUC__)
    #include <experimental/filesystem>
#else
	#include <filesystem>
#endif

#include <fstream>
#include <iostream>
#include <algorithm>

using namespace std;
using namespace blib;
namespace fs = std::experimental::filesystem;


bool tiffSize(const char * path, int *x, int *y, int * bytes, int *frames = nullptr)
{

	TinyTIFFReaderFile* tiffr = NULL;
	tiffr = TinyTIFFReader_open(path);

	if (!tiffr) return false;

	uint32_t width = TinyTIFFReader_getWidth(tiffr);
	uint32_t height = TinyTIFFReader_getHeight(tiffr);

	*x = width;
	*y = height;

	*bytes = TinyTIFFReader_getSampleFormat(tiffr);

	//Count frames
	if (frames != nullptr) {
		int cnt = 0;
		do { cnt++; } while (TinyTIFFReader_readNext(tiffr));

		*frames = cnt;
	}

	return true;
}

bool readTiff(const char * path, void * buffer)
{
	TinyTIFFReaderFile* tiffr = NULL;
	tiffr = TinyTIFFReader_open(path);
	if (!tiffr) return false;

	uint32_t width = TinyTIFFReader_getWidth(tiffr);
	uint32_t height = TinyTIFFReader_getHeight(tiffr);

	TinyTIFFReader_getSampleData(tiffr, buffer, 0);
	if (TinyTIFFReader_wasError(tiffr)) {
		std::cerr << "TinyTIFFReader Error: " << TinyTIFFReader_getLastError(tiffr) << std::endl;
		return false;
	}

	TinyTIFFReader_close(tiffr);
	return true;
}

size_t directoryFileCount(const std::string & path, const std::string & ext)
{
	return std::count_if(
		fs::directory_iterator(path),
		fs::directory_iterator(),
		[&ext](fs::path p) {
		return fs::is_regular_file(p) && p.extension() == ext;
	}
	);
		//static_cast<bool(*)(const fs::path&)>(fs::is_regular_file));
}



VolumeChannel blib::loadTiffFolder(const char * folder, bool commitToGPU)
{
		

	fs::path path(folder);

	if (!fs::is_directory(path))
		throw "Volume directory not found";


	int numSlices = static_cast<int>(directoryFileCount(path.string(), ".tiff"))
					+ static_cast<int>(directoryFileCount(path.string(), ".tif"));

	int x, y, bytes;

	//Find first tiff
	for (auto & f : fs::directory_iterator(path)) {
		if (fs::is_directory(f)) continue;		
		if (f.path().extension() != ".tiff" && f.path().extension() != ".tif") continue;
		

		if (!tiffSize(f.path().string().c_str(), &x, &y, &bytes))
			throw "Couldn't read tiff file";
		else
			break;
	}
	

	if (bytes != 1)
		throw "only uint8 supported right now";
	

	VolumeChannel volume({ x,y,numSlices }, TYPE_UCHAR, false) ;

	uchar * ptr = (uchar*)volume.getCurrentPtr().getCPU();

	

	
	uint sliceIndex = 0;
	for (auto & f : fs::directory_iterator(path)) {
		if (fs::is_directory(f)) continue;		
		if (f.path().extension() != ".tiff" && f.path().extension() != ".tif") continue;
		
		if (!readTiff(f.path().string().c_str(), ptr + (sliceIndex * x * y))) {
			throw "Failed to read slices";
		}

		sliceIndex++;
	}	

	if(commitToGPU)
		volume.getCurrentPtr().commit();

	return volume;
}

BLIB_EXPORT bool blib::saveVolumeBinary(const char * path, const VolumeChannel & channel)
{

	std::ofstream f(path, std::ios::binary);
	if (!f.good()) return false;

	const auto & dataptr = channel.getCurrentPtr();
	const void * data = dataptr.getCPU();

	bool doubleBuffered = channel.isDoubleBuffered();

	PrimitiveType type = channel.type();
	ivec3 dim = channel.dim();

	f.write((const char *)&type, sizeof(PrimitiveType));
	f.write((const char *)&dim, sizeof(ivec3));
	f.write((const char *)&doubleBuffered, sizeof(bool));
	f.write((const char *)data, dataptr.byteSize());
	f.close();

	return true;
}

BLIB_EXPORT VolumeChannel blib::loadVolumeBinary(const char * path)
{

	std::ifstream f(path, std::ios::binary);
	if (!f.good()) 
		throw "Couldn't read file";

	PrimitiveType type;
	ivec3 dim;
	bool doubleBuffered;

	f.read((char *)&type, sizeof(PrimitiveType));
	f.read((char *)&dim, sizeof(ivec3));
	f.read((char *)&doubleBuffered, sizeof(bool));


	VolumeChannel vol(dim, type, doubleBuffered);
	auto & dataptr = vol.getCurrentPtr();
	void * data = dataptr.getCPU();
	
	f.read((char *)data, dataptr.byteSize());

	dataptr.commit();

	return vol;
}

