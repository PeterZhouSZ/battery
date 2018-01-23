#include "VolumeIO.h"


#include "tinytiff/tinytiffreader.h"

#include <filesystem>

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

	TinyTIFFReader_close(tiffr);
	return true;
}

size_t directoryFileCount(const char * path)
{
	return std::count_if(
		fs::directory_iterator(path),
		fs::directory_iterator(),
		static_cast<bool(*)(const fs::path&)>(fs::is_regular_file));
}



VolumeChannel blib::loadTiffFolder(const char * folder)
{
		

	fs::path path(folder);

	if (!fs::is_directory(path))
		throw "Volume directory not found";


	int numSlices = static_cast<int>(directoryFileCount(path.string().c_str()));

	int x, y, bytes;

	//Find first tiff
	for (auto & f : fs::directory_iterator(path)) {
		if (fs::is_directory(f)) continue;		

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
		
		readTiff(f.path().string().c_str(), ptr + (sliceIndex * x * y));

		sliceIndex++;
	}	

	volume.getCurrentPtr().commit();

	return volume;
}


