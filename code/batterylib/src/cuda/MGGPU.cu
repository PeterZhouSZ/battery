#include "MGGPU.cuh"




__global__ void ___generateDomain(
	const MGGPU_Volume & binaryMask,
	double value_zero,
	double value_one,
	MGGPU_Volume & output
) {
	VOLUME_VOX_GUARD(output.res);	

	//Read mask
	uchar c = read<uchar>(binaryMask.surf, vox);		

	//Write value
	write<double>(output.surf, vox, (c > 0) ? value_one : value_zero);
}



void MGGPU_GenerateDomain(
	const MGGPU_Volume & binaryMask,
	double value_zero,
	double value_one,
	MGGPU_Volume & output
) {

	BLOCKS3D(2, output.res);	
	___generateDomain<< < numBlocks, block >> > (binaryMask,value_zero, value_one, output);


}