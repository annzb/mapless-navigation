#ifndef COLORADAR_CUDA_H
#define COLORADAR_CUDA_H

#include "coloradar_tools.h"
#include "cuda_kernels.h"

#include <cuda_runtime.h>
#include <cuComplex.h>
#include <vector>
#include <complex>


namespace coloradar {

    std::vector<float> cubeToHeatmap(std::vector<int16_t> datacube, coloradar::RadarConfig* config);

}

#endif
