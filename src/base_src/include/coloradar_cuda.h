#ifndef COLORADAR_CUDA_H
#define COLORADAR_CUDA_H

#include "coloradar_tools.h"

#include <cuda_runtime.h>
#include <cuComplex.h>
#include <vector>
#include <complex>


namespace coloradar {

    std::vector<float> cubeToHeatmap(const std::vector<std::complex<double>>& datacube, coloradar::RadarConfig* config);

}

#endif
