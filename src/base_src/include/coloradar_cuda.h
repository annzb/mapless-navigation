#ifndef COLORADAR_CUDA_H
#define COLORADAR_CUDA_H

#include "coloradar_tools.h"
#include "cuda_kernels.h"

#include <cuda_runtime.h>
#include <cuComplex.h>
#include <vector>
#include <complex>


namespace coloradar {
    template<typename T>
    void cudaCopy(T* dest, std::vector<T> source) {
        cudaError_t err = cudaMemcpy(dest, source.data(), sizeof(T) * source.size(), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            std::cerr << "CUDA memcpy error:  " << cudaGetErrorString(err) << std::endl;
        }
    }

    std::vector<float> cubeToHeatmap(std::vector<int16_t> datacube, coloradar::RadarConfig* config);

}

#endif
