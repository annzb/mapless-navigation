#include <fstream>
#include <sstream>
#include <thread>
#include <math.h>
#include <Eigen/Core>
#include <mutex>

#include <cufft.h>
#include "cuda_kernels.h"
#include "coloradar_cuda.h"


double blackman(int i, int n) {
    double a0 = 0.42;
    double a1 = 0.5;
    double a2 = 0.08;
    return (a0 - a1 * cos((2.0 * M_PI * double(i)) / double(n)) + a2 * cos((4.0 * M_PI * double(i)) / double(n)));
}


std::vector<float> coloradar::cubeToHeatmap(const std::vector<std::complex<double>>& datacube, coloradar::RadarConfig* config) {
    bool collapse_doppler_ = false;
    bool remove_antenna_coupling_ = true;
    bool phase_freq_calib_ = false;
    
    cufftHandle range_plan_; // fft plan for range fft
    cufftHandle doppler_plan_; // fft plan for doppler fft
    cufftHandle angle_plan_; // fft plan for angle of arrival fft
    cuDoubleComplex* coupling_signature_; // coupling signature
    cuDoubleComplex* phase_calib_mat_; // phase calibration matrix
    cuDoubleComplex* freq_calib_mat_; // frequency calibration matrix
    cuDoubleComplex* range_fft_data_; // range fft data buffer
    cuDoubleComplex* doppler_fft_data_; // doppler fft data buffer
    cuDoubleComplex* angle_fft_data_; // angle fft data buffer
    double* range_window_func_; // range window function values
    double* doppler_window_func_; // doppler window function values
    double* az_window_func_; // azimuth window function values
    double* el_window_func_; // elevation window function values
    float* magnitudes_out_; // complex magnitudes of aoa fft output for publishing
    float* static_bins_; // only the static doppler bins of the steered output
    int16_t* int_frame_data_; // container for int-valued adc data from ros message
    std::vector<std::complex<double>> out_data_;
   
    // set calibration parameters and allocate memory for calibration
    int num_coupling_frames_ = 2;
    int couple_idx_ = 0;
    int max_val_ = 0.0;
    cudaMalloc(&coupling_signature_, sizeof(cuDoubleComplex) * config->numPosRangeBins * config->numTxAntennas * config->numRxAntennas);
    cudaMalloc(&freq_calib_mat_, sizeof(cuDoubleComplex) * config->numRangeBins * config->numTxAntennas * config->numRxAntennas);
    cudaMalloc(&phase_calib_mat_, sizeof(cuDoubleComplex) * config->numTxAntennas * config->numRxAntennas);

    coupling_signature_.resize(config->numPosRangeBins * config->numRxAntennas * config->numTxAntennas, std::complex<double>(0,0));
    int rank = 1;
    int angle_rank = 2;
    int n_range [1] = {config->numRangeBins};
    int n_doppler [1] = {config->numDopplerBins};
    int n_angle [2] = {config->numAzimuthBeams, config->numElevationBeams};
    int howmany_range = config->numTxAntennas * config->numRxAntennas * config->numDopplerBins;
    int howmany_doppler = config->numTxAntennas * config->numRxAntennas * config->numPosRangeBins;
    int howmany_angle = config->numPosRangeBins * config->numDopplerBins;
    int range_dist = config->numRangeBins;
    int doppler_dist = 1;
    int angle_dist = config->numAzimuthBeams * config->numElevationBeams;
    int range_stride = 1;
    int doppler_stride = config->numPosRangeBins * config->numTxAntennas * config->numRxAntennas;
    int angle_stride = 1;
    int *range_embed = n_range;
    int *doppler_embed = n_doppler;
    int *angle_embed = n_angle;
    cudaMalloc(&int_frame_data_, sizeof(int16_t) * 2 * config->numRangeBins * config->numDopplerBins * config->numTxAntennas * config->numRxAntennas);
    cudaMalloc(&range_fft_data_, sizeof(cuDoubleComplex) * config->numRangeBins * config->numDopplerBins * config->numTxAntennas * config->numRxAntennas);
    cudaMalloc(&doppler_fft_data_, sizeof(cuDoubleComplex) * config->numPosRangeBins * config->numDopplerBins * config->numTxAntennas * config->numRxAntennas);
    cudaMalloc(&angle_fft_data_, sizeof(cuDoubleComplex) * config->numPosRangeBins * config->numDopplerBins * config->numAzimuthBeams * config->numElevationBeams);
    cudaMalloc(&magnitudes_out_, sizeof(float) * config->numPosRangeBins * config->numDopplerBins * config->numAzimuthBeams * config->numElevationBeams);
    if (collapse_doppler_)
        cudaMalloc(&static_bins_, sizeof(float) * 2 * config->numPosRangeBins * config->numAzimuthBeams * config->numElevationBeams);
    cufftPlanMany(&range_plan_, rank, n_range, range_embed, range_stride, range_dist, range_embed, range_stride, range_dist, CUFFT_Z2Z , howmany_range);
    cufftPlanMany(&doppler_plan_, rank, n_doppler, doppler_embed, doppler_stride, doppler_dist, doppler_embed, doppler_stride, doppler_dist, CUFFT_Z2Z, howmany_doppler);
    cufftPlanMany(&angle_plan_, angle_rank, n_angle, angle_embed, angle_stride, angle_dist, angle_embed, angle_stride, angle_dist, CUFFT_Z2Z, howmany_angle);

    out_data_.resize(config->numPosRangeBins * config->numDopplerBins * config->numAngles);
    max_val_ = 0.0;
        cudaMalloc(&range_window_func_, sizeof(double) * config->numRangeBins);
    cudaMalloc(&doppler_window_func_, sizeof(double) * config->numDopplerBins);
    cudaMalloc(&az_window_func_, sizeof(double) * config->azimuthApertureLen);
    cudaMalloc(&el_window_func_, sizeof(double) * config->elevationApertureLen);

    std::vector<double> range_window_local(config->numRangeBins);
    std::vector<double> doppler_window_local(config->numDopplerBins);
    std::vector<double> az_window_local(config->azimuthApertureLen);
    std::vector<double> el_window_local(config->elevationApertureLen);
    for (int range_idx = 0; range_idx < config->numRangeBins; range_idx++)
      range_window_local[range_idx] = blackman(range_idx, config->numRangeBins);
    for (int doppler_idx = 0; doppler_idx < config->numDopplerBins; doppler_idx++)
      doppler_window_local[doppler_idx] = blackman(doppler_idx, config->numDopplerBins);
    for (int az_idx = 0; az_idx < config->azimuthApertureLen; az_idx++)
      az_window_local[az_idx] = blackman(az_idx, config->azimuthApertureLen);
    for (int el_idx = 0; el_idx < config->elevationApertureLen; el_idx++)
      el_window_local[el_idx] = blackman(el_idx, config->elevationApertureLen);
    cudaMemcpy(range_window_func_, &range_window_local[0], sizeof(double) * config->numRangeBins, cudaMemcpyDefault);
    cudaMemcpy(doppler_window_func_, &doppler_window_local[0], sizeof(double) * config->numDopplerBins, cudaMemcpyDefault);
    cudaMemcpy(az_window_func_, &az_window_local[0], sizeof(double) * config->azimuthApertureLen, cudaMemcpyDefault);
    cudaMemcpy(el_window_func_, &el_window_local[0], sizeof(double) * config->elevationApertureLen, cudaMemcpyDefault);
               
    cudaMemcpy(int_frame_data_, &datacube[0], sizeof(int16_t) * 2 * config->numRangeBins * config->numDopplerBins * config->numTxAntennas * config->numRxAntennas, cudaMemcpyDefault);
    setFrameData(config->numRangeBins, config->numDopplerBins, config->numTxAntennas, config->numRxAntennas, int_frame_data_, range_fft_data_);
    if (phase_freq_calib_)
        applyPhaseFreqCal(config->numRangeBins, config->numDopplerBins, config->numTxAntennas, config->numRxAntennas, range_fft_data_, freq_calib_mat_, phase_calib_mat_);

    applyWindow(config->numRangeBins, 1, config->numRangeBins, config->numTxAntennas * config->numRxAntennas * config->numDopplerBins, range_window_func_, range_fft_data_);
    // run range fft
    cufftExecZ2Z(range_plan_, range_fft_data_, range_fft_data_, CUFFT_FORWARD);
    cudaDeviceSynchronize();

    // remove DC and negative frequency values from the range fft output
    removeNegSpectrum(config->numRangeBins, config->numPosRangeBins, config->numDopplerBins, config->numTxAntennas * config->numRxAntennas, range_fft_data_, doppler_fft_data_);
    
    if (remove_antenna_coupling_ && couple_idx_ >= num_coupling_frames_)
        removeCoupling(config->numPosRangeBins, config->numDopplerBins, config->numTxAntennas * config->numRxAntennas, doppler_fft_data_, coupling_signature_);

    // apply doppler window function
    applyWindow(1, config->numPosRangeBins * config->numTxAntennas * config->numRxAntennas, config->numDopplerBins, config->numTxAntennas * config->numRxAntennas * config->numPosRangeBins, doppler_window_func_, doppler_fft_data_);
    // run doppler fft
    cufftExecZ2Z(doppler_plan_, doppler_fft_data_, doppler_fft_data_, CUFFT_FORWARD);
    cudaDeviceSynchronize();

    // memset angle fft data back to zero
    // entries that are unset after array is filled with
    // samples will become zero padding
    cudaMemset(angle_fft_data_, 0, sizeof(cuDoubleComplex) * config->numAzimuthBeams * config->numElevationBeams * config->numPosRangeBins * config->numDopplerBins);

    // move doppler fft result into angle fft data array
    // and apply azimuth and elevation window functions
    // not using the applyWindow kernel because it's not compatible
    // with the data layout required for the angle fft
    rearrangeData(config->numPosRangeBins, config->numDopplerBins, config->numTxAntennas, config->numRxAntennas, config->numAzimuthBeams, config->numElevationBeams, config->numVirtualElements, config->virtualArrayMap, az_window_func_, el_window_func_, doppler_fft_data_, angle_fft_data_);
    // run angle fft
    cufftExecZ2Z(angle_plan_, angle_fft_data_, angle_fft_data_, CUFFT_FORWARD);
    cudaDeviceSynchronize();

    // reorder data for publication
    // includes rearranging the doppler, azimuth, and elevation dimensions
    // so zero frequency is centered (fftshift in Matlab and SciPy)
    assembleMsg(config->numPosRangeBins, config->numDopplerBins, config->numAzimuthBeams, config->numElevationBeams, angle_fft_data_, magnitudes_out_);
    if (collapse_doppler_)
        collapseDoppler(config->numPosRangeBins, config->numDopplerBins, config->numAngles, config->dopplerBinWidth, magnitudes_out_, static_bins_);
        
    std::memcpy(config->elevationBins[0], &config->elevationAngles[0], sizeof(float) * config->numElevationBeams);
    std::memcpy(config->azimuthBins[0], &config->azimuthAngles[0], sizeof(float) * config->numAzimuthBeams);

    std::vector<float> image;
    if (collapse_doppler_) {
      image.resize(2 * config->numAngles * config->numPosRangeBins);
    } else {
      image.resize(config->numAngles * config->numPosRangeBins * config->numDopplerBins);
    }
    if (collapse_doppler_)
      cudaMemcpy(&image[0], static_bins_, sizeof(float) * 2 * config->numPosRangeBins * config->numAngles, cudaMemcpyDefault);
    else
      cudaMemcpy(&image[0], magnitudes_out_, sizeof(float) * config->numPosRangeBins * config->numDopplerBins * config->numAngles, cudaMemcpyDefault);
    return image;

}
