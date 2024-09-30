#pragma once
#define PCL_NO_PRECOMPILE
#include <defines.h>
#include <dca_types.h>
#include <mutex>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <condition_variable>
#include <chrono>
#include <deque>
#include <ThreadSafeDeque.h>
#include <vector>
#include <algorithm>
#include <string>
#include <sstream>
#include <complex>
#include <stdexcept>


struct Position
{
  int x_;
  int y_;

  Position()
  {
    x_ = 0;
    y_ = 0;
  }

  Position(int x, int y)
  {
    x_ = x;
    y_ = y;
  }

  int& x()
  {
    return x_;
  }

  int& y()
  {
    return y_;
  }
};

typedef std::vector<Position> PositionList;


struct RadarPoint
{
  PCL_ADD_POINT4D;
  float intensity;
  float range;
  float doppler;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;

POINT_CLOUD_REGISTER_POINT_STRUCT (RadarPoint,
                                  (float, x, x)
                                  (float, y, y)
                                  (float, z, z)
                                  (float, intensity, intensity)
                                  (float, range, range)
                                  (float, doppler, doppler))

typedef pcl::PointCloud<RadarPoint> RadarPointCloud;

struct Chirp
{
  Chirp(){}

  Chirp(int num_tx, int num_rx, int samples_per_chirp)
  {
    setParams(num_tx, num_rx, samples_per_chirp);
  }

  void setParams(int num_tx, int num_rx, int samples_per_chirp)
  {
    num_tx_ = num_tx;
    num_rx_ = num_rx;
    num_adc_samples_per_chirp_ = samples_per_chirp;

    samples_.resize(num_rx_);
    for (int i = 0; i < num_rx_; i++)
    {
      samples_[i].resize(num_tx_);
      for (int j = 0; j < num_tx_; j++)
        samples_[i][j].resize(num_adc_samples_per_chirp_, std::complex<double>(0,0));
    }
  }

  std::complex<double>& operator()(int rx_idx, int tx_idx, int sample_idx)
  {
    if (rx_idx >= num_rx_)
      throw std::out_of_range("receiver index out of range");
    if (tx_idx >= num_tx_)
      throw std::out_of_range("transmitter index out of range");
    if (sample_idx >= num_adc_samples_per_chirp_)
      throw std::out_of_range("sample index out of range");

    return samples_[rx_idx][tx_idx][sample_idx];
  }

  std::vector<std::complex<double>>& operator()(int rx_idx, int tx_idx)
  {
    if (rx_idx >= num_rx_)
      throw std::out_of_range("receiver index out of range");
    if (tx_idx >= num_tx_)
      throw std::out_of_range("transmitter index out of range");

    return samples_[rx_idx][tx_idx];
  }

  size_t size()
  {
    return num_adc_samples_per_chirp_;
  }

  std::vector<std::vector<std::vector<std::complex<double>>>> samples_;
  int num_tx_;
  int num_rx_;
  int num_adc_samples_per_chirp_;
};

typedef struct Chirp Chirp;

