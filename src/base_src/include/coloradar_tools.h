#ifndef COLORADAR_TOOLS_H
#define COLORADAR_TOOLS_H

#include "utils.h"

#include <string>
#include <vector>
#include <map>
#include <Eigen/Dense>


namespace coloradar {

class RadarConfig {
protected:
    virtual void init(const std::filesystem::path& calibDir) = 0;
    void initAntennaParams(const std::filesystem::path& antennaCfgFile);
    void initHeatmapParams(const std::filesystem::path& heatmapCfgFile);
    void initWaveformParams(const std::filesystem::path& waveformCfgFile);
    void initCouplingParams(const std::filesystem::path& couplingCfgFile);
    void initPhaseFrequencyParams(const std::filesystem::path& phaseFrequencyCfgFile);
    void initInternalParams();

public:
    double c = 299792458; // speed of light in m/s

    // heatmap params
    int numRangeBins;
    int numPosRangeBins;
    int numElevationBins;
    int numAzimuthBins;
    double rangeBinWidth;
    std::vector<float> azimuthBins;
    std::vector<float> elevationBins;

    // antenna params
    double designFrequency;
    int numTxAntennas;
    int numRxAntennas;
    std::vector<pcl::PointXY> txCenters;
    std::vector<pcl::PointXY> rxCenters;

    // waveform params
    int numAdcSamplesPerChirp;
    int numChirpsPerFrame;
    int adcSampleFrequency;
    double startFrequency;
    double idleTime;
    double adcStartTime;
    double rampEndTime;
    double frequencySlope;

    // calibration params
    int numDopplerBins;
    std::vector<std::complex<double>> couplingCalibMatrix;

    // phase frequency params
    int calibAdcSampleFrequency;
    double calibFrequencySlope;
    std::vector<std::complex<double>> frequencyCalibMatrix;
    std::vector<std::complex<double>> phaseCalibMatrix;

    // internal params
    int numAzimuthBeams;
    int numElevationBeams;
    int azimuthApertureLen;
    int elevationApertureLen;
    int numAngles;
    int numVirtualElements;
    std::vector<int> virtualArrayMap;
    std::vector<float> azimuthAngles;
    std::vector<float> elevationAngles;
    double dopplerBinWidth;

    RadarConfig() = default;
};

class SingleChipConfig : public RadarConfig {
public:
    SingleChipConfig() = default;
    SingleChipConfig(const std::filesystem::path& calibDir);

protected:
    void init(const std::filesystem::path& calibDir) override;
};

class CascadeConfig : public RadarConfig {
public:
    CascadeConfig() = default;
    CascadeConfig(const std::filesystem::path& calibDir);

protected:
    void init(const std::filesystem::path& calibDir) override;
};

struct RadarPoint
{
  PCL_ADD_POINT4D;
  float intensity;
  float doppler;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;


template <Pcl4dPointType PointT, template <PclCloudType> class CloudT> void octreeToPcl(const octomap::OcTree& tree, CloudT<PointT>& cloud);
template <PclPointType PointT, template <PclCloudType> class CloudT> void filterFov(CloudT<PointT>& cloud, const float& horizontalFov, const float& verticalFov, const float& range);
pcl::PointCloud<RadarPoint> heatmapToPointcloud(const std::vector<float>& heatmap, coloradar::RadarConfig* config, const float& intensityThresholdPercent = 0.0);


class OctoPointcloud : public octomap::Pointcloud {
public:
    OctoPointcloud() = default;
    OctoPointcloud(const OctoPointcloud& other) : octomap::Pointcloud(other) {}
    template <PclPointType PointT, template <PclCloudType> class CloudT> OctoPointcloud(const CloudT<PointT>& cloud);

    template <PclCloudType CloudT> CloudT toPcl();

    void filterFov(const float& horizontalFovTan, const float& verticalFovTan, const float& range);
    void transform(const Eigen::Affine3f& transformMatrix);
    using octomap::Pointcloud::transform;
};


class ColoradarRun {
protected:
    std::filesystem::path runDirPath;
    std::filesystem::path posesDirPath;
    std::filesystem::path lidarScansDirPath;
    std::filesystem::path radarScansDirPath;
    std::filesystem::path cascadeScansDirPath;
    std::filesystem::path lidarCloudsDirPath;
    std::filesystem::path lidarMapsDirPath;
    std::filesystem::path cascadeHeatmapsDirPath;
    std::filesystem::path cascadeCubesDirPath;
    std::filesystem::path cascadePointcloudsDirPath;
    std::filesystem::path radarHeatmapsDirPath;
    std::filesystem::path radarCubesDirPath;
    std::filesystem::path radarPointcloudsDirPath;

    std::vector<double> readTimestamps(const std::filesystem::path& path);
    int findClosestEarlierTimestamp(const double& targetTs, const std::vector<double>& timestamps);

public:
    const std::string name;

    ColoradarRun(const std::filesystem::path& runPath);

    std::vector<double> getPoseTimestamps();
    std::vector<double> getLidarTimestamps();
    std::vector<double> getRadarTimestamps();
    std::vector<double> getCascadeTimestamps();
    std::vector<double> getCascadeCubeTimestamps();

    template<PoseType PoseT> std::vector<PoseT> getPoses();
    template<coloradar::PoseType PoseT> std::vector<PoseT> interpolatePoses(const std::vector<PoseT>& poses, const std::vector<double>& poseTimestamps, const std::vector<double>& targetTimestamps);

    template<PclCloudType CloudT> CloudT getLidarPointCloud(const std::filesystem::path& binPath);
    template<OctomapCloudType CloudT> CloudT getLidarPointCloud(const std::filesystem::path& binPath);
    template<CloudType CloudT> CloudT getLidarPointCloud(const int& cloudIdx);

    std::vector<int16_t> getDatacube(const std::filesystem::path& binFilePath, RadarConfig* config);
    std::vector<int16_t> getDatacube(const int& cubeIdx, RadarConfig* config);
    std::vector<float> getHeatmap(const std::filesystem::path& binFilePath, RadarConfig* config);
    std::vector<float> getHeatmap(const int& hmIdx, RadarConfig* config);
    std::vector<float> clipHeatmapImage(const std::vector<float>& image, const float& horizontalFov, const float& verticalFov, const float& range, coloradar::RadarConfig* config);
    std::vector<float> clipHeatmapImage(const std::vector<float>& image, const int& azimuthMaxBin, const int& elevationMaxBin, const int& rangeMaxBin, coloradar::RadarConfig* config);

    octomap::OcTree buildLidarOctomap(
        const double& mapResolution,
        const float& lidarTotalHorizontalFov,
        const float& lidarTotalVerticalFov,
        const float& lidarMaxRange,
        Eigen::Affine3f lidarTransform = Eigen::Affine3f::Identity()
    );
    void saveLidarOctomap(const octomap::OcTree& tree);
    pcl::PointCloud<pcl::PointXYZI> readLidarOctomap();

    void sampleMapFrames(
        const float& horizontalFov,
        const float& verticalFov,
        const float& range,
        const Eigen::Affine3f& mapPreTransform = Eigen::Affine3f::Identity(),
        std::vector<octomath::Pose6D> poses = {}
    );
    pcl::PointCloud<pcl::PointXYZI> readMapFrame(const int& frameIdx);

    void createRadarPointclouds(RadarConfig* config, const float& intensityThresholdPercent = 0.0);
};


class ColoradarDataset {
protected:
    std::filesystem::path coloradarDirPath;
    std::filesystem::path calibDirPath;
    std::filesystem::path transformsDirPath;
    std::filesystem::path runsDirPath;

    Eigen::Affine3f loadTransform(const std::filesystem::path& filePath);

public:
    SingleChipConfig singleChipConfig;
    CascadeConfig cascadeConfig;

    ColoradarDataset(const std::filesystem::path& coloradarPath);

    Eigen::Affine3f getBaseToLidarTransform();
    Eigen::Affine3f getBaseToRadarTransform();
    Eigen::Affine3f getBaseToCascadeRadarTransform();
    std::vector<std::string> listRuns();
    ColoradarRun getRun(const std::string& runName);

    void createMaps(
        const double& mapResolution,
        const float& lidarTotalHorizontalFov,
        const float& lidarTotalVerticalFov,
        const float& lidarMaxRange,
        const std::vector<std::string>& targetRuns = std::vector<std::string>()
    );
};

}

#include "pcl_functions.hpp"
#include "octo_pointcloud.hpp"
#include "coloradar_run.hpp"

#endif
