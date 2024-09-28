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
    // heatmap params
    int numRangeBins;
    int numElevationBins;
    int numAzimuthBins;
    double rangeBinWidth;
    std::vector<double> azimuthBins;
    std::vector<double> elevationBins;

    // antenna params
    double designFrequency;
    int numTxAntennas;
    int numRxAntennas;
    std::vector<int> txDistance;
    std::vector<int> txElevation;
    std::vector<int> rxDistance;
    std::vector<int> rxElevation;

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

    // phase calibration params
    std::vector<double> frequencyCalibMatrix;
    std::vector<std::complex<double>> phaseCalibMatrix;

    // internal params
    int numVirtualElements;
    int* virtualArrayMap = nullptr;
    double* rangeWindowFunc = nullptr;
    double* dopplerWindowFunc = nullptr;

    RadarConfig() = default;
    ~RadarConfig();
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


template <Pcl4dPointType PointT, template <PclCloudType> class CloudT> void octreeToPcl(const octomap::OcTree& tree, CloudT<PointT>& cloud);
template <PclPointType PointT, template <PclCloudType> class CloudT> void filterFov(CloudT<PointT>& cloud, const float& horizontalFov, const float& verticalFov, const float& range);


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
    std::filesystem::path radarHeatmapsDirPath;
    std::filesystem::path radarCubesDirPath;

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

    std::vector<std::complex<double>> getDatacube(const std::filesystem::path& binFilePath, RadarConfig* config);
    std::vector<std::complex<double>> getDatacube(const int& cubeIdx, RadarConfig* config);
    std::vector<float> getHeatmap(const std::filesystem::path& binFilePath, RadarConfig* config);
    std::vector<float> getHeatmap(const int& hmIdx, RadarConfig* config);

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
