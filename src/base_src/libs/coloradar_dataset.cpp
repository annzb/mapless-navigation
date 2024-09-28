#include "coloradar_tools.h"

#include <pcl/io/pcd_io.h>
#include <fstream>
#include <sstream>


coloradar::ColoradarDataset::ColoradarDataset(const std::filesystem::path& coloradarPath) : coloradarDirPath(coloradarPath) {
    coloradar::internal::checkPathExists(coloradarDirPath);
    calibDirPath = coloradarDirPath / "calib";
    coloradar::internal::checkPathExists(calibDirPath);
    transformsDirPath = calibDirPath / "transforms";
    coloradar::internal::checkPathExists(transformsDirPath);
    runsDirPath = coloradarDirPath / "kitti";
    coloradar::internal::checkPathExists(runsDirPath);
    singleChipConfig = coloradar::SingleChipConfig(calibDirPath);
    cascadeConfig = coloradar::CascadeConfig(calibDirPath);
}

Eigen::Affine3f coloradar::ColoradarDataset::getBaseToLidarTransform() {
    std::filesystem::path baseToLidarTransformPath = transformsDirPath / "base_to_lidar.txt";
    Eigen::Affine3f transform = loadTransform(baseToLidarTransformPath);
    return transform;
}

Eigen::Affine3f coloradar::ColoradarDataset::getBaseToRadarTransform() {
    std::filesystem::path baseToRadarTransformPath = transformsDirPath / "base_to_single_chip.txt";
    Eigen::Affine3f transform = loadTransform(baseToRadarTransformPath);
    return transform;
}

Eigen::Affine3f coloradar::ColoradarDataset::getBaseToCascadeRadarTransform() {
    std::filesystem::path baseToRadarTransformPath = transformsDirPath / "base_to_cascade.txt";
    Eigen::Affine3f transform = loadTransform(baseToRadarTransformPath);
    return transform;
}

std::vector<std::string> coloradar::ColoradarDataset::listRuns() {
    std::vector<std::string> runs;
    for (const auto& entry : std::filesystem::directory_iterator(runsDirPath)) {
        if (entry.is_directory()) {
            runs.push_back(entry.path().filename().string());
        }
    }
    return runs;
}

coloradar::ColoradarRun coloradar::ColoradarDataset::getRun(const std::string& runName) {
    std::filesystem::path runPath = runsDirPath / runName;
    return ColoradarRun(runPath);
}

Eigen::Affine3f coloradar::ColoradarDataset::loadTransform(const std::filesystem::path& filePath) {
    coloradar::internal::checkPathExists(filePath);
    Eigen::Affine3f transform = Eigen::Affine3f::Identity();
    Eigen::Vector3f translation;
    Eigen::Quaternionf rotation;

    std::ifstream file(filePath);
    std::string line;
    std::getline(file, line);
    std::istringstream iss(line);
    iss >> translation.x() >> translation.y() >> translation.z();

    std::getline(file, line);
    iss.str(line);
    iss.clear();
    iss >> rotation.x() >> rotation.y() >> rotation.z() >> rotation.w();

    transform.translate(translation);
    transform.rotate(rotation);
    return transform;
}

void coloradar::ColoradarDataset::createMaps(const double& mapResolution, const float& lidarTotalHorizontalFov, const float& lidarTotalVerticalFov, const float& lidarMaxRange, const std::vector<std::string>& targetRuns) {
    std::vector<std::string> runNames = targetRuns.empty() ? listRuns() : targetRuns;
    Eigen::Affine3f transform = getBaseToLidarTransform();

    for (size_t i = 0; i < targetRuns.size(); ++i) {
        coloradar::ColoradarRun run = getRun(targetRuns[i]);
        octomap::OcTree tree = run.buildLidarOctomap(mapResolution, lidarTotalHorizontalFov, lidarTotalVerticalFov, lidarMaxRange, transform);
        run.saveLidarOctomap(tree);
    }
}
