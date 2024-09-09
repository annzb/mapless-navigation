#include "coloradar_tools.h"

#include <random>
#include <pcl/io/pcd_io.h>
#include <filesystem>
#include <stdexcept>


namespace fs = std::filesystem;


void createDirectoryIfNotExists(const fs::path& dirPath) {
    if (!fs::exists(dirPath)) {
        fs::create_directories(dirPath);
    }
}

std::unordered_map<std::string, std::string> parseArguments(int argc, char** argv) {
    std::unordered_map<std::string, std::string> arguments;
    for (int i = 1; i < argc; ++i) {
        std::string arg(argv[i]);
        if (arg.find("=") != std::string::npos) {
            auto pos = arg.find("=");
            std::string key = arg.substr(0, pos);
            std::string value = arg.substr(pos + 1);
            arguments[key] = value;
        }
    }
    return arguments;
}


bool generateRandomEmptySpace(float probability) {
    static std::mt19937 gen(42);  // seed
    static std::uniform_real_distribution<> dis(0.0, 1.0);
    return dis(gen) < probability;
}

coloradar::OctoPointcloud generateSpherePointCloud(float radius, float step, float emptySpaceProbability) {
    coloradar::OctoPointcloud cloud;
    for (float x = -radius; x <= radius; x += step) {
        for (float y = -radius; y <= radius; y += step) {
            for (float z = -radius; z <= radius; z += step) {
                if (x*x + y*y + z*z <= radius * radius) {
                    if (!generateRandomEmptySpace(emptySpaceProbability)) {
                        cloud.push_back(octomap::point3d(x, y, z));
                    }
                }
            }
        }
    }
    return cloud;
}


int main(int argc, char** argv) {
    auto args = parseArguments(argc, argv);
    std::string pcdFile = args.find("pcdFilePath") != args.end() ? args["pcdFilePath"] : "";
    std::string outputDir = args.find("outputDir") != args.end() ? args["outputDir"] : "";
    float randomPclRadius = args.find("randomPclRadius") != args.end() ? std::stod(args["randomPclRadius"]) : 10.0;
    float randomPclStep = args.find("randomPclStep") != args.end() ? std::stod(args["randomPclStep"]) : 0.5;
    float randomPclEmptyPortion = args.find("randomPclEmptyPortion") != args.end() ? std::stod(args["randomPclEmptyPortion"]) : 0.5;
    float verticalFov = args.find("verticalFov") != args.end() ? std::stod(args["verticalFov"]) : 180.0;
    float horizontalFov = args.find("horizontalFov") != args.end() ? std::stod(args["horizontalFov"]) : 360.0;
    float maxRange = args.find("range") != args.end() ? std::stod(args["range"]) : 0.0;

    fs::path outputDirPath;
    coloradar::OctoPointcloud octoCloud;

    if (!outputDir.empty()) {
        outputDirPath = fs::path(outputDir);
        createDirectoryIfNotExists(outputDirPath);
    }
    if (!pcdFile.empty()) {
        std::string validExtension = ".pcd";
        fs::path pcdFilePath(pcdFile);
        std::string fileExtension = pcdFilePath.extension();
        if (fileExtension != validExtension) {
            throw std::runtime_error("Invalid pcdFilePath: expected ending with " + validExtension + ", got " + fileExtension);
        }
        if (!fs::exists(pcdFilePath)) {
            throw std::runtime_error("File not found " + pcdFile);
        }
        outputDirPath = pcdFilePath.parent_path();

        pcl::PointCloud<pcl::PointXYZ> cloud;
        pcl::io::loadPCDFile<pcl::PointXYZ>(pcdFilePath.string(), cloud);
        octoCloud = coloradar::OctoPointcloud(cloud);
    } else {
        if (outputDirPath.empty()) {
            std::cerr << "Either pcdFilePath or outputDir expected." << std::endl;
            std::cerr << "Usage: " << argv[0] << " [pcdFilePath=<str>.pcd] [outputDir=<str>] [randomPclRadius=<meters>] [randomPclStep=<meters>] [randomPclEmptyPortion=<meters>] [verticalFov=<degrees>] [horizontalFov=<degrees>] [range=<meters>]" << std::endl;
            return -1;
        }
        octoCloud = generateSpherePointCloud(randomPclRadius, randomPclStep, randomPclEmptyPortion);
    }
    pcl::io::savePCDFile(outputDirPath / "original_cloud.pcd", octoCloud.toPcl());

    float range = maxRange == 0 ? std::numeric_limits<float>::max() : maxRange;
    octoCloud.filterFov(horizontalFov, verticalFov, range);
    pcl::io::savePCDFile(outputDirPath / "filtered_cloud.pcd", octoCloud.toPcl());
}