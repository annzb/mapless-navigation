#include "coloradar_tools.h"


coloradar::RadarConfig::RadarConfig(const std::filesystem::path& calibDir) { init(calibDir); }

void coloradar::RadarConfig::initHeatmapParams(const std::filesystem::path& heatmapCfgFile) {
    coloradar::internal::checkPathExists(heatmapCfgFile);
    std::ifstream infile(heatmapCfgFile);
    if (!infile.is_open()) {
        throw std::runtime_error("Unable to open heatmap config file: " + heatmapCfgFile.string());
    }
    bool hasNumRangeBins = false, hasNumElevationBins = false;
    bool hasNumAzimuthBins = false;
    bool hasRangeBinWidth = false;
    bool hasAzimuthBins = false;
    bool hasElevationBins = false;
    std::string key;

    while (infile >> key) {
        if (key == "num_range_bins") {
            infile >> numRangeBins;
            hasNumRangeBins = true;
        } else if (key == "num_elevation_bins") {
            infile >> numElevationBins;
            hasNumElevationBins = true;
        } else if (key == "num_azimuth_bins") {
            infile >> numAzimuthBins;
            hasNumAzimuthBins = true;
        } else if (key == "range_bin_width") {
            infile >> rangeBinWidth;
            hasRangeBinWidth = true;
        } else if (key == "azimuth_bins") {
            azimuthBins.clear();
            double binValue;
            while (infile.peek() != '\n' && infile >> binValue) {
                azimuthBins.push_back(binValue);
            }
            hasAzimuthBins = true;
        } else if (key == "elevation_bins") {
            elevationBins.clear();
            double binValue;
            while (infile.peek() != '\n' && infile >> binValue) {
                elevationBins.push_back(binValue);
            }
            hasElevationBins = true;
        } else {
            throw std::runtime_error("Unknown key in config file: " + key);
        }
    }
    infile.close();

    if (!hasNumRangeBins || !hasNumElevationBins || !hasNumAzimuthBins || !hasRangeBinWidth || !hasAzimuthBins || !hasElevationBins) {
        throw std::runtime_error("Missing required parameters in the heatmap config file.");
    }
    if (azimuthBins.size() != static_cast<size_t>(numAzimuthBins)) {
        throw std::runtime_error("Mismatch in number of azimuth bins. Expected: " + std::to_string(numAzimuthBins) + ", Found: " + std::to_string(azimuthBins.size()));
    }
    if (elevationBins.size() != static_cast<size_t>(numElevationBins)) {
        throw std::runtime_error("Mismatch in number of elevation bins. Expected: " + std::to_string(numElevationBins) + ", Found: " + std::to_string(elevationBins.size()));
    }
}


