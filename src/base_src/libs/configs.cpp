#include "coloradar_tools.h"

#include <json/json.h>
#include <map>
#include <sstream>
#include <regex>


std::map<std::string, std::vector<std::string>> readConfig(const std::filesystem::path& configFile) {
    coloradar::internal::checkPathExists(configFile);
    std::ifstream infile(configFile);
    if (!infile.is_open()) {
        throw std::runtime_error("Unable to open config file: " + configFile.string());
    }
    std::map<std::string, std::vector<std::string>> configMap;
    std::string line;
    std::regex keyValueRegex(R"(\s*([^#\s]+)\s*[:=\s]\s*(.*))");
    const size_t maxKeySearchLength = 256;

    while (std::getline(infile, line)) {
        if (line.empty() || line.find_first_not_of(" \t") == std::string::npos || line.find('#') == 0) {
            continue;
        }
        std::string lineHead = line.substr(0, maxKeySearchLength);
        std::smatch match;
        if (std::regex_match(lineHead, match, keyValueRegex)) {
            std::string key = match[1];
            size_t valueStartPos = match.position(2);
            std::string valueSection = line.substr(valueStartPos);
            std::istringstream valuesStream(valueSection);
            std::string value;
            std::vector<std::string> values;
            while (valuesStream >> value) {
                values.push_back(value);
            }
            configMap[key] = values;
        }
    }
    infile.close();
    return configMap;
}

Json::Value readJsonConfig(const std::filesystem::path& configFile) {
    if (!std::filesystem::exists(configFile)) {
        throw std::runtime_error("Config file does not exist: " + configFile.string());
    }
    std::ifstream file(configFile);
    if (!file.is_open()) {
        throw std::runtime_error("Unable to open config file: " + configFile.string());
    }
    Json::Value root;
    file >> root;
    return root;
}


void coloradar::RadarConfig::initAntennaParams(const std::filesystem::path& antennaCfgFile) {
    auto configMap = readConfig(antennaCfgFile);
    bool hasNumRx = false, hasNumTx = false, hasDesignF = false;
    rxCenters.clear();
    txCenters.clear();

    std::fstream cfg_file(antennaCfgFile, std::iostream::in);
    std::string line;
    int rx_count = 0;
    int tx_count = 0;
    while (std::getline(cfg_file,line))
    {
      std::stringstream ss(line);
      std::string token;
      std::getline(ss, token, ' ');
      if (token.compare("num_rx") == 0)
      {
        std::getline(ss, token, ' ');
        numRxAntennas = std::stoi(token);
        rxCenters.resize(numRxAntennas);
        hasNumRx = true;
      }
      if (token.compare("num_tx") == 0)
      {
        std::getline(ss, token, ' ');
        numTxAntennas = std::stoi(token);
        txCenters.resize(numTxAntennas);
        hasNumTx = true;
      }
      if (token.compare("F_design") == 0)
      {
        std::getline(ss, token, ' ');
        designFrequency = std::stoi(token) * 1e9;  // convert from GHz to Hz
        hasDesignF = true;
      }
      if (token.compare("rx") == 0)
      {
        std::string token;
        std::getline(ss, token, ' ');
        int idx = std::stoi(token);
        std::getline(ss, token, ' ');
        double x = std::stod(token);
        std::getline(ss, token, ' ');
        double y = std::stod(token);
        pcl::PointXY rx_center (x, y);
        rxCenters[idx] = rx_center;
        rx_count++;
      }
      if (token.compare("tx") == 0)
      {
        std::string token;
        std::getline(ss, token, ' ');
        int idx = std::stoi(token);
        std::getline(ss, token, ' ');
        double x = std::stod(token);
        std::getline(ss, token, ' ');
        double y = std::stod(token);
        pcl::PointXY tx_center (x, y);
        txCenters[idx] = tx_center;
        tx_count++;
      }
    }
    if (!hasNumRx || !hasNumTx || !hasDesignF) {
        throw std::runtime_error("Missing num_rx or num_tx or F_design in antenna config.");
    }
    if (rx_count != rxCenters.size())
    {
      throw std::runtime_error("antenna config specified num_rx = " + std::to_string(rxCenters.size()) + " but only " + std::to_string(rx_count) + " rx positions found.");
    }
    if (tx_count != txCenters.size())
    {
      throw std::runtime_error("antenna config specified num_tx = " + std::to_string(txCenters.size()) + " but only " + std::to_string(tx_count) + " tx positions found.");
    }
}

void coloradar::RadarConfig::initHeatmapParams(const std::filesystem::path& heatmapCfgFile) {
    auto configMap = readConfig(heatmapCfgFile);
    azimuthBins.clear();
    elevationBins.clear();

    auto it = configMap.find("num_range_bins");
    if (it != configMap.end()) {
        numRangeBins = std::stoi(it->second[0]);
    } else {
        throw std::runtime_error("Missing num_range_bins in heatmap config.");
    }
    it = configMap.find("num_elevation_bins");
    if (it != configMap.end()) {
        numElevationBins = std::stoi(it->second[0]);
    }
    it = configMap.find("num_azimuth_bins");
    if (it != configMap.end()) {
        numAzimuthBins = std::stoi(it->second[0]);
    }
    it = configMap.find("range_bin_width");
    if (it != configMap.end()) {
        rangeBinWidth = std::stod(it->second[0]);
    }
    it = configMap.find("azimuth_bins");
    if (it != configMap.end()) {
        for (const auto& bin : it->second) {
            azimuthBins.push_back(std::stod(bin));
        }
    }
    it = configMap.find("elevation_bins");
    if (it != configMap.end()) {
        for (const auto& bin : it->second) {
            elevationBins.push_back(std::stod(bin));
        }
    }
}


void coloradar::RadarConfig::initWaveformParams(const std::filesystem::path& waveformCfgFile) {
    auto configMap = readConfig(waveformCfgFile);

    auto it = configMap.find("num_adc_samples_per_chirp");
    if (it != configMap.end()) {
        numAdcSamplesPerChirp = std::stoi(it->second[0]);
    } else {
        throw std::runtime_error("Missing num_adc_samples_per_chirp in waveform config.");
    }
    it = configMap.find("num_chirps_per_frame");
    if (it != configMap.end()) {
        numChirpsPerFrame = std::stoi(it->second[0]);
    }
    it = configMap.find("adc_sample_frequency");
    if (it != configMap.end()) {
        adcSampleFrequency = std::stod(it->second[0]);
    }
    it = configMap.find("start_frequency");
    if (it != configMap.end()) {
        startFrequency = std::stod(it->second[0]);
    }
    it = configMap.find("idle_time");
    if (it != configMap.end()) {
        idleTime = std::stod(it->second[0]);
    }
    it = configMap.find("adc_start_time");
    if (it != configMap.end()) {
        adcStartTime = std::stod(it->second[0]);
    }
    it = configMap.find("ramp_end_time");
    if (it != configMap.end()) {
        rampEndTime = std::stod(it->second[0]);
    }
    it = configMap.find("frequency_slope");
    if (it != configMap.end()) {
        frequencySlope = std::stod(it->second[0]);
    }
}

void coloradar::RadarConfig::initCouplingParams(const std::filesystem::path& couplingCfgFile) {
    std::ifstream file(couplingCfgFile);
    if (!file.is_open())
        throw std::runtime_error("Unable to open coupling config file: " + couplingCfgFile.string());
    std::string line, token;
    std::stringstream ss;
    couplingCalibMatrix.clear();

    while (std::getline(file, line)) {
        ss.clear();
        ss.str(line);
        std::getline(ss, token, ':');
        if (token == "data") {
            std::vector<double> values;
            while (std::getline(ss, token, ','))
                values.push_back(std::stod(token));
            size_t expectedSize = numTxAntennas * numRxAntennas * numRangeBins * 2;
            if (values.size() != expectedSize)
                throw std::runtime_error("Mismatch in the size of the coupling calibration matrix. Expected: " + std::to_string(expectedSize) + ", Found: " + std::to_string(values.size()));
            for (size_t i = 0; i < values.size(); i += 2)
                couplingCalibMatrix.push_back(std::complex<double>(values[i], values[i + 1]));
        }
    }
    file.close();
    if (couplingCalibMatrix.empty())
        throw std::runtime_error("Coupling calibration data not found in config.");
}


//void coloradar::RadarConfig::initCouplingParams(const std::filesystem::path& couplingCfgFile) {
//    auto configMap = readConfig(couplingCfgFile);
//    couplingCalibMatrix.clear();
//
//    auto it = configMap.find("num_doppler_bins");
//    if (it != configMap.end()) {
//        numDopplerBins = std::stoi(it->second[0]);
//    } else {
//        throw std::runtime_error("Missing num_doppler_bins in coupling config.");
//    }
//    it = configMap.find("data");
//    if (it != configMap.end()) {
//        for (const auto& entry : it->second) {
//            size_t pos;
//            double real = std::stod(entry, &pos);
//            double imag = std::stod(entry.substr(pos + 1));
//            couplingCalibMatrix.push_back(std::complex<double>(real, imag));
//        }
//    } else {
//        throw std::runtime_error("Missing coupling calibration data in config.");
//    }
//}

void coloradar::RadarConfig::initPhaseFrequencyParams(const std::filesystem::path& phaseFrequencyCfgFile) {
    const Json::Value& configMap = readJsonConfig(phaseFrequencyCfgFile);
    frequencyCalibMatrix.clear();
    phaseCalibMatrix.clear();
    if (!configMap.isMember("antennaCalib")) {
        throw std::runtime_error("Missing antennaCalib in phase frequency config.");
    }
    const Json::Value& config = configMap["antennaCalib"];

    if (config.isMember("frequencyCalibrationMatrix")) {
        const Json::Value& frequencyMatrix = config["frequencyCalibrationMatrix"];
        for (const auto& value : frequencyMatrix) {
            frequencyCalibMatrix.push_back(value.asDouble());
        }
    } else {
        throw std::runtime_error("Missing frequencyCalibrationMatrix in phase frequency config.");
    }
    if (config.isMember("phaseCalibrationMatrix")) {
        const Json::Value& phaseMatrix = config["phaseCalibrationMatrix"];
        if (phaseMatrix.size() % 2 != 0) {
            throw std::runtime_error("Invalid phaseCalibrationMatrix: Expecting pairs of real and imaginary values.");
        }

        for (Json::ArrayIndex i = 0; i < phaseMatrix.size(); i += 2) {
            double real = phaseMatrix[i].asDouble();
            double imag = phaseMatrix[i + 1].asDouble();
            phaseCalibMatrix.push_back(std::complex<double>(real, imag));
        }
    } else {
        throw std::runtime_error("Missing phaseCalibrationMatrix in phase frequency config.");
    }
}

void coloradar::RadarConfig::initInternalParams() {
    numAzimuthBeams = 64;
    numElevationBeams = 8;
    azimuthApertureLen = 0;
    elevationApertureLen = 0;

    azimuthAngles.clear();
    elevationAngles.clear();
    azimuthAngles.resize(numAzimuthBeams);
    elevationAngles.resize(numElevationBeams);

    numPosRangeBins = numRangeBins / 2;
    numAngles = numAzimuthBeams * numElevationBeams;

    std::cout << "start RadarConfig::initInternalParams" << std::endl;
    std::cout << "numVirtualElements " << numVirtualElements << std::endl;
    if (virtualArrayMap != nullptr) {
        std::cout << "virtualArrayMap != nullptrs" << std::endl;
        delete[] virtualArrayMap;
        virtualArrayMap = nullptr;
    }
    std::cout << "cleared arrays in RadarConfig::initInternalParams" << std::endl;

//    std::vector<pcl::PointXY> tx_centers_reordered(config->numTxAntennas);
//    for (int tx_idx = 0; tx_idx < config->numTxAntennas; tx_idx++)
//      tx_centers_reordered[tx_idx] = tx_centers[radar_msg->tx_order[tx_idx]];
//    tx_centers = tx_centers_reordered;
    numVirtualElements = 0;
    std::vector<int> virtual_map_local;
    for (int tx_idx = 0; tx_idx < numTxAntennas; tx_idx++)
    {
      for (int rx_idx = 0; rx_idx < numRxAntennas; rx_idx++)
      {
        int virtual_x = rxCenters[rx_idx].x + txCenters[tx_idx].x;
        int virtual_y = rxCenters[rx_idx].y + txCenters[tx_idx].y;

        // check to ensure this antenna pair doesn't map to the same virtual
        // location as a previously evaluated antenna pair
        bool redundant = false;
        for (int i = 0; i < numVirtualElements; i++)
        {
          int idx = i * 4;
          if (virtual_map_local[idx] == virtual_x
            && virtual_map_local[idx+1] == virtual_y)
            redundant = true;
        }
        // record mapping from antenna pair index to virtual antenna location
        // stored in vector with entries grouped into 4-tuples of
        // [azimuth_location, elevation_location, rx_index, tx_index]
        if (!redundant)
        {
          if (virtual_x + 1 > azimuthApertureLen)
            azimuthApertureLen = virtual_x + 1;
          if (virtual_y + 1 > elevationApertureLen)
            elevationApertureLen = virtual_y + 1;

          virtual_map_local.push_back(virtual_x);
          virtual_map_local.push_back(virtual_y);
          virtual_map_local.push_back(rx_idx);
          virtual_map_local.push_back(tx_idx);
          numVirtualElements++;
        }
      }
    }
    virtualArrayMap = new int[virtual_map_local.size()];
    std::copy(virtual_map_local.begin(), virtual_map_local.end(), virtualArrayMap);

    std::cout << "init virtualArrayMap in RadarConfig::initInternalParams" << std::endl;

    double wavelength = c / (startFrequency + adcStartTime * frequencySlope);
    double chirp_time = idleTime + rampEndTime;
    double v_max = wavelength / (8.0 * numTxAntennas * chirp_time);
    dopplerBinWidth = v_max / numDopplerBins;

    double center_frequency = startFrequency + numRangeBins / adcSampleFrequency * frequencySlope / 2.0;
    double d = 0.5 * center_frequency / designFrequency; // unit antenna distance in wavelengths
    double az_d_phase = (2. * M_PI) / numAzimuthBeams;
    double phase_dif = (az_d_phase / 2.) - M_PI;
    for (int i = 0; i < numAzimuthBeams; i++)
    {
      azimuthAngles[i] = asin(phase_dif / (2. * M_PI * d));
      phase_dif += az_d_phase;
    }
    double el_d_phase = (2.*M_PI) / numElevationBeams;
    phase_dif = (el_d_phase / 2.) - M_PI;
    for(int i = 0; i < numElevationBeams; i++)
    {
      elevationAngles[i] = asin(phase_dif / (2. * M_PI * d));
      phase_dif += el_d_phase;
    }
}

coloradar::RadarConfig::~RadarConfig() {
    if (virtualArrayMap != nullptr) {
        delete[] virtualArrayMap;
        virtualArrayMap = nullptr;
    }
}


coloradar::SingleChipConfig::SingleChipConfig(const std::filesystem::path& calibDir) {
    coloradar::internal::checkPathExists(calibDir);
    init(calibDir);
}

void coloradar::SingleChipConfig::init(const std::filesystem::path& calibDir) {
    std::cout << "start SingleChipConfig::init" << std::endl;
    std::filesystem::path configDir = calibDir / "single_chip";
    coloradar::internal::checkPathExists(configDir);
    std::filesystem::path antennaConfigFilePath = configDir / "antenna_cfg.txt";
    std::filesystem::path heatmapConfigFilePath = configDir / "heatmap_cfg.txt";
    std::filesystem::path waveformConfigFilePath = configDir / "waveform_cfg.txt";
    std::filesystem::path couplingConfigFilePath = configDir / "coupling_calib.txt";
    initAntennaParams(antennaConfigFilePath);
    initHeatmapParams(heatmapConfigFilePath);
    initWaveformParams(waveformConfigFilePath);
    initCouplingParams(couplingConfigFilePath);
    initInternalParams();
    std::cout << "finish SingleChipConfig::init" << std::endl;
}

coloradar::CascadeConfig::CascadeConfig(const std::filesystem::path& calibDir) {
    coloradar::internal::checkPathExists(calibDir);
    init(calibDir);
}

void coloradar::CascadeConfig::init(const std::filesystem::path& calibDir) {
    std::cout << "start CascadeConfig::init" << std::endl;
    std::filesystem::path configDir = calibDir / "cascade";
    coloradar::internal::checkPathExists(configDir);
    std::filesystem::path antennaConfigFilePath = configDir / "antenna_cfg.txt";
    std::filesystem::path heatmapConfigFilePath = configDir / "heatmap_cfg.txt";
    std::filesystem::path waveformConfigFilePath = configDir / "waveform_cfg.txt";
    std::filesystem::path couplingConfigFilePath = configDir / "coupling_calib.txt";
    std::filesystem::path phaseFrequencyConfigFilePath = configDir / "phase_frequency_calib.txt";
    initAntennaParams(antennaConfigFilePath);
    initHeatmapParams(heatmapConfigFilePath);
    initWaveformParams(waveformConfigFilePath);
    initCouplingParams(couplingConfigFilePath);
    initPhaseFrequencyParams(phaseFrequencyConfigFilePath);
    initInternalParams();
    std::cout << "finish CascadeConfig::init" << std::endl;
}
